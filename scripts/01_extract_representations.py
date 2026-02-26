#!/usr/bin/env python3
"""Extract hidden representations from multimodal models for modality collapse analysis.

Loads a specified multimodal model (Ultravox, Qwen2-Audio, or LLaVA), feeds
dataset samples through it one at a time, hooks into encoder, adapter, and LLM
layers, and stores mean-pooled representations plus labels in an HDF5 file.

Usage:
    UV_PROJECT_ENVIRONMENT=~/venvs/modality_collapse uv run python scripts/01_extract_representations.py \
        --model ultravox --dataset librispeech --device cuda:0

    UV_PROJECT_ENVIRONMENT=~/venvs/modality_collapse uv run python scripts/01_extract_representations.py \
        --model llava --dataset coco --device cuda:0

    UV_PROJECT_ENVIRONMENT=~/venvs/modality_collapse uv run python scripts/01_extract_representations.py \
        --model qwen2audio --dataset cremad --config configs/experiment1.yaml
"""

from __future__ import annotations

import modality_collapse.utils.env  # noqa: F401  # set HF_HOME early

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────────
from modality_collapse.models import (
    UltravoxExtractor,
    Qwen2AudioExtractor,
    LlavaExtractor,
    get_model_config,
)
try:
    from modality_collapse.models import PrismaticExtractor
except ImportError:
    PrismaticExtractor = None
from modality_collapse.extraction.hooks import RepresentationExtractor
from modality_collapse.extraction.storage import create_hdf5, append_to_hdf5
from modality_collapse.data.speech import (
    LibriSpeechLoader,
    CREMADLoader,
    ESC50Loader,
    VoxCeleb1Loader,
    SpeechSample,
)
from modality_collapse.data.vision import COCOLoader, GQALoader, VisionSample

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_representations")

# ── constants ────────────────────────────────────────────────────────────────
SUPPORTED_MODELS = ("ultravox", "qwen2audio", "llava", "prismatic_dinov2", "prismatic_siglip")
SUPPORTED_DATASETS = (
    "librispeech",
    "cremad",
    "esc50",
    "voxceleb1",
    "coco",
    "gqa",
)
SPEECH_DATASETS = {"librispeech", "cremad", "esc50", "voxceleb1"}
VISION_DATASETS = {"coco", "gqa"}

VRAM_LOG_INTERVAL = 100  # print VRAM usage every N samples


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _load_config(path: str | None) -> dict:
    """Load YAML config, or return an empty dict when *path* is ``None``."""
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


def _get_extractor(model_name: str, device: torch.device, checkpoint: str | None = None):
    """Instantiate the appropriate model extractor."""
    if model_name == "ultravox":
        return UltravoxExtractor(device=device, checkpoint_path=checkpoint)
    if model_name == "qwen2audio":
        return Qwen2AudioExtractor(device=device)
    if model_name == "llava":
        return LlavaExtractor(device=device)
    if model_name == "prismatic_dinov2":
        if PrismaticExtractor is None:
            raise ImportError("PrismaticExtractor requires the prismatic package. "
                              "Use ~/venvs/prismatic venv.")
        return PrismaticExtractor(device=device, variant="dinov2-224px+7b")
    if model_name == "prismatic_siglip":
        if PrismaticExtractor is None:
            raise ImportError("PrismaticExtractor requires the prismatic package. "
                              "Use ~/venvs/prismatic venv.")
        return PrismaticExtractor(device=device, variant="siglip-224px+7b")
    raise ValueError(f"Unsupported model: {model_name}")


def _build_dataset(dataset_name: str, config: dict):
    """Instantiate the appropriate dataset loader from *config*."""
    ds_cfg: dict = {}
    # Try to pull dataset-specific config from the YAML.
    if config:
        speech_cfg = config.get("datasets", {}).get("speech", {})
        vision_cfg = config.get("datasets", {}).get("vision", {})
        ds_cfg = speech_cfg.get(dataset_name, {}) or vision_cfg.get(dataset_name, {})
    ds_cfg = dict(ds_cfg)  # shallow copy so we can pop

    if dataset_name == "librispeech":
        return LibriSpeechLoader(
            split=ds_cfg.get("split", "test"),
            max_samples=ds_cfg.get("max_samples", 2620),
        )
    if dataset_name == "cremad":
        return CREMADLoader(
            data_dir=ds_cfg.get("data_dir", "data/raw/CREMA-D"),
            max_samples=ds_cfg.get("max_samples"),
        )
    if dataset_name == "esc50":
        return ESC50Loader(
            data_dir=ds_cfg.get("data_dir", "data/raw/ESC-50"),
            max_samples=ds_cfg.get("max_samples"),
        )
    if dataset_name == "voxceleb1":
        return VoxCeleb1Loader(
            data_dir=ds_cfg.get("data_dir", "data/raw/VoxCeleb1"),
            n_speakers=ds_cfg.get("n_speakers", 40),
            samples_per_speaker=ds_cfg.get("samples_per_speaker", 100),
            max_samples=ds_cfg.get("max_samples", 4000),
        )
    if dataset_name == "coco":
        return COCOLoader(
            data_dir=ds_cfg.get("data_dir", "data/raw/COCO"),
            split=ds_cfg.get("split", "val2017"),
            max_samples=ds_cfg.get("max_samples", 5000),
        )
    if dataset_name == "gqa":
        return GQALoader(
            data_dir=ds_cfg.get("data_dir", "data/raw/GQA"),
            split=ds_cfg.get("split", "val"),
            max_samples=ds_cfg.get("max_samples", 5000),
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _preprocess_sample(extractor, sample):
    """Run the extractor's preprocessing for a single sample.

    Returns a dict of tensors ready for ``model(**inputs)``.
    """
    if isinstance(sample, SpeechSample):
        return extractor.preprocess(sample.audio, sample.sample_rate)
    if isinstance(sample, VisionSample):
        return extractor.preprocess(sample.image)
    raise TypeError(f"Unknown sample type: {type(sample)}")


def _log_vram(step: int) -> None:
    """Print current CUDA VRAM usage (no-op if CUDA is unavailable)."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    logger.info(
        "VRAM at step %d: %.2f GB allocated, %.2f GB reserved", step, alloc, reserved
    )


def _collect_label(sample) -> dict[str, str | None]:
    """Extract all available label fields from a sample."""
    if isinstance(sample, SpeechSample):
        return {
            "transcript": sample.transcript,
            "speaker_id": sample.speaker_id,
            "emotion": sample.emotion,
            "sound_class": sample.sound_class,
        }
    if isinstance(sample, VisionSample):
        return {
            "caption": sample.caption,
            "question": sample.question,
            "answer": sample.answer,
            "object_category": sample.object_category,
            "super_category": sample.super_category,
            "question_type": sample.question_type,
        }
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Main extraction loop
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract representations from multimodal models."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=SUPPORTED_MODELS,
        help="Model to extract from.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset to process.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device (default: cuda:0).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML experiment config (optional).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to fine-tuned checkpoint (e.g. LoRA). Currently Ultravox only.",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Override model name in output filename (e.g. 'ultravox_lora').",
    )
    args = parser.parse_args()

    # ── sanity checks ────────────────────────────────────────────────────
    model_name: str = args.model
    dataset_name: str = args.dataset
    device = torch.device(args.device)
    config = _load_config(args.config)

    model_cfg = get_model_config(model_name)

    # Validate modality / dataset compatibility.
    if dataset_name in SPEECH_DATASETS and model_cfg.modality == "vision":
        logger.error(
            "Model '%s' is a vision model but dataset '%s' is speech.",
            model_name,
            dataset_name,
        )
        sys.exit(1)
    if dataset_name in VISION_DATASETS and model_cfg.modality == "speech":
        logger.error(
            "Model '%s' is a speech model but dataset '%s' is vision.",
            model_name,
            dataset_name,
        )
        sys.exit(1)

    output_dir = Path(config.get("extraction", {}).get("output_dir", "data/representations"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_tag = args.output_tag or model_name
    output_path = output_dir / f"{output_tag}_{dataset_name}.h5"

    logger.info("Model:   %s  (%s)", model_name, model_cfg.hf_path)
    logger.info("Dataset: %s", dataset_name)
    logger.info("Device:  %s", device)
    logger.info("Output:  %s", output_path)

    # ── load model ───────────────────────────────────────────────────────
    logger.info("Loading model ...")
    extractor = _get_extractor(model_name, device, checkpoint=args.checkpoint)
    model, processor = extractor.load()
    logger.info("Model loaded.")

    # ── load dataset ─────────────────────────────────────────────────────
    logger.info("Loading dataset ...")
    dataset = _build_dataset(dataset_name, config)
    n_samples = len(dataset)
    logger.info("Dataset ready: %d samples.", n_samples)

    # ── resolve hook points ──────────────────────────────────────────────
    hook_points: dict[str, str] = dict(model_cfg.hook_points)
    hook_names = list(hook_points.keys())
    logger.info("Hook points: %s", hook_names)

    # ── first-pass: run one sample to discover hidden dims ───────────────
    logger.info("Running probe forward pass to discover hidden dimensions ...")
    sample_iter = iter(dataset)
    probe_sample = next(sample_iter)
    probe_inputs = _preprocess_sample(extractor, probe_sample)

    rep_extractor = RepresentationExtractor(model, hook_points, pool_strategy="mean")
    with torch.no_grad():
        rep_extractor.register_hooks()
        try:
            model(**probe_inputs)
        finally:
            rep_extractor.remove_hooks()
    probe_acts = rep_extractor.get_activations()

    hidden_dims: dict[str, int] = {}
    for name, tensor in probe_acts.items():
        # After mean pooling: (batch=1, hidden_dim) or (hidden_dim,)
        dim = tensor.shape[-1]
        hidden_dims[name] = dim
        logger.info("  %s: hidden_dim = %d", name, dim)

    # ── create HDF5 file ─────────────────────────────────────────────────
    metadata = {
        "model_name": model_name,
        "hf_path": model_cfg.hf_path,
        "dataset_name": dataset_name,
        "hook_points": hook_points,
        "hook_names": hook_names,
        "hidden_dims": hidden_dims,
        "n_samples": n_samples,
        "device": str(device),
        "pool_strategy": "mean",
        "created_at": datetime.utcnow().isoformat(),
        "config": config if config else {},
    }

    create_hdf5(
        path=str(output_path),
        hook_names=hook_names,
        hidden_dims=hidden_dims,
        max_samples=n_samples,
        metadata=metadata,
    )

    # ── create label datasets in HDF5 ───────────────────────────────────
    # Determine which label fields are relevant for this dataset.
    probe_labels = _collect_label(probe_sample)
    label_fields = [k for k, v in probe_labels.items() if v is not None]

    # Create string datasets for sample_ids and each label field.
    dt_vlen_str = h5py.special_dtype(vlen=str)
    with h5py.File(str(output_path), "a") as f:
        f.create_dataset("sample_ids", shape=(n_samples,), dtype=dt_vlen_str)
        for field in label_fields:
            f.create_dataset(field, shape=(n_samples,), dtype=dt_vlen_str)
        f.attrs["label_fields"] = json.dumps(label_fields)

    logger.info("Label fields: %s", label_fields)
    logger.info("HDF5 file created: %s", output_path)

    # ── extraction loop ──────────────────────────────────────────────────
    logger.info("Starting extraction ...")
    t_start = time.time()
    current_idx = 0

    # We already consumed the probe sample, so write it first.
    with h5py.File(str(output_path), "a") as f:
        f["sample_ids"][0] = probe_sample.sample_id if hasattr(probe_sample, "sample_id") else ""
        probe_lbl = _collect_label(probe_sample)
        for field in label_fields:
            val = probe_lbl.get(field)
            f[field][0] = val if val is not None else ""

    for name in hook_names:
        tensor = probe_acts[name]
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        append_to_hdf5(str(output_path), name, tensor.numpy(), current_idx)

    current_idx = 1  # probe sample written at index 0

    rep_extractor_loop = RepresentationExtractor(
        model, hook_points, pool_strategy="mean"
    )

    with rep_extractor_loop:
        for i, sample in enumerate(
            tqdm(sample_iter, total=n_samples - 1, desc="Extracting", initial=0),
        ):
            idx = current_idx  # row in HDF5

            # Preprocess
            inputs = _preprocess_sample(extractor, sample)

            # Forward pass
            with torch.no_grad():
                model(**inputs)

            activations = rep_extractor_loop.get_activations()

            # Write representations
            for name in hook_names:
                tensor = activations[name]
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                append_to_hdf5(str(output_path), name, tensor.numpy(), idx)

            # Write sample ID and labels
            with h5py.File(str(output_path), "a") as f:
                f["sample_ids"][idx] = sample.sample_id if hasattr(sample, "sample_id") else ""
                labels = _collect_label(sample)
                for field in label_fields:
                    val = labels.get(field)
                    f[field][idx] = val if val is not None else ""

            current_idx += 1

            # Periodic VRAM logging
            if (current_idx % VRAM_LOG_INTERVAL) == 0:
                _log_vram(current_idx)

    elapsed = time.time() - t_start

    # ── trim HDF5 if we got fewer samples than expected ──────────────────
    if current_idx < n_samples:
        logger.warning(
            "Expected %d samples but only extracted %d. Trimming HDF5.",
            n_samples,
            current_idx,
        )
        with h5py.File(str(output_path), "a") as f:
            f.attrs["n_samples_actual"] = current_idx

    # ── summary ──────────────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("Extraction complete.")
    logger.info("  Samples extracted: %d", current_idx)
    logger.info("  Time elapsed:      %.1f s (%.2f s/sample)", elapsed, elapsed / max(current_idx, 1))
    logger.info("  Output file:       %s", output_path)
    for name in hook_names:
        logger.info("  %-20s  shape = (%d, %d)", name, current_idx, hidden_dims[name])
    logger.info("  Label fields:      %s", label_fields)
    logger.info("=" * 72)

    # ── cleanup ──────────────────────────────────────────────────────────
    del model, processor, extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract text-baseline representations through Llama-3.1-8B.

Reads the transcripts / captions stored in an HDF5 file produced by
``01_extract_representations.py``, feeds each text through the frozen
Llama-3.1-8B, hooks at ``llm_hidden_16`` and ``llm_final``, and saves
mean-pooled representations to a new HDF5 file.  This gives us the
text-law distribution P_T against which multimodal representations are
compared.

Usage:
    UV_PROJECT_ENVIRONMENT=~/venvs/modality_collapse uv run python scripts/02_extract_text_baseline.py \
        --source-file data/representations/ultravox_librispeech.h5 --device cuda:0

    UV_PROJECT_ENVIRONMENT=~/venvs/modality_collapse uv run python scripts/02_extract_text_baseline.py \
        --source-file data/representations/llava_coco.h5 --device cuda:1

    UV_PROJECT_ENVIRONMENT=~/venvs/modality_collapse uv run python scripts/02_extract_text_baseline.py \
        --source-file data/representations/qwen2audio_cremad.h5 \
        --config configs/experiment1.yaml
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
from modality_collapse.models import TextBaselineExtractor, get_model_config
from modality_collapse.extraction.hooks import RepresentationExtractor
from modality_collapse.extraction.storage import (
    create_hdf5,
    append_to_hdf5,
    load_metadata,
)

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_text_baseline")

# Text baseline uses only these two hook points.
TEXT_HOOK_NAMES = ["llm_hidden_16", "llm_final"]

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


def _read_source_texts(source_path: str) -> tuple[list[str], list[str], dict]:
    """Read sample IDs and text strings from the source HDF5 file.

    The function looks for a text field in this priority order:
      1. ``transcript``  (speech datasets)
      2. ``caption``     (vision datasets like COCO)
      3. ``question``    (VQA datasets like GQA)

    Returns:
        ``(sample_ids, texts, source_metadata)``
    """
    source_metadata = load_metadata(source_path)

    with h5py.File(source_path, "r") as f:
        # Read sample IDs.
        if "sample_ids" not in f:
            raise KeyError(
                f"Source HDF5 '{source_path}' has no 'sample_ids' dataset. "
                "Was it created by 01_extract_representations.py?"
            )
        sample_ids = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in f["sample_ids"][:]]

        # Determine which text field to use.
        text_field: str | None = None
        for candidate in ("transcript", "caption", "question"):
            if candidate in f:
                text_field = candidate
                break

        if text_field is None:
            raise KeyError(
                f"Source HDF5 '{source_path}' has no text field "
                "(tried: transcript, caption, question). "
                "Cannot build text baseline without text data."
            )

        raw_texts = f[text_field][:]
        texts = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in raw_texts]

    logger.info("Using text field '%s' from source file.", text_field)
    source_metadata["_text_field_used"] = text_field
    return sample_ids, texts, source_metadata


def _log_vram(step: int) -> None:
    """Print current CUDA VRAM usage (no-op if CUDA is unavailable)."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    logger.info(
        "VRAM at step %d: %.2f GB allocated, %.2f GB reserved", step, alloc, reserved
    )


def _derive_output_path(source_path: str, output_dir: Path, text_model_name: str = "llama") -> Path:
    """Build the output HDF5 path from the source filename.

    ``ultravox_librispeech.h5`` -> ``llama_from_ultravox_librispeech.h5``
    ``llava_coco.h5`` (vicuna)  -> ``vicuna_from_llava_coco.h5``
    """
    stem = Path(source_path).stem  # e.g. "ultravox_librispeech"
    return output_dir / f"{text_model_name}_from_{stem}.h5"


# ═══════════════════════════════════════════════════════════════════════════════
# Main extraction loop
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text-baseline representations through a frozen LLM."
    )
    parser.add_argument(
        "--source-file",
        required=True,
        help="Path to the HDF5 file produced by 01_extract_representations.py.",
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
        "--text-model",
        default="llama",
        help=(
            "Text baseline model name from registry. Use 'llama' for "
            "standalone Llama-3.1-8B (default). For multimodal models "
            "whose LLM backbone is needed (e.g. 'qwen2audio'), the LLM "
            "backbone is extracted from the full model."
        ),
    )
    args = parser.parse_args()

    source_path: str = args.source_file
    device = torch.device(args.device)
    config = _load_config(args.config)
    text_model_name: str = args.text_model

    if not Path(source_path).is_file():
        logger.error("Source file not found: %s", source_path)
        sys.exit(1)

    # Determine if text model is standalone or extracted from multimodal.
    text_model_cfg = get_model_config(text_model_name)
    is_multimodal = text_model_cfg.modality != "text"

    # ── read source data ─────────────────────────────────────────────────
    logger.info("Reading source file: %s", source_path)
    sample_ids, texts, source_meta = _read_source_texts(source_path)
    n_samples = len(texts)
    source_model = source_meta.get("model_name", "unknown")
    source_dataset = source_meta.get("dataset_name", "unknown")
    logger.info(
        "Source: model=%s, dataset=%s, %d samples.",
        source_model,
        source_dataset,
        n_samples,
    )

    # Filter out empty texts (some datasets may have None/empty entries).
    valid_mask = [bool(t and t.strip()) for t in texts]
    n_valid = sum(valid_mask)
    if n_valid == 0:
        logger.error("No non-empty text entries found in source file.")
        sys.exit(1)
    if n_valid < n_samples:
        logger.warning(
            "%d / %d texts are empty and will be skipped.", n_samples - n_valid, n_samples
        )

    # ── output path ──────────────────────────────────────────────────────
    output_dir = Path(config.get("extraction", {}).get("output_dir", "data/representations"))
    output_dir.mkdir(parents=True, exist_ok=True)
    # Name encodes the text model and source.
    stem = Path(source_path).stem  # e.g. "qwen2audio_librispeech"
    if is_multimodal:
        # e.g. "qwen2audio_llm_from_qwen2audio_librispeech.h5"
        output_path = output_dir / f"{text_model_name}_llm_from_{stem}.h5"
    else:
        output_path = _derive_output_path(source_path, output_dir, text_model_name)

    logger.info("Output: %s", output_path)

    # ── load model ───────────────────────────────────────────────────────
    logger.info("Loading text baseline model: %s (multimodal=%s) ...", text_model_name, is_multimodal)
    text_extractor = TextBaselineExtractor(
        device=device,
        model_name=text_model_name,
        multimodal=is_multimodal,
    )
    model, tokenizer = text_extractor.load()
    logger.info("Model loaded.")

    # Determine hook points: for multimodal models, use the LLM hook
    # points from the multimodal config (e.g. language_model.model.layers.16).
    # For standalone text models, use their own hook points.
    hook_points: dict[str, str] = {
        name: text_model_cfg.hook_points[name]
        for name in TEXT_HOOK_NAMES
        if name in text_model_cfg.hook_points
    }
    hook_names = list(hook_points.keys())
    logger.info("Hook points: %s", hook_names)

    # ── probe forward pass for hidden dims ───────────────────────────────
    logger.info("Running probe forward pass to discover hidden dimensions ...")
    # Find the first valid text for probing.
    probe_text = next(t for t in texts if t and t.strip())
    probe_inputs = text_extractor.preprocess(probe_text)

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
        dim = tensor.shape[-1]
        hidden_dims[name] = dim
        logger.info("  %s: hidden_dim = %d", name, dim)

    # ── create HDF5 file ─────────────────────────────────────────────────
    # For display/metadata: use "qwen2audio_llm" for multimodal, model name otherwise.
    baseline_label = f"{text_model_name}_llm" if is_multimodal else text_model_name
    metadata = {
        "model_name": baseline_label,
        "hf_path": text_model_cfg.hf_path,
        "source_file": str(Path(source_path).resolve()),
        "source_model": source_model,
        "source_dataset": source_dataset,
        "text_field_used": source_meta.get("_text_field_used", "unknown"),
        "hook_points": hook_points,
        "hook_names": hook_names,
        "hidden_dims": hidden_dims,
        "n_samples": n_valid,
        "device": str(device),
        "pool_strategy": "mean",
        "created_at": datetime.utcnow().isoformat(),
        "config": config if config else {},
    }

    create_hdf5(
        path=str(output_path),
        hook_names=hook_names,
        hidden_dims=hidden_dims,
        max_samples=n_valid,
        metadata=metadata,
    )

    # Create string datasets for sample IDs and the source text.
    dt_vlen_str = h5py.special_dtype(vlen=str)
    with h5py.File(str(output_path), "a") as f:
        f.create_dataset("sample_ids", shape=(n_valid,), dtype=dt_vlen_str)
        f.create_dataset("text", shape=(n_valid,), dtype=dt_vlen_str)

    logger.info("HDF5 file created: %s", output_path)

    # ── extraction loop ──────────────────────────────────────────────────
    logger.info("Starting extraction (%d valid texts) ...", n_valid)
    t_start = time.time()
    write_idx = 0

    rep_extractor_loop = RepresentationExtractor(
        model, hook_points, pool_strategy="mean"
    )

    with rep_extractor_loop:
        for i in tqdm(range(n_samples), desc="Extracting", total=n_samples):
            if not valid_mask[i]:
                continue

            text = texts[i]
            sid = sample_ids[i]

            # Tokenize
            inputs = text_extractor.preprocess(text)

            # Forward pass
            with torch.no_grad():
                model(**inputs)

            activations = rep_extractor_loop.get_activations()

            # Write representations
            for name in hook_names:
                tensor = activations[name]
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                append_to_hdf5(str(output_path), name, tensor.numpy(), write_idx)

            # Write sample ID and source text
            with h5py.File(str(output_path), "a") as f:
                f["sample_ids"][write_idx] = sid
                f["text"][write_idx] = text

            write_idx += 1

            # Periodic VRAM logging
            if (write_idx % VRAM_LOG_INTERVAL) == 0:
                _log_vram(write_idx)

    elapsed = time.time() - t_start

    # ── trim if needed ───────────────────────────────────────────────────
    if write_idx < n_valid:
        logger.warning(
            "Expected %d valid samples but only wrote %d. Updating metadata.",
            n_valid,
            write_idx,
        )
        with h5py.File(str(output_path), "a") as f:
            f.attrs["n_samples_actual"] = write_idx

    # ── summary ──────────────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("Text baseline extraction complete.")
    logger.info("  Source file:       %s", source_path)
    logger.info("  Source model:      %s", source_model)
    logger.info("  Source dataset:    %s", source_dataset)
    logger.info("  Samples extracted: %d", write_idx)
    logger.info("  Time elapsed:      %.1f s (%.2f s/sample)", elapsed, elapsed / max(write_idx, 1))
    logger.info("  Output file:       %s", output_path)
    for name in hook_names:
        logger.info("  %-20s  shape = (%d, %d)", name, write_idx, hidden_dims[name])
    logger.info("=" * 72)

    # ── cleanup ──────────────────────────────────────────────────────────
    del model, tokenizer, text_extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

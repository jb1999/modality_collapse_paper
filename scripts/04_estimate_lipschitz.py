#!/usr/bin/env python3
"""Estimate L_log -- the Lipschitz constant of the LLM's log-likelihood.

For each sample, we compute the gradient of log q(y|z,c) with respect to
the adapter output z and take its Frobenius norm as a per-sample Lipschitz
estimate.  The global L_log is estimated as the 95th/99th percentile of
these norms.

We report percentile statistics and the product L_log * D (diameter).

Usage:
    python scripts/04_estimate_lipschitz.py \\
        --model ultravox \\
        --representations data/representations/ultravox_librispeech.h5 \\
        --device cuda:0

    python scripts/04_estimate_lipschitz.py \\
        --model qwen2audio \\
        --representations data/representations/qwen2audio_cremad.h5 \\
        --n-samples 200 --device cuda:1
"""

from __future__ import annotations

import modality_collapse.utils.env  # noqa: F401  # set HF_HOME early

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from modality_collapse.extraction.storage import load_metadata, load_representations  # noqa: E402
from modality_collapse.models.registry import get_model_config  # noqa: E402
from modality_collapse.utils.config import load_config  # noqa: E402
from modality_collapse.utils.device import clear_gpu, get_device, log_vram  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading helpers (reuses project extractors where possible)
# ---------------------------------------------------------------------------

EXTRACTOR_MAP = {
    "ultravox": "UltravoxExtractor",
    "qwen2audio": "Qwen2AudioExtractor",
    "llava": "LlavaExtractor",
    "prismatic_dinov2": "PrismaticExtractor",
    "prismatic_siglip": "PrismaticExtractor",
}


def load_multimodal_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
):
    """Load a multimodal model using the project's extractor classes.

    Returns ``(model, extractor)`` — the extractor holds the processor
    and provides a ``preprocess()`` method that correctly builds the
    model-specific input dict (e.g. text prompt with ``<|audio|>``).
    """
    from modality_collapse.models import (
        LlavaExtractor,
        Qwen2AudioExtractor,
        UltravoxExtractor,
    )

    extractor_classes = {
        "ultravox": UltravoxExtractor,
        "qwen2audio": Qwen2AudioExtractor,
        "llava": LlavaExtractor,
    }

    # Handle Prismatic models (separate venv, lazy import).
    if model_name in ("prismatic_dinov2", "prismatic_siglip"):
        from modality_collapse.models.prismatic import PrismaticExtractor
        variant = "dinov2-224px+7b" if "dinov2" in model_name else "siglip-224px+7b"
        ext = PrismaticExtractor(device=device, dtype=dtype, variant=variant)
        model, _ = ext.load()
        return model, ext

    if model_name not in extractor_classes:
        raise ValueError(
            f"Unsupported model '{model_name}' for Lipschitz estimation. "
            f"Supported: {list(extractor_classes.keys())}"
        )

    ext = extractor_classes[model_name](device=device, dtype=dtype)
    model, processor = ext.load()
    return model, ext


def _get_llm_submodule(model, model_name: str):
    """Return the LLM (decoder) submodule of a multimodal model.

    Different architectures nest the LLM under different attribute names.
    """
    # Prismatic: llm_backbone.llm
    if hasattr(model, "llm_backbone") and hasattr(model.llm_backbone, "llm"):
        return model.llm_backbone.llm

    # Try common attribute paths.
    for attr in ("language_model", "model", "llm"):
        if hasattr(model, attr):
            return getattr(model, attr)

    raise AttributeError(
        f"Cannot find the LLM submodule for model '{model_name}'. "
        f"Checked attributes: llm_backbone.llm, language_model, model, llm."
    )


def _get_adapter_module_path(model_name: str) -> str:
    """Return the dot-separated module path to the adapter output layer."""
    config = get_model_config(model_name)
    path = config.hook_points.get("adapter_output")
    if path is None:
        raise ValueError(
            f"No adapter_output hook point defined for model '{model_name}'."
        )
    return path


def _resolve_module(model, path: str):
    """Navigate a dot-separated path to a submodule."""
    module = model
    for attr in path.split("."):
        module = getattr(module, attr)
    return module


# ---------------------------------------------------------------------------
# Core gradient-based Lipschitz estimation
# ---------------------------------------------------------------------------


def estimate_lipschitz_single(
    model,
    extractor,
    model_name: str,
    sample_data: dict,
    device: torch.device,
) -> float | None:
    """Estimate L_log for a single sample via gradient of log q(y|z,c).

    Args:
        model: The full multimodal model.
        extractor: The model's extractor (provides ``preprocess()``).
        model_name: Registry key (e.g. "ultravox").
        sample_data: Dict with keys needed to construct model inputs.
            For speech models: {"audio": np.ndarray, "sr": int, "transcript": str}
            For vision models: {"image": PIL.Image, "caption": str}
        device: Torch device.

    Returns:
        L_sample (float) or None on failure.
    """
    adapter_path = _get_adapter_module_path(model_name)
    config = get_model_config(model_name)

    # Storage for the adapter output captured by the hook.
    captured = {}

    def capture_hook(module, input, output):
        """Capture the adapter output tensor."""
        if isinstance(output, tuple):
            output = output[0]
        captured["z"] = output

    # Register a forward hook on the adapter.
    adapter_module = _resolve_module(model, adapter_path)
    handle = adapter_module.register_forward_hook(capture_hook)

    try:
        # --- Phase 1: Forward through encoder+adapter to capture z ----------
        with torch.no_grad():
            try:
                if config.modality == "speech":
                    inputs = extractor.preprocess(
                        sample_data["audio"],
                        sample_data.get("sr", 16000),
                    )
                else:
                    # Vision — extractor.preprocess() handles prompt + image
                    inputs = extractor.preprocess(sample_data["image"])
            except Exception as e:
                logger.warning("Failed to preprocess sample: %s", e)
                return None
            # Forward pass to populate the hook.
            _ = model(**inputs)

        if "z" not in captured:
            logger.warning("Hook did not fire for adapter_output. Skipping sample.")
            return None

        z_original = captured["z"].detach().clone()

    finally:
        handle.remove()
        captured.clear()

    # --- Phase 2: Forward z through the LLM with gradient tracking ----------
    # We need to replace the adapter output with our own tensor that has
    # requires_grad=True, then compute the log-likelihood and backprop.

    z = z_original.float()  # Use float32 for gradient stability.
    z.requires_grad_(True)

    # Replacement hook: intercept the adapter output and substitute z.
    def replace_hook(module, input, output):
        return z

    handle_replace = adapter_module.register_forward_hook(replace_hook)

    try:
        # Forward the full model again; the adapter output is replaced by z.
        # Use mixed precision for the forward pass.
        # Use the model's native dtype for the forward pass.
        autocast_dtype = torch.bfloat16 if z_original.dtype == torch.bfloat16 else torch.float16
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            outputs = model(**inputs)

        # Compute log probability of the actual tokens.
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        logits = logits.float()  # Compute loss in float32.

        # Shift for next-token prediction: logits[:, :-1], labels[:, 1:]
        if "labels" in inputs:
            labels = inputs["labels"]
        elif "input_ids" in inputs:
            labels = inputs["input_ids"]
        else:
            logger.warning("Cannot determine target tokens. Skipping sample.")
            return None

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Mask out padding tokens (label == -100 or pad_token_id).
        valid_mask = shift_labels >= 0
        if valid_mask.sum() == 0:
            logger.warning("No valid target tokens found. Skipping sample.")
            return None

        # Compute per-token log probabilities.
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        # Gather log probs of the target tokens.
        target_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.clamp(min=0).unsqueeze(-1),
        ).squeeze(-1)

        # Average log prob over valid tokens = log q(y|z,c).
        log_q = (target_log_probs * valid_mask.float()).sum() / valid_mask.sum()

        # --- Phase 3: Compute gradient of log_q w.r.t. z --------------------
        grad_z = torch.autograd.grad(
            log_q,
            z,
            create_graph=False,
            retain_graph=False,
        )[0]

        # Frobenius norm over all dimensions (seq x hidden).
        L_sample = grad_z.float().norm().item()
        return L_sample

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning("OOM during gradient computation. Clearing cache and skipping.")
            clear_gpu(device)
            return None
        raise

    finally:
        handle_replace.remove()
        # Explicitly clean up tensors.
        del z, z_original
        if "outputs" in dir():
            del outputs
        if "logits" in dir():
            del logits


def _prepare_inputs(
    processor,
    model_name: str,
    sample_data: dict,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    """Prepare model inputs from sample data using the processor.

    Returns None on failure.
    """
    config = get_model_config(model_name)

    try:
        if config.modality == "speech":
            audio = sample_data["audio"]
            sr = sample_data.get("sr", 16000)

            # Processor API varies by model.
            if model_name == "ultravox":
                inputs = processor(
                    audio=audio,
                    sampling_rate=sr,
                    return_tensors="pt",
                )
            elif model_name == "qwen2audio":
                inputs = processor(
                    audios=audio,
                    sampling_rate=sr,
                    return_tensors="pt",
                )
            else:
                inputs = processor(
                    audio=audio,
                    sampling_rate=sr,
                    return_tensors="pt",
                )

        elif config.modality == "vision":
            image = sample_data["image"]
            prompt = sample_data.get("prompt", "<image>\nDescribe this image.")
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            )
        else:
            logger.warning("Unsupported modality: %s", config.modality)
            return None

        return {k: v.to(device) for k, v in inputs.items()}

    except Exception as e:
        logger.warning("Failed to preprocess sample: %s", e)
        return None


# ---------------------------------------------------------------------------
# Sample loading from HDF5 + dataset
# ---------------------------------------------------------------------------


def load_samples_for_lipschitz(
    h5_path: str,
    model_name: str,
    n_samples: int,
) -> list[dict]:
    """Load sample data for Lipschitz estimation.

    We need raw inputs (audio waveforms / images) to feed through the
    full pipeline.  This function reads the sample_ids from the HDF5
    metadata and loads the corresponding raw data from the original datasets.

    If raw data loading is not feasible (datasets not available), we fall
    back to creating synthetic perturbations from stored representations.
    For now, we try to load from the dataset first.
    """
    metadata = load_metadata(h5_path)
    dataset_name = metadata.get("dataset_name", metadata.get("dataset", "unknown"))
    n_stored = int(metadata.get("n_samples", 0))

    # Load sample IDs if stored.
    sample_ids = None
    with h5py.File(h5_path, "r") as f:
        if "sample_ids" in f:
            raw = f["sample_ids"][:]
            sample_ids = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in raw[:n_stored]
            ]

    # Load audio/transcript data if stored in HDF5.
    samples = []
    with h5py.File(h5_path, "r") as f:
        has_audio = "audio_waveforms" in f
        has_transcripts = "transcript" in f or "transcripts" in f

        if has_audio:
            logger.info(
                "Loading raw audio waveforms from HDF5 for %d samples.",
                min(n_samples, n_stored),
            )
            for i in range(min(n_samples, n_stored)):
                audio = f["audio_waveforms"][i]
                # Audio may be stored as variable-length; trim trailing zeros.
                if isinstance(audio, np.ndarray):
                    audio = audio[audio != 0] if audio.ndim == 1 else audio
                    audio = audio.astype(np.float32)

                sample = {
                    "audio": audio,
                    "sr": 16000,
                    "index": i,
                }
                if has_transcripts:
                    tkey = "transcript" if "transcript" in f else "transcripts"
                    t = f[tkey][i]
                    if isinstance(t, bytes):
                        t = t.decode("utf-8")
                    sample["transcript"] = str(t)

                if sample_ids and i < len(sample_ids):
                    sample["sample_id"] = sample_ids[i]

                samples.append(sample)

            return samples

    # Fallback: try to re-load from the original dataset.
    logger.info(
        "Raw audio not in HDF5. Attempting to reload from dataset '%s'.",
        dataset_name,
    )
    samples = _reload_from_dataset(dataset_name, model_name, n_samples, metadata)
    return samples


def _reload_from_dataset(
    dataset_name: str,
    model_name: str,
    n_samples: int,
    metadata: dict,
) -> list[dict]:
    """Try to reload raw data from the original dataset loaders."""
    from modality_collapse.models.registry import get_model_config

    config = get_model_config(model_name)
    samples = []

    if config.modality == "speech":
        loader = _get_speech_loader(dataset_name, metadata, n_samples)
        if loader is None:
            return samples

        for i, sample in enumerate(loader):
            if i >= n_samples:
                break
            samples.append({
                "audio": sample.audio,
                "sr": sample.sample_rate,
                "transcript": sample.transcript or "",
                "sample_id": sample.sample_id,
                "index": i,
            })

    elif config.modality == "vision":
        loader = _get_vision_loader(dataset_name, metadata, n_samples)
        if loader is None:
            return samples

        for i, sample in enumerate(loader):
            if i >= n_samples:
                break
            samples.append({
                "image": sample.image,
                "caption": sample.caption or "",
                "prompt": "<image>\nDescribe this image.",
                "sample_id": sample.sample_id,
                "index": i,
            })

    return samples


def _get_speech_loader(
    dataset_name: str,
    metadata: dict,
    n_samples: int,
):
    """Create a speech dataset loader from metadata."""
    from modality_collapse.data.speech import (
        CREMADLoader,
        ESC50Loader,
        LibriSpeechLoader,
        VoxCeleb1Loader,
    )

    try:
        if dataset_name == "librispeech":
            return LibriSpeechLoader(
                split=metadata.get("split", "test"),
                max_samples=n_samples,
            )
        elif dataset_name in ("crema-d", "cremad"):
            data_dir = metadata.get("data_dir", "data/raw/CREMA-D")
            return CREMADLoader(data_dir=data_dir, max_samples=n_samples)
        elif dataset_name in ("esc-50", "esc50"):
            data_dir = metadata.get("data_dir", "data/raw/ESC-50")
            return ESC50Loader(data_dir=data_dir, max_samples=n_samples)
        elif dataset_name in ("voxceleb1", "voxceleb"):
            data_dir = metadata.get("data_dir", "data/raw/VoxCeleb1")
            return VoxCeleb1Loader(
                data_dir=data_dir,
                max_samples=n_samples,
            )
        else:
            logger.warning("Unknown speech dataset: %s", dataset_name)
            return None
    except FileNotFoundError as e:
        logger.warning("Dataset not available: %s", e)
        return None


def _get_vision_loader(
    dataset_name: str,
    metadata: dict,
    n_samples: int,
):
    """Create a vision dataset loader from metadata."""
    from modality_collapse.data.vision import COCOLoader, GQALoader

    try:
        if dataset_name == "coco":
            data_dir = metadata.get("data_dir", "data/raw/COCO")
            return COCOLoader(
                data_dir=data_dir,
                split=metadata.get("split", "val2017"),
                max_samples=n_samples,
            )
        elif dataset_name == "gqa":
            data_dir = metadata.get("data_dir", "data/raw/GQA")
            return GQALoader(
                data_dir=data_dir,
                split=metadata.get("split", "val"),
                max_samples=n_samples,
            )
        else:
            logger.warning("Unknown vision dataset: %s", dataset_name)
            return None
    except FileNotFoundError as e:
        logger.warning("Dataset not available: %s", e)
        return None


# ---------------------------------------------------------------------------
# Representation space diameter estimation
# ---------------------------------------------------------------------------


def estimate_diameter(
    h5_path: str,
    hook_name: str = "adapter_output",
    n_subsample: int = 500,
) -> float:
    """Estimate the diameter of the representation space.

    Uses std(X) * sqrt(dim) as a proxy for the diameter, plus
    a max-pairwise-distance estimate on a subsample.

    Returns the max of the two estimates.
    """
    X = load_representations(h5_path, hook_name)
    n, dim = X.shape

    # Sub-sample if needed.
    if n > n_subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=n_subsample, replace=False)
        X = X[idx]

    # Method 1: std * sqrt(dim).
    std_val = np.std(X).item()
    D_std = std_val * np.sqrt(dim)

    # Method 2: max pairwise distance on a small subsample.
    n_pairs = min(200, len(X))
    rng = np.random.default_rng(123)
    idx_a = rng.choice(len(X), size=n_pairs, replace=True)
    idx_b = rng.choice(len(X), size=n_pairs, replace=True)
    dists = np.linalg.norm(X[idx_a] - X[idx_b], axis=1)
    D_max = float(np.max(dists))

    D = float(max(D_std, D_max))
    logger.info(
        "Diameter estimates: std*sqrt(d)=%.4f, max_pairwise=%.4f => D=%.4f",
        D_std,
        D_max,
        D,
    )
    return D


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def run_sanity_checks(L_samples: np.ndarray, D: float) -> dict:
    """Run diagnostic checks on the Lipschitz estimates.

    Returns a dict of check results.
    """
    checks = {}

    # 1. L_log finite?
    n_finite = np.isfinite(L_samples).sum()
    checks["all_finite"] = bool(n_finite == len(L_samples))
    checks["n_finite"] = int(n_finite)
    checks["n_total"] = len(L_samples)

    # Filter to finite values for remaining checks.
    valid = L_samples[np.isfinite(L_samples)]
    if len(valid) < 5:
        checks["sufficient_samples"] = False
        checks["variation_ratio"] = float("inf")
        checks["L_D_product_95"] = float("inf")
        return checks

    checks["sufficient_samples"] = True

    # 2. Roughly uniform? Check ratio of 95th to 5th percentile.
    p5 = float(np.percentile(valid, 5))
    p95 = float(np.percentile(valid, 95))
    variation = p95 / max(p5, 1e-10)
    checks["p5"] = p5
    checks["p95"] = p95
    checks["variation_ratio"] = float(variation)
    checks["variation_ok"] = bool(variation < 10.0)  # <10x variation

    # 3. L_log * D in O(1-5)?
    L_95 = p95
    L_D = float(L_95 * D)
    checks["L_D_product_95"] = L_D
    checks["L_D_in_range"] = bool(0.01 <= L_D <= 50.0)  # Generous range

    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate L_log: the Lipschitz constant of the LLM's "
            "log-likelihood with respect to adapter representations."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["ultravox", "qwen2audio", "llava", "prismatic_dinov2", "prismatic_siglip"],
        help="Which multimodal model to use.",
    )
    parser.add_argument(
        "--representations",
        required=True,
        help="Path to the HDF5 file (for metadata, sample IDs, and diameter estimation).",
    )
    parser.add_argument(
        "--config",
        default="configs/experiment1.yaml",
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device (default: auto).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples for Lipschitz estimation (default: from config or 500).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write results (default: from config or results/exp1/lipschitz/).",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=50,
        help="Clear CUDA cache every N samples (default: 50).",
    )
    args = parser.parse_args()

    # ---- Logging ----------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- Load config -------------------------------------------------------
    cfg = load_config(args.config)
    lipschitz_cfg = cfg.get("lipschitz", {})

    n_samples = args.n_samples or int(lipschitz_cfg.get("n_samples", 500))
    output_dir = Path(
        args.output_dir or lipschitz_cfg.get("output_dir", "results/exp1/lipschitz/")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    flush_every = args.flush_every
    use_gradient_checkpointing = lipschitz_cfg.get("gradient_checkpointing", True)

    device = get_device(args.device)

    logger.info("=" * 80)
    logger.info("Lipschitz Estimation: L_log")
    logger.info("Model: %s", args.model)
    logger.info("HDF5:  %s", args.representations)
    logger.info("Device: %s", device)
    logger.info("N samples: %d", n_samples)
    logger.info("Output dir: %s", output_dir)
    logger.info("=" * 80)

    # ---- Load model -------------------------------------------------------
    logger.info("Loading model %s ...", args.model)
    model, extractor = load_multimodal_model(
        args.model, device, dtype=torch.float16,
    )
    log_vram(device, "after model load")

    # Enable gradient checkpointing on the LLM part for memory efficiency.
    if use_gradient_checkpointing:
        try:
            llm = _get_llm_submodule(model, args.model)
            if hasattr(llm, "gradient_checkpointing_enable"):
                llm.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing on LLM submodule.")
            else:
                logger.warning(
                    "LLM submodule does not support gradient_checkpointing_enable()."
                )
        except AttributeError as e:
            logger.warning("Could not enable gradient checkpointing: %s", e)

    # ---- Load samples -----------------------------------------------------
    logger.info("Loading samples for Lipschitz estimation ...")
    samples = load_samples_for_lipschitz(
        args.representations,
        args.model,
        n_samples,
    )

    if not samples:
        logger.error(
            "No samples available for Lipschitz estimation. "
            "Ensure the HDF5 file contains raw data or the dataset is accessible."
        )
        sys.exit(1)

    actual_n = min(n_samples, len(samples))
    logger.info("Loaded %d samples (requested %d).", len(samples), n_samples)

    # ---- Estimate diameter ------------------------------------------------
    logger.info("Estimating representation space diameter ...")
    try:
        D = estimate_diameter(args.representations, hook_name="adapter_output")
    except Exception as e:
        logger.warning("Failed to estimate diameter: %s. Using D=1.0.", e)
        D = 1.0

    # ---- Run Lipschitz estimation -----------------------------------------
    L_samples = []
    t_start = time.time()

    for i in range(actual_n):
        sample = samples[i]

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (actual_n - i - 1) / max(rate, 1e-6)
            logger.info(
                "[%d/%d] Processing sample (%.1f samples/min, ETA: %.0fs) ...",
                i + 1,
                actual_n,
                rate * 60,
                eta,
            )
            log_vram(device, f"sample {i+1}")

        L = estimate_lipschitz_single(
            model=model,
            extractor=extractor,
            model_name=args.model,
            sample_data=sample,
            device=device,
        )

        if L is not None:
            L_samples.append(L)
        else:
            logger.warning("  Sample %d returned None; skipping.", i)

        # Periodic cleanup.
        if (i + 1) % flush_every == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    elapsed_total = time.time() - t_start
    L_samples_arr = np.array(L_samples, dtype=np.float64)

    logger.info(
        "Completed %d/%d samples in %.1f seconds.",
        len(L_samples),
        actual_n,
        elapsed_total,
    )

    if len(L_samples_arr) == 0:
        logger.error("No valid Lipschitz samples collected. Exiting.")
        sys.exit(1)

    # ---- Compute statistics -----------------------------------------------
    valid = L_samples_arr[np.isfinite(L_samples_arr)]

    stats = {
        "model": args.model,
        "dataset": load_metadata(args.representations).get("dataset_name",
                    load_metadata(args.representations).get("dataset", "unknown")),
        "n_requested": n_samples,
        "n_computed": len(L_samples_arr),
        "n_valid": len(valid),
        "L_log_mean": float(np.mean(valid)),
        "L_log_std": float(np.std(valid)),
        "L_log_median": float(np.median(valid)),
        "L_log_p5": float(np.percentile(valid, 5)),
        "L_log_p25": float(np.percentile(valid, 25)),
        "L_log_p75": float(np.percentile(valid, 75)),
        "L_log_p95": float(np.percentile(valid, 95)),
        "L_log_p99": float(np.percentile(valid, 99)),
        "L_log_min": float(np.min(valid)),
        "L_log_max": float(np.max(valid)),
        "D_diameter": float(D),
        "L_D_product_95": float(np.percentile(valid, 95) * D),
        "L_D_product_99": float(np.percentile(valid, 99) * D),
        "elapsed_seconds": float(elapsed_total),
    }

    # ---- Sanity checks ----------------------------------------------------
    checks = run_sanity_checks(L_samples_arr, D)
    stats["checks"] = checks

    # ---- Save results -----------------------------------------------------
    dataset_tag = stats["dataset"]

    # Per-sample values (dataset-specific filename).
    per_sample_path = output_dir / f"lipschitz_samples_{args.model}_{dataset_tag}.npy"
    np.save(str(per_sample_path), L_samples_arr)
    logger.info("Saved per-sample L values to %s", per_sample_path)

    # Summary JSON (dataset-specific filename).
    json_path = output_dir / f"lipschitz_summary_{args.model}_{dataset_tag}.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved summary to %s", json_path)

    # ---- Print summary ----------------------------------------------------
    print("\n" + "=" * 80)
    print("LIPSCHITZ ESTIMATION SUMMARY")
    print(f"Model: {args.model}")
    print(f"Dataset: {stats['dataset']}")
    print("=" * 80)
    print(f"  Samples computed: {stats['n_computed']} / {stats['n_requested']}")
    print(f"  Valid (finite):   {stats['n_valid']}")
    print()
    print(f"  L_log (mean +/- std): {stats['L_log_mean']:.6f} +/- {stats['L_log_std']:.6f}")
    print(f"  L_log (median):       {stats['L_log_median']:.6f}")
    print(f"  L_log (5th pctl):     {stats['L_log_p5']:.6f}")
    print(f"  L_log (95th pctl):    {stats['L_log_p95']:.6f}")
    print(f"  L_log (99th pctl):    {stats['L_log_p99']:.6f}")
    print(f"  L_log (min):          {stats['L_log_min']:.6f}")
    print(f"  L_log (max):          {stats['L_log_max']:.6f}")
    print()
    print(f"  D (diameter):         {D:.4f}")
    print(f"  L_log_95 * D:         {stats['L_D_product_95']:.4f}")
    print(f"  L_log_99 * D:         {stats['L_D_product_99']:.4f}")

    # ---- Sanity check printout --------------------------------------------
    print("\n" + "-" * 80)
    print("SANITY CHECKS")
    print("-" * 80)

    def _status(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print(f"  All finite?            {_status(checks['all_finite'])}  "
          f"({checks['n_finite']}/{checks['n_total']})")

    if checks.get("sufficient_samples"):
        print(f"  Variation (p95/p5):    {_status(checks['variation_ok'])}  "
              f"(ratio = {checks['variation_ratio']:.2f}, threshold < 10)")
        print(f"  L_log*D in range?      {_status(checks['L_D_in_range'])}  "
              f"(L*D = {checks['L_D_product_95']:.4f})")
    else:
        print("  Insufficient valid samples for further checks.")

    print("=" * 80)

    # ---- Cleanup ----------------------------------------------------------
    del model, extractor
    gc.collect()
    clear_gpu(device)

    logger.info("Done.")


if __name__ == "__main__":
    main()

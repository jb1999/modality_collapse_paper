#!/usr/bin/env python3
"""Gradient projection onto MS vs TA eigenmodes.

Computes g_k = E[|<grad_z log q_psi(y|z,c), u_k>|] for each eigenmode u_k,
where u_k are the eigenvectors of the adapter output covariance Sigma_M.

This provides *causal* evidence for "the decoder doesn't use modality-specific
directions": if g_k ~ 0 for MS modes (low alpha_tilde), the decoder's scoring
function is flat along those directions by construction.

Usage:
    python scripts/11_gradient_projection.py \
        --model ultravox \
        --modal-file data/representations/ultravox_librispeech.h5 \
        --text-file data/representations/llama_from_ultravox_librispeech.h5 \
        --n-samples 200 --device cuda:0
"""

from __future__ import annotations

import modality_collapse.utils.env  # noqa: F401

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from modality_collapse.extraction.storage import load_metadata, load_representations
from modality_collapse.models.registry import get_model_config
from modality_collapse.utils.config import load_config
from modality_collapse.utils.device import clear_gpu, get_device, log_vram

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (adapted from 04_estimate_lipschitz to avoid numeric-name import)
# ---------------------------------------------------------------------------


def _get_adapter_module_path(model_name: str) -> str:
    config = get_model_config(model_name)
    path = config.hook_points.get("adapter_output")
    if path is None:
        raise ValueError(f"No adapter_output hook point for '{model_name}'.")
    return path


def _resolve_module(model, path: str):
    module = model
    for attr in path.split("."):
        module = getattr(module, attr)
    return module


def _get_llm_submodule(model, model_name: str):
    if hasattr(model, "llm_backbone") and hasattr(model.llm_backbone, "llm"):
        return model.llm_backbone.llm
    for attr in ("language_model", "model", "llm"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError(f"Cannot find LLM submodule for '{model_name}'.")


def load_multimodal_model(model_name: str, device, dtype=torch.float16):
    from modality_collapse.models import (
        LlavaExtractor, Qwen2AudioExtractor, UltravoxExtractor,
    )
    classes = {
        "ultravox": UltravoxExtractor,
        "qwen2audio": Qwen2AudioExtractor,
        "llava": LlavaExtractor,
    }
    if model_name in ("prismatic_dinov2", "prismatic_siglip"):
        from modality_collapse.models.prismatic import PrismaticExtractor
        variant = "dinov2-224px+7b" if "dinov2" in model_name else "siglip-224px+7b"
        ext = PrismaticExtractor(device=device, dtype=dtype, variant=variant)
        model, _ = ext.load()
        return model, ext
    ext = classes[model_name](device=device, dtype=dtype)
    model, _ = ext.load()
    return model, ext


def load_samples_for_lipschitz(h5_path, model_name, n_samples):
    """Delegate to the function in 04_estimate_lipschitz via importlib."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lipschitz_script",
        str(Path(__file__).resolve().parent / "04_estimate_lipschitz.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_samples_for_lipschitz(h5_path, model_name, n_samples)


def compute_eigenmodes(
    modal_file: str,
    text_file: str,
    hook_point: str = "adapter_output",
    n_modes: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute eigenmodes of adapter output and their text alignment scores.

    Returns:
        (eigenvectors, eigenvalues, alpha_tilde) where:
        - eigenvectors: (d, n_modes) column-major
        - eigenvalues: (n_modes,)
        - alpha_tilde: (n_modes,) alignment scores
    """
    # Load adapter output representations
    X_modal = load_representations(modal_file, hook_point).astype(np.float64)
    mask = np.any(X_modal != 0, axis=1)
    X_modal = X_modal[mask]

    # Compute modal covariance
    X_centered = X_modal - X_modal.mean(axis=0, keepdims=True)
    cov_M = (X_centered.T @ X_centered) / (len(X_centered) - 1)

    # Eigendecompose (sorted descending)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_M)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx][:n_modes]
    eigenvectors = eigenvectors[:, idx][:, :n_modes]

    # Compute text covariance for alignment scores
    # Use the same hook point from text file, or the closest available
    text_hooks = []
    with h5py.File(text_file, "r") as f:
        non_rep = {"sample_ids", "text", "transcript", "speaker_id",
                   "emotion", "sound_class", "caption", "question", "answer",
                   "object_category", "super_category", "object_count",
                   "avg_obj_size", "spatial_spread"}
        text_hooks = [k for k in f.keys() if k not in non_rep]

    # Use adapter_output from text if available, else llm_hidden_16
    text_hook = hook_point if hook_point in text_hooks else text_hooks[0]
    X_text = load_representations(text_file, text_hook).astype(np.float64)
    mask_t = np.any(X_text != 0, axis=1)
    X_text = X_text[mask_t]

    # If dimensions don't match (text might be different dim), project
    if X_text.shape[1] != X_modal.shape[1]:
        logger.warning(
            "Dim mismatch: modal=%d, text=%d. Using llm_hidden_16.",
            X_modal.shape[1], X_text.shape[1],
        )
        # Use llm_hidden_16 from both files
        X_modal_lh = load_representations(modal_file, "llm_hidden_16").astype(np.float64)
        X_text_lh = load_representations(text_file, "llm_hidden_16").astype(np.float64)
        X_modal_lh = X_modal_lh[np.any(X_modal_lh != 0, axis=1)]
        X_text_lh = X_text_lh[np.any(X_text_lh != 0, axis=1)]
        X_text = X_text_lh
        # Recompute with matched dimensions
        X_centered = X_modal_lh - X_modal_lh.mean(axis=0, keepdims=True)
        cov_M = (X_centered.T @ X_centered) / (len(X_centered) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_M)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx][:n_modes]
        eigenvectors = eigenvectors[:, idx][:, :n_modes]

    X_text_c = X_text - X_text.mean(axis=0, keepdims=True)
    cov_T = (X_text_c.T @ X_text_c) / (len(X_text_c) - 1)

    # Compute alpha_tilde(u_k) = u_k^T Sigma_T u_k / lambda_k
    alpha = np.zeros(n_modes)
    for k in range(n_modes):
        u_k = eigenvectors[:, k]
        alpha[k] = (u_k @ cov_T @ u_k) / max(eigenvalues[k], 1e-10)

    logger.info(
        "Computed %d eigenmodes. Top-5 alpha: %s",
        n_modes,
        [f"{a:.4f}" for a in alpha[:5]],
    )
    return eigenvectors, eigenvalues, alpha


def compute_gradient_projections(
    model,
    extractor,
    model_name: str,
    samples: list[dict],
    eigenvectors: np.ndarray,
    device: torch.device,
    n_samples: int = 200,
) -> np.ndarray:
    """Compute g_k = |<grad_z log q_psi, u_k>| for each sample and eigenmode.

    Returns:
        (n_samples, n_modes) array of absolute inner products.
    """
    adapter_path = _get_adapter_module_path(model_name)
    adapter_module = _resolve_module(model, adapter_path)
    config = get_model_config(model_name)

    n_modes = eigenvectors.shape[1]
    U = torch.tensor(eigenvectors, dtype=torch.float32, device=device)  # (d, n_modes)

    all_projections = []
    actual_n = min(n_samples, len(samples))

    t_start = time.time()
    for i in range(actual_n):
        sample = samples[i]

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (actual_n - i - 1) / max(rate, 1e-6)
            logger.info(
                "[%d/%d] Computing gradient projection (%.1f/min, ETA: %.0fs)",
                i + 1, actual_n, rate * 60, eta,
            )

        captured = {}

        def capture_hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            captured["z"] = output

        handle = adapter_module.register_forward_hook(capture_hook)
        try:
            with torch.no_grad():
                if config.modality == "speech":
                    inputs = extractor.preprocess(
                        sample["audio"], sample.get("sr", 16000),
                    )
                else:
                    inputs = extractor.preprocess(sample["image"])
                _ = model(**inputs)

            if "z" not in captured:
                logger.warning("Hook didn't fire for sample %d", i)
                continue
            z_original = captured["z"].detach().clone()
        finally:
            handle.remove()
            captured.clear()

        # Replace adapter output with grad-enabled tensor
        z = z_original.float()
        z.requires_grad_(True)

        def replace_hook(module, input, output):
            return z

        handle_r = adapter_module.register_forward_hook(replace_hook)
        try:
            autocast_dtype = torch.bfloat16 if z_original.dtype == torch.bfloat16 else torch.float16
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                outputs = model(**inputs)

            logits = outputs.logits.float() if hasattr(outputs, "logits") else outputs[0].float()
            labels = inputs.get("labels", inputs.get("input_ids"))
            if labels is None:
                continue

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            valid_mask = shift_labels >= 0
            if valid_mask.sum() == 0:
                continue

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            target_log_probs = log_probs.gather(
                dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1),
            ).squeeze(-1)
            log_q = (target_log_probs * valid_mask.float()).sum() / valid_mask.sum()

            grad_z = torch.autograd.grad(log_q, z, create_graph=False)[0]

            # Mean-pool gradient across sequence length to get (d,) vector
            grad_pooled = grad_z.float().squeeze(0).mean(dim=0)  # (d,)

            # Project onto eigenmodes: |<grad, u_k>| for each k
            projections = torch.abs(grad_pooled @ U)  # (n_modes,)
            all_projections.append(projections.detach().cpu().numpy())

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("OOM at sample %d, skipping", i)
                clear_gpu(device)
                continue
            raise
        finally:
            handle_r.remove()
            del z, z_original
            if (i + 1) % 30 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    if not all_projections:
        logger.error("No gradient projections computed!")
        return np.array([])

    return np.stack(all_projections, axis=0)  # (n_valid, n_modes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute gradient projections onto MS/TA eigenmodes."
    )
    parser.add_argument("--model", required=True,
                        choices=["ultravox", "qwen2audio", "llava",
                                 "prismatic_dinov2", "prismatic_siglip"])
    parser.add_argument("--modal-file", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-modes", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="results/exp1/gradient_projection/")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Compute eigenmodes (CPU)
    logger.info("Computing eigenmodes from adapter_output representations...")
    eigenvectors, eigenvalues, alpha = compute_eigenmodes(
        args.modal_file, args.text_file, n_modes=args.n_modes,
    )

    # Step 2: Load model (GPU)
    logger.info("Loading model %s ...", args.model)
    model, extractor = load_multimodal_model(args.model, device, torch.float16)

    # Enable gradient checkpointing
    try:
        llm = _get_llm_submodule(model, args.model)
        if hasattr(llm, "gradient_checkpointing_enable"):
            llm.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing.")
    except Exception as e:
        logger.warning("Could not enable gradient checkpointing: %s", e)

    # Step 3: Load samples
    logger.info("Loading samples...")
    samples = load_samples_for_lipschitz(args.modal_file, args.model, args.n_samples)
    logger.info("Loaded %d samples.", len(samples))

    # Step 4: Compute gradient projections
    logger.info("Computing gradient projections...")
    projections = compute_gradient_projections(
        model, extractor, args.model, samples,
        eigenvectors, device, n_samples=args.n_samples,
    )

    if projections.size == 0:
        logger.error("No projections computed. Exiting.")
        sys.exit(1)

    # Step 5: Analyze results
    g_k = np.mean(projections, axis=0)  # E[|<grad, u_k>|]
    g_k_std = np.std(projections, axis=0)

    # Classify modes
    ms_mask = alpha < 0.5
    ta_mask = alpha >= 0.5

    g_ms = g_k[ms_mask].mean() if ms_mask.any() else 0.0
    g_ta = g_k[ta_mask].mean() if ta_mask.any() else 0.0
    ratio = g_ta / max(g_ms, 1e-10)

    # Correlation between g_k and alpha
    corr = np.corrcoef(g_k, alpha)[0, 1] if len(g_k) > 2 else 0.0

    # Save results
    metadata = load_metadata(args.modal_file)
    dataset = metadata.get("dataset_name", metadata.get("dataset", "unknown"))
    tag = f"{args.model}_{dataset}"

    results = {
        "model": args.model,
        "dataset": dataset,
        "n_samples": int(projections.shape[0]),
        "n_modes": int(args.n_modes),
        "g_ms_mean": float(g_ms),
        "g_ta_mean": float(g_ta),
        "g_ta_over_g_ms": float(ratio),
        "correlation_g_alpha": float(corr),
        "n_ms_modes": int(ms_mask.sum()),
        "n_ta_modes": int(ta_mask.sum()),
    }

    json_path = output_dir / f"gradient_projection_{tag}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    np.savez(
        output_dir / f"gradient_projection_{tag}_arrays.npz",
        g_k=g_k,
        g_k_std=g_k_std,
        alpha=alpha,
        eigenvalues=eigenvalues,
        projections=projections,
    )

    # Print summary
    print("\n" + "=" * 80)
    print(f"GRADIENT PROJECTION: {args.model} / {dataset}")
    print("=" * 80)
    print(f"  Samples:   {projections.shape[0]}")
    print(f"  Modes:     {args.n_modes} ({ms_mask.sum()} MS, {ta_mask.sum()} TA)")
    print(f"  g_MS mean: {g_ms:.6f}")
    print(f"  g_TA mean: {g_ta:.6f}")
    print(f"  g_TA/g_MS: {ratio:.1f}x")
    print(f"  Corr(g_k, alpha): {corr:.4f}")
    print()
    print("  Top-10 modes:")
    print(f"  {'Mode':>5} {'alpha':>8} {'g_k':>10} {'Type':>5}")
    print("  " + "-" * 35)
    for k in range(min(10, len(g_k))):
        mtype = "MS" if alpha[k] < 0.5 else "TA"
        print(f"  {k:>5} {alpha[k]:>8.4f} {g_k[k]:>10.6f} {mtype:>5}")
    print("=" * 80)

    # Cleanup
    del model, extractor
    gc.collect()
    clear_gpu(device)


if __name__ == "__main__":
    main()

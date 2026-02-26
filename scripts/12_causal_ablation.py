#!/usr/bin/env python3
"""
12_causal_ablation.py — Causal ablation of eigenmodes at the adapter output.

For each sample, run 4 forward passes:
  1. Baseline (no ablation)
  2. MS-ablated: project out ALL modality-specific modes (α̃ < 0.5)
  3. TA-matched: project out top-K text-aligned modes by eigenvalue (K = |MS modes|)
  4. Random-matched: project out K random eigenmodes (fixed seed)

Compare cross-entropy across conditions.
Prediction from the mismatched-decoder theory:
  - MS ablation removes most variance but causes minimal loss change
  - TA ablation removes less variance but causes substantial loss change
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import modality_collapse.utils.env  # noqa: F401,E402  # set HF_HOME early
from modality_collapse.models.registry import get_model_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Model / dataset configs
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "ultravox": {
        "modal_h5": "data/representations/ultravox_librispeech.h5",
        "text_h5": "data/representations/llama_from_ultravox_librispeech.h5",
        "tag": "ultravox_librispeech",
    },
    "prismatic_dinov2": {
        "modal_h5": "data/representations/prismatic_dinov2_coco.h5",
        "text_h5": "data/representations/vicuna_from_llava_coco.h5",
        "tag": "prismatic_dinov2_coco",
    },
    "prismatic_siglip": {
        "modal_h5": "data/representations/prismatic_siglip_coco.h5",
        "text_h5": "data/representations/vicuna_from_llava_coco.h5",
        "tag": "prismatic_siglip_coco",
    },
}

# ---------------------------------------------------------------------------
# Eigenmodes (recomputed from HDF5; eigenvectors not saved by script 07)
# ---------------------------------------------------------------------------


def compute_eigenmodes(
    modal_h5: str, text_h5: str, n_modes: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (eigenvectors, eigenvalues, alpha_tilde) from adapter_output."""
    with h5py.File(modal_h5, "r") as f:
        X_m = f["adapter_output"][:]
    with h5py.File(text_h5, "r") as f:
        key = "adapter_output" if "adapter_output" in f else "llm_hidden_16"
        X_t = f[key][:]

    # dimension mismatch → fall back to llm_hidden_16 for both
    if X_m.shape[1] != X_t.shape[1]:
        with h5py.File(modal_h5, "r") as f:
            X_m = f["llm_hidden_16"][:]
        with h5py.File(text_h5, "r") as f:
            X_t = f["llm_hidden_16"][:]

    # drop zero rows
    mask = np.any(X_m != 0, axis=1)
    X_m = X_m[mask]

    # modal covariance + eigendecomposition
    X_mc = X_m - X_m.mean(0)
    cov_m = X_mc.T @ X_mc / (len(X_m) - 1)
    evals, evecs = np.linalg.eigh(cov_m)
    idx = np.argsort(evals)[::-1][:n_modes]
    evals = evals[idx]
    evecs = evecs[:, idx]  # (d, n_modes)

    # text covariance → alignment scores
    X_tc = X_t - X_t.mean(0)
    cov_t = X_tc.T @ X_tc / (len(X_t) - 1)
    alpha = np.array(
        [evecs[:, k] @ cov_t @ evecs[:, k] / max(evals[k], 1e-10) for k in range(n_modes)]
    )
    return evecs, evals, alpha


# ---------------------------------------------------------------------------
# Model helpers (inlined; can't import from 04_ due to numeric prefix)
# ---------------------------------------------------------------------------


def _get_adapter_module_path(model_name: str) -> str:
    return get_model_config(model_name).hook_points["adapter_output"]


def _resolve_module(model: torch.nn.Module, path: str) -> torch.nn.Module:
    mod = model
    for p in path.split("."):
        mod = getattr(mod, p)
    return mod


def load_multimodal_model(
    model_name: str, device: torch.device, dtype: torch.dtype = torch.float16
):
    from modality_collapse.models import (
        LlavaExtractor,
        Qwen2AudioExtractor,
        UltravoxExtractor,
    )

    if model_name == "ultravox":
        ext = UltravoxExtractor(device=device, dtype=dtype)
    elif model_name == "qwen2audio":
        ext = Qwen2AudioExtractor(device=device, dtype=dtype)
    elif model_name == "llava":
        ext = LlavaExtractor(device=device, dtype=dtype)
    elif model_name.startswith("prismatic"):
        from modality_collapse.models.prismatic import PrismaticExtractor

        variant = {
            "prismatic_dinov2": "dinov2-224px+7b",
            "prismatic_siglip": "siglip-224px+7b",
        }[model_name]
        ext = PrismaticExtractor(device=device, dtype=torch.bfloat16, variant=variant)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model, _ = ext.load()
    return model, ext


def _load_samples(h5_path: str, model_name: str, n_samples: int):
    spec = importlib.util.spec_from_file_location(
        "estimate_lipschitz",
        Path(__file__).parent / "04_estimate_lipschitz.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_samples_for_lipschitz(h5_path, model_name, n_samples)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


def _compute_loss(model, inputs: dict, device: torch.device) -> float | None:
    """Next-token cross-entropy (teacher-forcing)."""
    with torch.no_grad():
        outputs = model(
            **{k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        )
    logits = outputs.logits  # (1, seq, vocab)
    labels = inputs.get("labels", inputs.get("input_ids"))
    if hasattr(labels, "to"):
        labels = labels.to(device)
    s_logits = logits[:, :-1, :]
    s_labels = labels[:, 1:]
    valid = s_labels >= 0
    if valid.sum() == 0:
        return None
    lp = F.log_softmax(s_logits.float(), dim=-1)
    tgt = lp.gather(-1, s_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return -(tgt * valid.float()).sum().item() / valid.sum().item()


# ---------------------------------------------------------------------------
# Ablation hook factory
# ---------------------------------------------------------------------------


def _make_ablation_hook(U_ablate: torch.Tensor):
    """Return a forward hook that projects out columns of U_ablate from the output.

    U_ablate: (d, K) tensor on the model's device.
    """

    def hook(module, input, output):
        z = output[0] if isinstance(output, tuple) else output
        z_f = z.float()
        proj = z_f @ U_ablate  # (b, seq, K)
        z_abl = (z_f - proj @ U_ablate.T).to(z.dtype)
        if isinstance(output, tuple):
            return (z_abl,) + output[1:]
        return z_abl

    return hook


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_causal_ablation(
    model_name: str,
    n_samples: int = 200,
    n_modes: int = 100,
    ms_threshold: float = 0.5,
    device: torch.device | None = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = MODEL_CONFIGS[model_name]
    modal_h5 = str(ROOT / cfg["modal_h5"])
    text_h5 = str(ROOT / cfg["text_h5"])
    tag = cfg["tag"]

    # --- eigenmodes ---
    print(f"Computing eigenmodes ({n_modes} modes) …")
    evecs, evals, alpha = compute_eigenmodes(modal_h5, text_h5, n_modes)

    ms_mask = alpha < ms_threshold
    ta_mask = ~ms_mask
    n_ms = int(ms_mask.sum())
    n_ta = int(ta_mask.sum())
    ms_var_frac = evals[ms_mask].sum() / evals.sum() * 100
    ta_var_frac = evals[ta_mask].sum() / evals.sum() * 100

    print(f"  MS modes: {n_ms}/{n_modes}  ({ms_var_frac:.1f}% variance)")
    print(f"  TA modes: {n_ta}/{n_modes}  ({ta_var_frac:.1f}% variance)")

    # build ablation subspaces on GPU
    U = torch.tensor(evecs, dtype=torch.float32, device=device)
    U_ms = U[:, ms_mask]  # (d, n_ms)

    # TA-matched: top-K TA modes by eigenvalue, K = n_ms
    ta_indices = np.where(ta_mask)[0]
    ta_by_eval = ta_indices[np.argsort(evals[ta_indices])[::-1]]
    k_match = min(n_ms, len(ta_by_eval))
    U_ta_matched = U[:, ta_by_eval[:k_match]]  # (d, k_match)
    ta_matched_var = evals[ta_by_eval[:k_match]].sum() / evals.sum() * 100

    # random-matched: K random eigenmodes (fixed seed)
    rng = np.random.default_rng(42)
    rand_indices = rng.choice(n_modes, size=k_match, replace=False)
    U_rand = U[:, rand_indices]
    rand_var = evals[rand_indices].sum() / evals.sum() * 100

    print(f"  TA-matched: top-{k_match} TA modes ({ta_matched_var:.1f}% variance)")
    print(f"  Random:     {k_match} random modes ({rand_var:.1f}% variance)")

    # --- model ---
    print(f"\nLoading {model_name} …")
    model, extractor = load_multimodal_model(model_name, device)
    model.eval()
    adapter_path = _get_adapter_module_path(model_name)
    adapter_module = _resolve_module(model, adapter_path)

    # --- samples ---
    print(f"Loading samples (n={n_samples}) …")
    samples = _load_samples(modal_h5, model_name, n_samples)
    print(f"  Loaded {len(samples)} samples")

    # --- forward passes ---
    conditions = [
        ("ms_all", U_ms),
        ("ta_matched", U_ta_matched),
        ("random_matched", U_rand),
    ]

    rows = []
    for i, sample in enumerate(tqdm(samples[:n_samples], desc="Causal ablation")):
        try:
            # preprocess
            if model_name in ("ultravox", "qwen2audio"):
                inputs = extractor.preprocess(sample["audio"], sample["sr"])
            else:
                inputs = extractor.preprocess(sample["image"], sample.get("prompt", ""))

            # baseline
            loss_base = _compute_loss(model, inputs, device)
            if loss_base is None:
                continue

            row = {
                "sample_id": sample.get("sample_id", str(i)),
                "loss_baseline": loss_base,
            }

            # ablation conditions
            for name, U_abl in conditions:
                hook = _make_ablation_hook(U_abl)
                handle = adapter_module.register_forward_hook(hook)
                loss = _compute_loss(model, inputs, device)
                handle.remove()
                row[f"loss_{name}"] = loss if loss is not None else float("nan")
                row[f"delta_{name}"] = (
                    (loss - loss_base) if loss is not None else float("nan")
                )

            rows.append(row)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n  OOM on sample {i}, skipping")
                gc.collect()
                torch.cuda.empty_cache()
                continue
            raise

        if (i + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # --- save ---
    out_dir = ROOT / "results" / "exp1" / "causal_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"causal_ablation_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-sample results → {csv_path}")

    # summary
    deltas = {
        name: [r[f"delta_{name}"] for r in rows if not np.isnan(r[f"delta_{name}"])]
        for name, _ in conditions
    }
    baseline_mean = float(np.mean([r["loss_baseline"] for r in rows]))
    summary = {
        "model": model_name,
        "tag": tag,
        "n_samples": len(rows),
        "n_modes": n_modes,
        "ms_threshold": ms_threshold,
        "n_ms_modes": n_ms,
        "n_ta_modes": n_ta,
        "ms_var_fraction": round(float(ms_var_frac), 1),
        "ta_matched_var_fraction": round(float(ta_matched_var), 1),
        "random_var_fraction": round(float(rand_var), 1),
        "mean_loss_baseline": round(baseline_mean, 4),
    }
    for name, _ in conditions:
        d = deltas[name]
        m = float(np.mean(d))
        summary[f"mean_delta_{name}"] = round(m, 4)
        summary[f"median_delta_{name}"] = round(float(np.median(d)), 4)
        summary[f"std_delta_{name}"] = round(float(np.std(d)), 4)
        summary[f"pct_change_{name}"] = round(m / baseline_mean * 100, 2)

    json_path = out_dir / f"causal_ablation_{tag}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {json_path}")

    # print summary table
    print(f"\n{'='*60}")
    print(f"CAUSAL ABLATION — {model_name}")
    print(f"{'='*60}")
    print(f"Baseline loss: {summary['mean_loss_baseline']:.4f}")
    print(f"{'Condition':<20} {'Modes':>6} {'Var%':>6} {'ΔLoss':>10} {'%Change':>8}")
    print(f"{'-'*60}")
    labels = [
        ("ms_all", f"{n_ms}", f"{ms_var_frac:.1f}"),
        ("ta_matched", f"{k_match}", f"{ta_matched_var:.1f}"),
        ("random_matched", f"{k_match}", f"{rand_var:.1f}"),
    ]
    for name, nm, vf in labels:
        dl = summary[f"mean_delta_{name}"]
        pc = summary[f"pct_change_{name}"]
        print(f"{name:<20} {nm:>6} {vf:>6} {dl:>+10.4f} {pc:>+7.2f}%")
    print(f"{'='*60}")

    return summary


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal ablation of eigenmodes")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to ablate",
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-modes", type=int, default=100)
    parser.add_argument("--ms-threshold", type=float, default=0.5)
    args = parser.parse_args()

    run_causal_ablation(
        model_name=args.model,
        n_samples=args.n_samples,
        n_modes=args.n_modes,
        ms_threshold=args.ms_threshold,
    )

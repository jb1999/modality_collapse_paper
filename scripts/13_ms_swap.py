#!/usr/bin/env python3
"""
13_ms_swap.py — MS-subspace swap experiment.

For each sample from speaker A, replace the modality-specific (MS) component
at the adapter output with speaker B's mean MS signature, then measure:
  - KL divergence between original and swapped logit distributions
  - Cross-entropy change

Control: do the same with the text-aligned (TA) component.

Expected:
  - MS-swap → near-zero KL  (decoder ignores MS directions)
  - TA-swap → large KL       (decoder relies on TA directions)
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import modality_collapse.utils.env  # noqa: F401,E402
from modality_collapse.models.registry import get_model_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Eigenmodes
# ---------------------------------------------------------------------------


def compute_eigenmodes(
    modal_h5: str, text_h5: str, n_modes: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (eigenvectors, eigenvalues, alpha_tilde)."""
    with h5py.File(modal_h5, "r") as f:
        X_m = f["adapter_output"][:]
    with h5py.File(text_h5, "r") as f:
        key = "adapter_output" if "adapter_output" in f else "llm_hidden_16"
        X_t = f[key][:]
    if X_m.shape[1] != X_t.shape[1]:
        with h5py.File(modal_h5, "r") as f:
            X_m = f["llm_hidden_16"][:]
        with h5py.File(text_h5, "r") as f:
            X_t = f["llm_hidden_16"][:]
    mask = np.any(X_m != 0, axis=1)
    X_m = X_m[mask]
    X_mc = X_m - X_m.mean(0)
    cov_m = X_mc.T @ X_mc / (len(X_m) - 1)
    evals, evecs = np.linalg.eigh(cov_m)
    idx = np.argsort(evals)[::-1][:n_modes]
    evals = evals[idx]
    evecs = evecs[:, idx]
    X_tc = X_t - X_t.mean(0)
    cov_t = X_tc.T @ X_tc / (len(X_t) - 1)
    alpha = np.array(
        [evecs[:, k] @ cov_t @ evecs[:, k] / max(evals[k], 1e-10) for k in range(n_modes)]
    )
    return evecs, evals, alpha


# ---------------------------------------------------------------------------
# Per-speaker MS/TA signatures from stored representations
# ---------------------------------------------------------------------------


def compute_speaker_signatures(
    modal_h5: str,
    evecs: np.ndarray,
    alpha: np.ndarray,
    ms_threshold: float = 0.5,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute per-speaker mean MS and TA vectors from HDF5."""
    ms_mask = alpha < ms_threshold
    ta_mask = ~ms_mask
    U_ms = evecs[:, ms_mask]  # (d, n_ms)
    U_ta = evecs[:, ta_mask]  # (d, n_ta)

    with h5py.File(modal_h5, "r") as f:
        X = f["adapter_output"][:]
        speakers = [
            s.decode() if isinstance(s, bytes) else s for s in f["speaker_id"][:]
        ]

    # group by speaker
    spk_to_idx = defaultdict(list)
    for i, s in enumerate(speakers):
        spk_to_idx[s].append(i)

    signatures = {}
    for spk, idxs in spk_to_idx.items():
        mu = X[idxs].mean(axis=0)  # (d,)
        mu_ms = U_ms @ (U_ms.T @ mu)  # projection onto MS subspace
        mu_ta = U_ta @ (U_ta.T @ mu)  # projection onto TA subspace
        signatures[spk] = {
            "ms": mu_ms.astype(np.float32),
            "ta": mu_ta.astype(np.float32),
            "n_samples": len(idxs),
        }
    return signatures


# ---------------------------------------------------------------------------
# Model helpers (inlined from script 12)
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
        variant = {"prismatic_dinov2": "dinov2-224px+7b", "prismatic_siglip": "siglip-224px+7b"}[model_name]
        ext = PrismaticExtractor(device=device, dtype=torch.bfloat16, variant=variant)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model, _ = ext.load()
    return model, ext


def _load_samples(h5_path: str, model_name: str, n_samples: int):
    spec = importlib.util.spec_from_file_location(
        "estimate_lipschitz", Path(__file__).parent / "04_estimate_lipschitz.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_samples_for_lipschitz(h5_path, model_name, n_samples)


# ---------------------------------------------------------------------------
# KL divergence between logit distributions
# ---------------------------------------------------------------------------


def _kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean per-token KL(p || q) over valid positions."""
    # Shift for next-token prediction
    lp = logits_p[:, :-1, :]
    lq = logits_q[:, :-1, :]
    sl = labels[:, 1:]
    valid = sl >= 0
    if valid.sum() == 0:
        return float("nan")
    p = F.softmax(lp.float(), dim=-1)
    log_p = F.log_softmax(lp.float(), dim=-1)
    log_q = F.log_softmax(lq.float(), dim=-1)
    kl = (p * (log_p - log_q)).sum(dim=-1)  # (batch, seq)
    return (kl * valid.float()).sum().item() / valid.sum().item()


def _cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> float | None:
    """Mean next-token cross-entropy over valid positions."""
    sl = logits[:, :-1, :]
    tl = labels[:, 1:]
    valid = tl >= 0
    if valid.sum() == 0:
        return None
    lp = F.log_softmax(sl.float(), dim=-1)
    tgt = lp.gather(-1, tl.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return -(tgt * valid.float()).sum().item() / valid.sum().item()


# ---------------------------------------------------------------------------
# Swap hooks
# ---------------------------------------------------------------------------


def _make_mean_swap_hook(
    U_subspace: torch.Tensor,
    donor_mean_vec: torch.Tensor,
):
    """Swap the MEAN subspace component while preserving per-token residuals.

    For each token t:
      z_sub[t] = projection of z[t] onto U_subspace
      mean_sub = mean_t(z_sub[t])       # sample's own mean subspace component
      residual[t] = z_sub[t] - mean_sub  # per-token variation (preserved)
      z'[t] = (z[t] - z_sub[t]) + donor_mean_vec + residual[t]
            = z[t] - mean_sub + donor_mean_vec

    This only changes the "DC offset" in the subspace: the speaker identity
    shifts from A to B, but per-token structure is preserved.

    U_subspace:     (d, K) eigenvectors
    donor_mean_vec: (d,)   donor's mean subspace component (pre-projected)
    """

    def hook(module, input, output):
        z = output[0] if isinstance(output, tuple) else output
        z_f = z.float()
        # Per-token subspace projection
        coeffs = z_f @ U_subspace  # (1, seq, K)
        z_sub = coeffs @ U_subspace.T  # (1, seq, d)
        # Mean of this sample's subspace component across positions
        mean_sub = z_sub.mean(dim=1, keepdim=True)  # (1, 1, d)
        # Swap: replace sample's mean with donor's mean, keep residuals
        z_swapped = z_f - mean_sub + donor_mean_vec.unsqueeze(0).unsqueeze(0)
        z_swapped = z_swapped.to(z.dtype)
        if isinstance(output, tuple):
            return (z_swapped,) + output[1:]
        return z_swapped

    return hook


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_ms_swap(
    model_name: str = "ultravox",
    n_samples: int = 200,
    n_modes: int = 100,
    ms_threshold: float = 0.5,
    device: torch.device | None = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modal_h5 = str(ROOT / "data/representations/ultravox_librispeech.h5")
    text_h5 = str(ROOT / "data/representations/llama_from_ultravox_librispeech.h5")
    tag = "ultravox_librispeech"

    # --- eigenmodes ---
    print("Computing eigenmodes …")
    evecs, evals, alpha = compute_eigenmodes(modal_h5, text_h5, n_modes)
    ms_mask = alpha < ms_threshold
    ta_mask = ~ms_mask
    n_ms = int(ms_mask.sum())
    n_ta = int(ta_mask.sum())
    print(f"  MS modes: {n_ms}, TA modes: {n_ta}")

    U_ms = torch.tensor(evecs[:, ms_mask], dtype=torch.float32, device=device)
    U_ta = torch.tensor(evecs[:, ta_mask], dtype=torch.float32, device=device)

    # --- per-speaker signatures ---
    print("Computing per-speaker MS/TA signatures …")
    sigs = compute_speaker_signatures(modal_h5, evecs, alpha, ms_threshold)
    speakers = sorted(sigs.keys())
    print(f"  {len(speakers)} speakers")

    # Pre-convert to GPU tensors
    sig_tensors = {
        spk: {
            "ms": torch.tensor(s["ms"], dtype=torch.float32, device=device),
            "ta": torch.tensor(s["ta"], dtype=torch.float32, device=device),
        }
        for spk, s in sigs.items()
    }

    # --- model ---
    print(f"\nLoading {model_name} …")
    model, extractor = load_multimodal_model(model_name, device)
    model.eval()
    adapter_path = _get_adapter_module_path(model_name)
    adapter_module = _resolve_module(model, adapter_path)

    # --- samples ---
    print(f"Loading samples (n={n_samples}) …")
    samples = _load_samples(modal_h5, model_name, n_samples)

    # Map sample_id to speaker
    with h5py.File(modal_h5, "r") as f:
        all_ids = [s.decode() if isinstance(s, bytes) else s for s in f["sample_ids"][:]]
        all_spk = [s.decode() if isinstance(s, bytes) else s for s in f["speaker_id"][:]]
    id_to_spk = dict(zip(all_ids, all_spk))

    rng = np.random.default_rng(42)

    # --- swap experiment ---
    rows = []
    for i, sample in enumerate(tqdm(samples[:n_samples], desc="MS swap")):
        try:
            sid = sample.get("sample_id", str(i))
            spk_a = id_to_spk.get(sid)
            if spk_a is None:
                continue

            # Pick random different speaker
            other_speakers = [s for s in speakers if s != spk_a]
            spk_b = rng.choice(other_speakers)

            # Preprocess
            inputs = extractor.preprocess(sample["audio"], sample["sr"])
            labels = inputs.get("labels", inputs.get("input_ids"))
            if hasattr(labels, "to"):
                labels = labels.to(device)

            inputs_gpu = {
                k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()
            }

            # --- Baseline ---
            with torch.no_grad():
                out_base = model(**inputs_gpu)
            logits_base = out_base.logits
            loss_base = _cross_entropy(logits_base, labels)
            if loss_base is None:
                continue

            # Compute perturbation norms (‖donor_mean - sample_mean‖ in each subspace)
            # We need the sample's own mean subspace components
            # These are computed inside the hook, but we need them for normalization
            # Approximate: use stored representations to get sample's mean adapter output
            # For now, use the pre-computed speaker signatures as proxy
            delta_ms_norm = float(
                (sig_tensors[spk_b]["ms"] - sig_tensors[spk_a]["ms"]).norm().item()
            )
            delta_ta_norm = float(
                (sig_tensors[spk_b]["ta"] - sig_tensors[spk_a]["ta"]).norm().item()
            )

            # --- MS-swapped (replace MS with speaker B's mean MS) ---
            hook = _make_mean_swap_hook(U_ms, sig_tensors[spk_b]["ms"])
            handle = adapter_module.register_forward_hook(hook)
            with torch.no_grad():
                out_ms = model(**inputs_gpu)
            handle.remove()
            logits_ms = out_ms.logits
            loss_ms = _cross_entropy(logits_ms, labels)
            kl_ms = _kl_divergence(logits_base, logits_ms, labels)

            # --- TA-swapped (replace TA with speaker B's mean TA) ---
            hook = _make_mean_swap_hook(U_ta, sig_tensors[spk_b]["ta"])
            handle = adapter_module.register_forward_hook(hook)
            with torch.no_grad():
                out_ta = model(**inputs_gpu)
            handle.remove()
            logits_ta = out_ta.logits
            loss_ta = _cross_entropy(logits_ta, labels)
            kl_ta = _kl_divergence(logits_base, logits_ta, labels)

            rows.append({
                "sample_id": sid,
                "speaker_a": spk_a,
                "speaker_b": spk_b,
                "loss_baseline": loss_base,
                "loss_ms_swap": loss_ms if loss_ms is not None else float("nan"),
                "loss_ta_swap": loss_ta if loss_ta is not None else float("nan"),
                "kl_ms_swap": kl_ms,
                "kl_ta_swap": kl_ta,
                "delta_loss_ms": (loss_ms - loss_base) if loss_ms is not None else float("nan"),
                "delta_loss_ta": (loss_ta - loss_base) if loss_ta is not None else float("nan"),
                "delta_ms_norm": delta_ms_norm,
                "delta_ta_norm": delta_ta_norm,
                "kl_ms_normalized": kl_ms / max(delta_ms_norm ** 2, 1e-10),
                "kl_ta_normalized": kl_ta / max(delta_ta_norm ** 2, 1e-10),
            })

            # Free logits
            del logits_base, logits_ms, logits_ta, out_base, out_ms, out_ta

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
    out_dir = ROOT / "results" / "exp1" / "ms_swap"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"ms_swap_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-sample results → {csv_path}")

    # Summary
    kl_ms_vals = [r["kl_ms_swap"] for r in rows if not np.isnan(r["kl_ms_swap"])]
    kl_ta_vals = [r["kl_ta_swap"] for r in rows if not np.isnan(r["kl_ta_swap"])]
    dl_ms_vals = [r["delta_loss_ms"] for r in rows if not np.isnan(r["delta_loss_ms"])]
    dl_ta_vals = [r["delta_loss_ta"] for r in rows if not np.isnan(r["delta_loss_ta"])]
    kl_ms_norm = [r["kl_ms_normalized"] for r in rows if not np.isnan(r["kl_ms_normalized"])]
    kl_ta_norm = [r["kl_ta_normalized"] for r in rows if not np.isnan(r["kl_ta_normalized"])]
    dn_ms_vals = [r["delta_ms_norm"] for r in rows]
    dn_ta_vals = [r["delta_ta_norm"] for r in rows]
    baseline_mean = float(np.mean([r["loss_baseline"] for r in rows]))

    summary = {
        "model": model_name,
        "tag": tag,
        "n_samples": len(rows),
        "n_modes": n_modes,
        "n_ms_modes": n_ms,
        "n_ta_modes": n_ta,
        "n_speakers": len(speakers),
        "mean_loss_baseline": round(baseline_mean, 4),
        "mean_kl_ms_swap": round(float(np.mean(kl_ms_vals)), 6),
        "median_kl_ms_swap": round(float(np.median(kl_ms_vals)), 6),
        "std_kl_ms_swap": round(float(np.std(kl_ms_vals)), 6),
        "mean_kl_ta_swap": round(float(np.mean(kl_ta_vals)), 6),
        "median_kl_ta_swap": round(float(np.median(kl_ta_vals)), 6),
        "std_kl_ta_swap": round(float(np.std(kl_ta_vals)), 6),
        "kl_ratio_ta_over_ms": round(float(np.mean(kl_ta_vals)) / max(float(np.mean(kl_ms_vals)), 1e-10), 2),
        "mean_delta_loss_ms": round(float(np.mean(dl_ms_vals)), 4),
        "mean_delta_loss_ta": round(float(np.mean(dl_ta_vals)), 4),
        "pct_change_ms": round(float(np.mean(dl_ms_vals)) / baseline_mean * 100, 2),
        "pct_change_ta": round(float(np.mean(dl_ta_vals)) / baseline_mean * 100, 2),
        # Perturbation norms
        "mean_delta_ms_norm": round(float(np.mean(dn_ms_vals)), 4),
        "mean_delta_ta_norm": round(float(np.mean(dn_ta_vals)), 4),
        "norm_ratio_ms_over_ta": round(float(np.mean(dn_ms_vals)) / max(float(np.mean(dn_ta_vals)), 1e-10), 2),
        # Normalized KL (sensitivity per unit perturbation²)
        "mean_kl_ms_normalized": round(float(np.mean(kl_ms_norm)), 6),
        "mean_kl_ta_normalized": round(float(np.mean(kl_ta_norm)), 6),
        "normalized_kl_ratio_ta_over_ms": round(
            float(np.mean(kl_ta_norm)) / max(float(np.mean(kl_ms_norm)), 1e-10), 2
        ),
    }

    json_path = out_dir / f"ms_swap_{tag}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {json_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"MS-SWAP EXPERIMENT — {model_name}")
    print(f"{'='*70}")
    print(f"Baseline loss: {baseline_mean:.4f}")
    print(f"\n{'Swap':<12} {'KL div':>12} {'‖δ‖':>10} {'KL/‖δ‖²':>12} {'ΔLoss':>10} {'%Change':>8}")
    print(f"{'-'*70}")
    print(
        f"{'MS-swap':<12} {np.mean(kl_ms_vals):>12.4f} {np.mean(dn_ms_vals):>10.2f}"
        f" {np.mean(kl_ms_norm):>12.6f} {np.mean(dl_ms_vals):>+10.4f}"
        f" {np.mean(dl_ms_vals)/baseline_mean*100:>+7.2f}%"
    )
    print(
        f"{'TA-swap':<12} {np.mean(kl_ta_vals):>12.4f} {np.mean(dn_ta_vals):>10.2f}"
        f" {np.mean(kl_ta_norm):>12.6f} {np.mean(dl_ta_vals):>+10.4f}"
        f" {np.mean(dl_ta_vals)/baseline_mean*100:>+7.2f}%"
    )
    print(f"\nRaw KL ratio (TA/MS):        {np.mean(kl_ta_vals)/max(np.mean(kl_ms_vals), 1e-10):.2f}×")
    print(f"‖δ‖ ratio (MS/TA):           {np.mean(dn_ms_vals)/max(np.mean(dn_ta_vals), 1e-10):.2f}×")
    print(f"Normalized KL ratio (TA/MS): {np.mean(kl_ta_norm)/max(np.mean(kl_ms_norm), 1e-10):.2f}×")
    print(f"{'='*70}")

    return summary


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MS-subspace swap experiment")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-modes", type=int, default=100)
    parser.add_argument("--ms-threshold", type=float, default=0.5)
    args = parser.parse_args()

    run_ms_swap(
        n_samples=args.n_samples,
        n_modes=args.n_modes,
        ms_threshold=args.ms_threshold,
    )

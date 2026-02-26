#!/usr/bin/env python3
"""Estimate distributional distances between modal and text representations.

Computes MMD (RBF kernel), CKA (linear), and Wasserstein-1 distance for each
hook point present in both the modal and text HDF5 files.  CPU-only.

Usage:
    python scripts/05_estimate_wasserstein.py \
        --modal-file data/representations/ultravox_librispeech.h5 \
        --text-file  data/representations/llama_librispeech.h5 \
        --config configs/experiment1.yaml \
        --output-dir results/exp1/wasserstein/
"""

from __future__ import annotations

import modality_collapse.utils.env  # noqa: F401  # set HF_HOME early

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Distributional distance helpers
# ---------------------------------------------------------------------------


def _subsample(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Subsample without replacement; return X unchanged if len(X) <= n."""
    if len(X) <= n:
        return X
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx]


def _median_heuristic(X: np.ndarray, Y: np.ndarray) -> float:
    """Median heuristic for RBF kernel bandwidth."""
    # Use a random subset for efficiency when computing pairwise distances
    combined = np.concatenate([X[:500], Y[:500]], axis=0)
    dists = cdist(combined, combined, metric="sqeuclidean")
    # Take median of upper triangle (non-zero entries)
    upper = dists[np.triu_indices_from(dists, k=1)]
    return float(np.median(upper))


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma_sq: float) -> np.ndarray:
    """RBF kernel matrix K(X, Y) = exp(-||x-y||^2 / (2*sigma^2))."""
    sq_dists = cdist(X, Y, metric="sqeuclidean")
    return np.exp(-sq_dists / (2.0 * sigma_sq))


def compute_mmd2(
    X: np.ndarray,
    Y: np.ndarray,
    sigma_sq: float | None = None,
) -> float:
    """Unbiased estimator of MMD^2 with RBF kernel.

    Args:
        X: (n, d) samples from distribution P.
        Y: (m, d) samples from distribution Q.
        sigma_sq: Kernel bandwidth (squared). If None, uses median heuristic.

    Returns:
        Unbiased MMD^2 estimate.
    """
    if sigma_sq is None:
        sigma_sq = _median_heuristic(X, Y)
    n, m = len(X), len(Y)

    Kxx = _rbf_kernel(X, X, sigma_sq)
    Kyy = _rbf_kernel(Y, Y, sigma_sq)
    Kxy = _rbf_kernel(X, Y, sigma_sq)

    # Unbiased estimator: zero the diagonals
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    mmd2 = (
        Kxx.sum() / (n * (n - 1))
        + Kyy.sum() / (m * (m - 1))
        - 2.0 * Kxy.sum() / (n * m)
    )
    return float(mmd2)


def compute_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two representation matrices.

    Args:
        X: (n, d1) representation matrix.
        Y: (n, d2) representation matrix (same n samples).

    Returns:
        CKA score in [0, 1].
    """
    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # HSIC with linear kernel: HSIC(X,Y) = ||Y^T X||_F^2 / (n-1)^2
    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    XtY = X.T @ Y  # (d1, d2)
    XtX = X.T @ X  # (d1, d1)
    YtY = Y.T @ Y  # (d2, d2)

    hsic_xy = np.sum(XtY ** 2)
    hsic_xx = np.sum(XtX ** 2)
    hsic_yy = np.sum(YtY ** 2)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def compute_w1(
    X: np.ndarray,
    Y: np.ndarray,
    pca_dim: int = 256,
) -> float:
    """Wasserstein-1 distance via POT with PCA pre-processing.

    Args:
        X: (n, d) samples from distribution P.
        Y: (m, d) samples from distribution Q.
        pca_dim: Number of PCA dimensions.

    Returns:
        W_1 distance.
    """
    import ot

    # PCA: fit on combined, transform separately
    combined = np.concatenate([X, Y], axis=0).astype(np.float64)
    actual_dim = min(pca_dim, combined.shape[1], combined.shape[0])
    pca = PCA(n_components=actual_dim)
    pca.fit(combined)
    X_pca = pca.transform(X.astype(np.float64))
    Y_pca = pca.transform(Y.astype(np.float64))

    # Euclidean cost matrix
    M = cdist(X_pca, Y_pca, metric="euclidean").astype(np.float64)

    # Uniform weights
    a = np.ones(len(X_pca), dtype=np.float64) / len(X_pca)
    b = np.ones(len(Y_pca), dtype=np.float64) / len(Y_pca)

    w1 = ot.emd2(a, b, M)
    return float(w1)


def bootstrap_metric(
    X: np.ndarray,
    Y: np.ndarray,
    metric_fn,
    n_bootstrap: int = 200,
    rng: np.random.Generator | None = None,
    n_jobs: int = -1,
    **metric_kwargs,
) -> tuple[float, float, float]:
    """Bootstrap a metric with 95% CI, parallelised across CPU cores.

    Uses ThreadPoolExecutor since numpy/POT release the GIL.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    from concurrent.futures import ThreadPoolExecutor
    import os

    if rng is None:
        rng = np.random.default_rng(42)

    point = metric_fn(X, Y, **metric_kwargs)

    # Pre-generate bootstrap indices for reproducibility
    boot_idx_x = [rng.choice(len(X), size=len(X), replace=True) for _ in range(n_bootstrap)]
    boot_idx_y = [rng.choice(len(Y), size=len(Y), replace=True) for _ in range(n_bootstrap)]

    def _compute_one(i):
        return metric_fn(X[boot_idx_x[i]], Y[boot_idx_y[i]], **metric_kwargs)

    if n_jobs == -1:
        n_jobs = min(os.cpu_count() or 4, n_bootstrap)

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        boot_values = list(executor.map(_compute_one, range(n_bootstrap)))

    boot_values = np.array(boot_values)
    ci_lower = float(np.percentile(boot_values, 2.5))
    ci_upper = float(np.percentile(boot_values, 97.5))
    return point, ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# Datasets that are NOT representations (labels, metadata, etc.)
_NON_REP_KEYS = {
    "sample_ids", "text", "transcript", "speaker_id",
    "emotion", "sound_class", "caption", "question", "answer",
    "object_category", "super_category", "object_count",
    "avg_obj_size", "spatial_spread",
}


def get_hook_points(path: str) -> list[str]:
    """List representation dataset names (hook points) in an HDF5 file."""
    with h5py.File(path, "r") as f:
        return [k for k in f.keys() if k not in _NON_REP_KEYS]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate distributional distances (MMD, CKA, W1) "
        "between modal and text representations."
    )
    parser.add_argument(
        "--modal-file", required=True,
        help="HDF5 file with modal representations (from script 01).",
    )
    parser.add_argument(
        "--text-file", required=True,
        help="HDF5 file with text baseline representations (from script 02).",
    )
    parser.add_argument(
        "--config", default="configs/experiment1.yaml",
        help="Experiment config YAML (for n_subsample, pca_dim, etc.).",
    )
    parser.add_argument(
        "--output-dir", default="results/exp1/wasserstein/",
        help="Directory for output CSV.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # Load config for defaults
    try:
        from modality_collapse.utils.config import load_config
        cfg = load_config(args.config)
        n_subsample = cfg.get("wasserstein", {}).get("n_subsample", 1000)
        pca_dim = cfg.get("wasserstein", {}).get("pca_dim", 256)
        n_bootstrap = cfg.get("wasserstein", {}).get("n_bootstrap", 200)
    except Exception:
        n_subsample = 1000
        pca_dim = 256
        n_bootstrap = 200

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover shared hook points
    modal_hooks = set(get_hook_points(args.modal_file))
    text_hooks = set(get_hook_points(args.text_file))
    shared_hooks = sorted(modal_hooks & text_hooks)

    if not shared_hooks:
        print("ERROR: No shared hook points between modal and text files.")
        print(f"  Modal hooks: {sorted(modal_hooks)}")
        print(f"  Text hooks:  {sorted(text_hooks)}")
        sys.exit(1)

    print(f"Shared hook points: {shared_hooks}")
    print(f"Modal-only hooks:   {sorted(modal_hooks - text_hooks)}")
    print(f"Text-only hooks:    {sorted(text_hooks - modal_hooks)}")
    print(f"Settings: n_subsample={n_subsample}, pca_dim={pca_dim}, "
          f"n_bootstrap={n_bootstrap}")
    print()

    # ------------------------------------------------------------------
    # Compute metrics for each shared hook point
    # ------------------------------------------------------------------
    results: list[dict] = []

    for hook in shared_hooks:
        print(f"--- Hook: {hook} ---")

        from modality_collapse.extraction.storage import load_representations
        X_modal = load_representations(args.modal_file, hook)
        X_text = load_representations(args.text_file, hook)

        # Filter out all-zero rows (unused pre-allocated slots)
        modal_mask = np.any(X_modal != 0, axis=1)
        text_mask = np.any(X_text != 0, axis=1)
        X_modal = X_modal[modal_mask].astype(np.float64)
        X_text = X_text[text_mask].astype(np.float64)

        print(f"  Modal: {X_modal.shape}, Text: {X_text.shape}")

        # --- MMD (RBF, median heuristic) ---
        print("  Computing MMD (RBF kernel) ...")
        X_m_sub = _subsample(X_modal, n_subsample, rng)
        X_t_sub = _subsample(X_text, n_subsample, rng)

        sigma_sq = _median_heuristic(X_m_sub, X_t_sub)
        mmd2_val, mmd2_lo, mmd2_hi = bootstrap_metric(
            X_m_sub, X_t_sub,
            lambda x, y: compute_mmd2(x, y, sigma_sq=sigma_sq),
            n_bootstrap=n_bootstrap, rng=rng,
        )
        print(f"    MMD^2 = {mmd2_val:.6f} [{mmd2_lo:.6f}, {mmd2_hi:.6f}]")
        results.append({
            "hook_point": hook, "metric": "MMD2",
            "value": mmd2_val, "ci_lower": mmd2_lo, "ci_upper": mmd2_hi,
        })

        # --- CKA (linear) ---
        print("  Computing CKA (linear) ...")
        # CKA requires matched sample counts; use min of the two
        n_cka = min(len(X_modal), len(X_text), n_subsample)
        X_m_cka = _subsample(X_modal, n_cka, rng)
        X_t_cka = _subsample(X_text, n_cka, rng)
        cka_val = compute_linear_cka(X_m_cka, X_t_cka)
        print(f"    CKA(modal, text) at {hook} = {cka_val:.4f}")
        results.append({
            "hook_point": hook, "metric": "CKA",
            "value": cka_val, "ci_lower": np.nan, "ci_upper": np.nan,
        })

        # --- W1 via POT ---
        print("  Computing W1 (POT + PCA) ...")
        X_m_w = _subsample(X_modal, n_subsample, rng)
        X_t_w = _subsample(X_text, n_subsample, rng)
        w1_val, w1_lo, w1_hi = bootstrap_metric(
            X_m_w, X_t_w,
            lambda x, y: compute_w1(x, y, pca_dim=pca_dim),
            n_bootstrap=n_bootstrap, rng=rng,
        )
        print(f"    W1 = {w1_val:.4f} [{w1_lo:.4f}, {w1_hi:.4f}]")
        results.append({
            "hook_point": hook, "metric": "W1",
            "value": w1_val, "ci_lower": w1_lo, "ci_upper": w1_hi,
        })
        print()

    # ------------------------------------------------------------------
    # CKA between encoder_output and adapter_output (within modal file)
    # ------------------------------------------------------------------
    cka_check_pass = None
    if "encoder_output" in modal_hooks and "adapter_output" in modal_hooks:
        print("--- Cross-stage CKA within modal file ---")
        X_enc = load_representations(args.modal_file, "encoder_output")
        X_adp = load_representations(args.modal_file, "adapter_output")
        X_enc = X_enc[np.any(X_enc != 0, axis=1)].astype(np.float64)
        X_adp = X_adp[np.any(X_adp != 0, axis=1)].astype(np.float64)

        n_cross = min(len(X_enc), len(X_adp), n_subsample)
        X_enc_s = _subsample(X_enc, n_cross, rng)
        X_adp_s = _subsample(X_adp, n_cross, rng)

        cka_enc_adp = compute_linear_cka(X_enc_s, X_adp_s)
        print(f"  CKA(encoder_output, adapter_output) = {cka_enc_adp:.4f}")
        results.append({
            "hook_point": "encoder_output<->adapter_output",
            "metric": "CKA_cross_stage",
            "value": cka_enc_adp, "ci_lower": np.nan, "ci_upper": np.nan,
        })

        # Retrieve CKA(adapter_output, text) if available
        cka_adp_text_rows = [
            r for r in results
            if r["hook_point"] == "adapter_output" and r["metric"] == "CKA"
        ]
        if cka_adp_text_rows:
            cka_adp_text = cka_adp_text_rows[0]["value"]

            # Compute CKA(encoder_output, text) if encoder_output is shared
            if "encoder_output" in shared_hooks:
                cka_enc_text_rows = [
                    r for r in results
                    if r["hook_point"] == "encoder_output" and r["metric"] == "CKA"
                ]
                if cka_enc_text_rows:
                    cka_enc_text = cka_enc_text_rows[0]["value"]
                    cka_check_pass = cka_adp_text > cka_enc_text
                    print(f"\n  CHECK: CKA(adapter_output, text) = {cka_adp_text:.4f} "
                          f"> CKA(encoder_output, text) = {cka_enc_text:.4f}?")
                    print(f"  {'PASS' if cka_check_pass else 'FAIL'}")
        print()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)
    csv_path = output_dir / "wasserstein_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Pretty-print table
    print("\n" + "=" * 90)
    print(f"{'Hook Point':<35} {'Metric':<18} {'Value':>10} "
          f"{'CI Lower':>10} {'CI Upper':>10}")
    print("=" * 90)
    for _, row in df.iterrows():
        ci_lo = f"{row['ci_lower']:.4f}" if not np.isnan(row["ci_lower"]) else "  --"
        ci_hi = f"{row['ci_upper']:.4f}" if not np.isnan(row["ci_upper"]) else "  --"
        print(f"  {row['hook_point']:<33} {row['metric']:<18} "
              f"{row['value']:>10.4f} {ci_lo:>10} {ci_hi:>10}")
    print("=" * 90)

    if cka_check_pass is not None:
        label = "PASS" if cka_check_pass else "FAIL"
        print(f"\nAdapter alignment check: {label}")


if __name__ == "__main__":
    main()

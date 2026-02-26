#!/usr/bin/env python3
"""Eigendecomposition-based mode alignment analysis.

Computes alignment scores alpha_tilde(u_k) = u_k^T Sigma_T u_k / lambda_k
for each modal eigenmode u_k, measuring how "text-like" each eigendirection is.

CPU-only.

Usage:
    python scripts/07_mode_alignment.py \
        --modal-file data/representations/ultravox_librispeech.h5 \
        --text-file  data/representations/llama_librispeech.h5 \
        --output-dir results/exp1/mode_alignment/
"""

from __future__ import annotations

import modality_collapse.utils.env  # noqa: F401  # set HF_HOME early

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from modality_collapse.extraction.storage import load_representations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hook point names
MODAL_HOOK = "adapter_output"
TEXT_HOOK_CANDIDATES = ["llm_hidden_16", "llm_final"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_NON_REP_KEYS = {
    "sample_ids", "text", "transcript", "speaker_id",
    "emotion", "sound_class", "caption", "question", "answer",
}


def get_hook_points(path: str) -> list[str]:
    """List representation dataset names (hook points) in an HDF5 file."""
    with h5py.File(path, "r") as f:
        return [k for k in f.keys() if k not in _NON_REP_KEYS]


def compute_covariance(X: np.ndarray) -> np.ndarray:
    """Compute the sample covariance matrix.

    Args:
        X: (n, d) data matrix.

    Returns:
        (d, d) covariance matrix.
    """
    X_centered = X - X.mean(axis=0, keepdims=True)
    n = X_centered.shape[0]
    return (X_centered.T @ X_centered) / (n - 1)


def eigendecompose(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose a symmetric matrix, sorted descending by eigenvalue.

    Returns:
        (eigenvalues, eigenvectors) where eigenvectors[:, k] is the k-th
        eigenvector corresponding to eigenvalues[k].
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors


def compute_alignment_scores(
    eigenvalues_M: np.ndarray,
    eigenvectors_M: np.ndarray,
    cov_T: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute alignment score for each modal eigenmode.

    alpha_tilde(u_k) = u_k^T Sigma_T u_k / lambda_k

    Args:
        eigenvalues_M: Modal eigenvalues (K,).
        eigenvectors_M: Modal eigenvectors (d, K), column-major.
        cov_T: Text covariance matrix (d, d).
        eps: Small constant to avoid division by zero.

    Returns:
        Array of alignment scores (K,).
    """
    K = len(eigenvalues_M)
    alpha = np.zeros(K)

    for k in range(K):
        u_k = eigenvectors_M[:, k]  # (d,)
        numerator = u_k @ cov_T @ u_k  # u_k^T Sigma_T u_k
        denominator = max(eigenvalues_M[k], eps)
        alpha[k] = numerator / denominator

    return alpha


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eigendecomposition-based mode alignment analysis."
    )
    parser.add_argument(
        "--modal-file", required=True,
        help="HDF5 file with modal representations.",
    )
    parser.add_argument(
        "--text-file", required=True,
        help="HDF5 file with text baseline representations.",
    )
    parser.add_argument(
        "--modal-hook", default=MODAL_HOOK,
        help=f"Hook point for modal representations (default: {MODAL_HOOK}).",
    )
    parser.add_argument(
        "--text-hook", default=None,
        help="Hook point for text representations. If not specified, tries "
        f"{TEXT_HOOK_CANDIDATES} in order.",
    )
    parser.add_argument(
        "--output-dir", default="results/exp1/mode_alignment/",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--max-modes", type=int, default=None,
        help="Maximum number of eigenmodes to report (default: all).",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Tag for output filenames (e.g. 'ultravox'). If not specified, "
        "tries to infer from modal-file name.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output tag (for model-specific filenames)
    if args.tag:
        tag = args.tag
    else:
        # Infer from modal filename (e.g. "ultravox_librispeech.h5" → "ultravox")
        stem = Path(args.modal_file).stem
        tag = stem.split("_")[0] if "_" in stem else stem
    print(f"Output tag: {tag}")

    # ------------------------------------------------------------------
    # Determine text hook point
    # ------------------------------------------------------------------
    text_hooks = get_hook_points(args.text_file)
    if args.text_hook is not None:
        text_hook = args.text_hook
        if text_hook not in text_hooks:
            print(f"ERROR: text hook '{text_hook}' not found in {args.text_file}.")
            print(f"  Available: {text_hooks}")
            sys.exit(1)
    else:
        text_hook = None
        for candidate in TEXT_HOOK_CANDIDATES:
            if candidate in text_hooks:
                text_hook = candidate
                break
        if text_hook is None:
            print(f"ERROR: none of {TEXT_HOOK_CANDIDATES} found in text file.")
            print(f"  Available: {text_hooks}")
            sys.exit(1)

    print(f"Modal hook: {args.modal_hook}")
    print(f"Text hook:  {text_hook}")
    print()

    # ------------------------------------------------------------------
    # Load representations
    # ------------------------------------------------------------------
    print("[1/4] Loading representations ...")
    X_modal = load_representations(args.modal_file, args.modal_hook)
    X_text = load_representations(args.text_file, text_hook)

    # Filter zero rows (unused pre-allocated slots)
    modal_mask = np.any(X_modal != 0, axis=1)
    text_mask = np.any(X_text != 0, axis=1)
    X_modal = X_modal[modal_mask].astype(np.float64)
    X_text = X_text[text_mask].astype(np.float64)

    print(f"  Modal ({args.modal_hook}): {X_modal.shape}")
    print(f"  Text  ({text_hook}): {X_text.shape}")

    if X_modal.shape[1] != X_text.shape[1]:
        print(f"\n  WARNING: dimension mismatch! "
              f"Modal d={X_modal.shape[1]}, Text d={X_text.shape[1]}.")
        print("  Will project both to the smaller dimension via PCA.")
        from sklearn.decomposition import PCA
        d_min = min(X_modal.shape[1], X_text.shape[1])
        combined = np.concatenate([X_modal, X_text], axis=0)
        pca = PCA(n_components=d_min)
        pca.fit(combined)
        X_modal = pca.transform(X_modal)
        X_text = pca.transform(X_text)
        print(f"  After PCA: Modal {X_modal.shape}, Text {X_text.shape}")

    # ------------------------------------------------------------------
    # Compute covariance matrices
    # ------------------------------------------------------------------
    print("\n[2/4] Computing covariance matrices ...")
    cov_M = compute_covariance(X_modal)
    cov_T = compute_covariance(X_text)
    d = cov_M.shape[0]
    print(f"  Covariance shape: ({d}, {d})")

    # ------------------------------------------------------------------
    # Eigendecompose modal covariance
    # ------------------------------------------------------------------
    print("\n[3/4] Eigendecomposing Sigma_M ...")
    eigenvalues, eigenvectors = eigendecompose(cov_M)

    # Cumulative variance explained
    total_var = eigenvalues.sum()
    cum_var = np.cumsum(eigenvalues) / total_var

    print(f"  Total variance: {total_var:.4f}")
    print(f"  Top 10 eigenvalues: {eigenvalues[:10]}")
    print(f"  Variance explained by top 10: {cum_var[9]:.4f}")
    print(f"  Variance explained by top 50: {cum_var[min(49, d-1)]:.4f}")
    print(f"  Variance explained by top 100: {cum_var[min(99, d-1)]:.4f}")

    # ------------------------------------------------------------------
    # Compute alignment scores
    # ------------------------------------------------------------------
    print("\n[4/4] Computing alignment scores ...")
    alpha = compute_alignment_scores(eigenvalues, eigenvectors, cov_T)

    n_modes = args.max_modes if args.max_modes else d

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    n_high = int(np.sum(alpha[:n_modes] > 0.5))
    n_low = int(np.sum(alpha[:n_modes] < 0.1))
    n_near_one = int(np.sum(np.abs(alpha[:n_modes] - 1.0) < 0.1))

    print(f"\n{'='*70}")
    print("MODE ALIGNMENT SUMMARY")
    print(f"{'='*70}")
    print(f"  Total eigenmodes analysed:    {n_modes}")
    print(f"  Modes with alpha > 0.5:       {n_high} "
          f"({100*n_high/n_modes:.1f}%)")
    print(f"  Modes with alpha < 0.1:       {n_low} "
          f"({100*n_low/n_modes:.1f}%)")
    print(f"  Modes with alpha ~ 1.0 (+-0.1): {n_near_one} "
          f"({100*n_near_one/n_modes:.1f}%)")
    print(f"  Mean alpha (top 10 modes):    {np.mean(alpha[:10]):.4f}")
    print(f"  Mean alpha (top 50 modes):    {np.mean(alpha[:min(50,n_modes)]):.4f}")
    print(f"  Mean alpha (all {n_modes} modes): {np.mean(alpha[:n_modes]):.4f}")
    print(f"{'='*70}")

    # Print top and bottom modes
    print(f"\n  Top 10 most text-aligned modes:")
    top_idx = np.argsort(alpha[:n_modes])[::-1][:10]
    print(f"  {'Mode':>6} {'Eigenvalue':>12} {'Alpha':>8} {'Cum Var':>10}")
    for k in top_idx:
        print(f"  {k:>6d} {eigenvalues[k]:>12.4f} {alpha[k]:>8.4f} "
              f"{cum_var[k]:>10.4f}")

    print(f"\n  Top 10 most modality-specific modes (lowest alpha):")
    bot_idx = np.argsort(alpha[:n_modes])[:10]
    print(f"  {'Mode':>6} {'Eigenvalue':>12} {'Alpha':>8} {'Cum Var':>10}")
    for k in bot_idx:
        print(f"  {k:>6d} {eigenvalues[k]:>12.4f} {alpha[k]:>8.4f} "
              f"{cum_var[k]:>10.4f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    records = []
    for k in range(n_modes):
        records.append({
            "mode_index": k,
            "eigenvalue": eigenvalues[k],
            "alignment_score": alpha[k],
            "cumulative_variance_explained": cum_var[k],
        })
    df = pd.DataFrame(records)

    csv_path = output_dir / f"mode_alignment_{tag}_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Also save raw arrays for figure generation
    npz_path = output_dir / f"mode_alignment_{tag}_arrays.npz"
    np.savez(
        npz_path,
        eigenvalues=eigenvalues[:n_modes],
        alignment_scores=alpha[:n_modes],
        cumulative_variance=cum_var[:n_modes],
    )
    print(f"Raw arrays saved to {npz_path}")

    # Save summary
    summary_path = output_dir / f"mode_alignment_{tag}_summary.csv"
    pd.DataFrame([{
        "total_modes": n_modes,
        "n_alpha_gt_0.5": n_high,
        "n_alpha_lt_0.1": n_low,
        "n_alpha_near_1": n_near_one,
        "mean_alpha_top10": float(np.mean(alpha[:10])),
        "mean_alpha_top50": float(np.mean(alpha[:min(50, n_modes)])),
        "mean_alpha_all": float(np.mean(alpha[:n_modes])),
    }]).to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

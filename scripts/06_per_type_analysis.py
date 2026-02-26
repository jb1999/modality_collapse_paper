#!/usr/bin/env python3
"""Per-information-type decomposition analysis.

Two complementary analyses:
  A) Naive per-type W₁ — intra-modal class separation (baseline)
  B) Directional analysis — decomposes class-separating variation into
     text-aligned vs modality-specific eigenmodes from the mode alignment.
     Uses between-class variance fraction (dimension-free) as primary metric
     and dimension-matched W₁ as validation.

Usage:
    python scripts/06_per_type_analysis.py \
        --modal-files data/representations/ultravox_librispeech.h5 \
                      data/representations/ultravox_cremad.h5 \
                      data/representations/ultravox_esc50.h5 \
        --text-file   data/representations/llama_from_ultravox_librispeech.h5 \
        --probe-results results/exp1/probes/probe_results_ultravox_librispeech.csv \
                        results/exp1/probes/probe_results_ultravox_cremad.csv \
                        results/exp1/probes/probe_results_ultravox_esc50.csv \
        --output-dir  results/exp1/per_type/
"""

from __future__ import annotations

import modality_collapse.utils.env  # noqa: F401  # set HF_HOME early

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from modality_collapse.extraction.storage import load_representations, load_metadata

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INFO_TYPES = ["lexical", "emotion", "speaker", "acoustic"]

LABEL_KEYS = {
    "lexical": "transcript",
    "emotion": "emotion",
    "speaker": "speaker_id",
    "acoustic": "sound_class",
}

MODAL_HOOK = "adapter_output"

_NON_REP_KEYS = {
    "sample_ids", "text", "transcript", "speaker_id",
    "emotion", "sound_class", "caption", "question", "answer",
}

# ---------------------------------------------------------------------------
# Helpers — data loading
# ---------------------------------------------------------------------------


def _subsample(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(X) <= n:
        return X
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx]


def load_labels_for_type(
    hdf5_path: str,
    info_type: str,
    n_samples: int,
) -> np.ndarray | None:
    key = LABEL_KEYS.get(info_type)
    if key is None:
        return None
    try:
        with h5py.File(hdf5_path, "r") as f:
            if key not in f:
                return None
            raw = f[key][:]
    except Exception:
        return None

    labels = np.array([
        x.decode("utf-8") if isinstance(x, bytes) else str(x)
        for x in raw
    ])
    if len(labels) > n_samples:
        labels = labels[:n_samples]
    if info_type == "lexical":
        labels = _derive_word_labels(labels, top_k=50)
    return labels


def _derive_word_labels(transcripts: np.ndarray, top_k: int = 50) -> np.ndarray:
    import re
    from collections import Counter
    word_counter: Counter = Counter()
    tokenised: list[list[str]] = []
    for t in transcripts:
        words = re.findall(r"[a-z]+", t.lower())
        tokenised.append(words)
        word_counter.update(words)
    top_words = {w for w, _ in word_counter.most_common(top_k)}
    labels = np.full(len(transcripts), "__NONE__", dtype=object)
    for i, words in enumerate(tokenised):
        for w in words:
            if w in top_words:
                labels[i] = w
                break
    return labels


def _load_modal_with_labels(
    modal_files: list[str],
    info_type: str,
    hook: str = MODAL_HOOK,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    for modal_file in modal_files:
        labels = load_labels_for_type(modal_file, info_type, 999999)
        if labels is None:
            continue
        X_modal = load_representations(modal_file, hook)
        modal_mask = np.any(X_modal != 0, axis=1)
        X_modal = X_modal[modal_mask].astype(np.float64)
        n_modal = len(X_modal)
        if len(labels) > n_modal:
            labels = labels[:n_modal]
        elif len(labels) < n_modal:
            X_modal = X_modal[:len(labels)]
        valid_mask = labels != "__NONE__"
        if valid_mask.sum() < 20:
            continue
        return X_modal[valid_mask], labels[valid_mask], modal_file
    return None


# ---------------------------------------------------------------------------
# Helpers — W₁ computation
# ---------------------------------------------------------------------------


def compute_w1(X: np.ndarray, Y: np.ndarray, pca_dim: int = 256) -> float:
    """Wasserstein-1 distance via POT with optional PCA pre-processing."""
    import ot
    combined = np.concatenate([X, Y], axis=0).astype(np.float64)
    actual_dim = min(pca_dim, combined.shape[1], combined.shape[0])
    if actual_dim < 1:
        return float("nan")
    if actual_dim < combined.shape[1]:
        pca = PCA(n_components=actual_dim)
        pca.fit(combined)
        X = pca.transform(X.astype(np.float64))
        Y = pca.transform(Y.astype(np.float64))
    else:
        X = X.astype(np.float64)
        Y = Y.astype(np.float64)
    M = cdist(X, Y, metric="euclidean").astype(np.float64)
    a = np.ones(len(X), dtype=np.float64) / len(X)
    b = np.ones(len(Y), dtype=np.float64) / len(Y)
    return float(ot.emd2(a, b, M))


def compute_w1_interclass(
    X: np.ndarray,
    labels: np.ndarray,
    pca_dim: int = 256,
    n_subsample: int = 1000,
    max_pairs: int = 20,
    rng: np.random.Generator | None = None,
) -> float:
    """Mean pairwise W₁ between label groups (inter-class separation)."""
    if rng is None:
        rng = np.random.default_rng(42)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return float("nan")
    if len(unique_labels) <= 6:
        pairs = list(combinations(unique_labels, 2))
    else:
        all_pairs = []
        for i, li in enumerate(unique_labels):
            for lj in unique_labels[i + 1:]:
                all_pairs.append((li, lj))
        idx = rng.choice(len(all_pairs), size=min(max_pairs, len(all_pairs)),
                         replace=False)
        pairs = [all_pairs[i] for i in idx]

    w1_values = []
    for l1, l2 in pairs:
        X1 = X[labels == l1]
        X2 = X[labels == l2]
        if len(X1) < 5 or len(X2) < 5:
            continue
        n_sub = min(n_subsample // 2, len(X1), len(X2))
        X1_s = _subsample(X1, n_sub, rng)
        X2_s = _subsample(X2, n_sub, rng)
        w1_values.append(compute_w1(X1_s, X2_s, pca_dim=pca_dim))

    return float(np.mean(w1_values)) if w1_values else float("nan")


# ---------------------------------------------------------------------------
# Helpers — eigendecomposition & mode alignment
# ---------------------------------------------------------------------------


def eigendecompose(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]


def compute_alignment_scores(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    cov_T: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """α̃(u_k) = u_k^T Σ_T u_k / λ_k for each modal eigenmode."""
    proj = eigenvectors.T @ cov_T @ eigenvectors
    numerator = np.diag(proj)
    denominator = np.maximum(eigenvalues, eps)
    return numerator / denominator


# ---------------------------------------------------------------------------
# Helpers — between-class scatter
# ---------------------------------------------------------------------------


def compute_between_class_scatter(
    X: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Between-class scatter matrix Σ_B = Σ_k n_k (μ_k - μ)(μ_k - μ)^T."""
    mu = X.mean(axis=0)
    unique = np.unique(labels)
    d = X.shape[1]
    S_B = np.zeros((d, d), dtype=np.float64)
    for label in unique:
        mask = labels == label
        n_k = mask.sum()
        mu_k = X[mask].mean(axis=0)
        diff = (mu_k - mu).reshape(-1, 1)
        S_B += n_k * (diff @ diff.T)
    return S_B


def variance_fraction_in_subspace(
    S_B: np.ndarray,
    V: np.ndarray,
) -> float:
    """Fraction of total between-class variance captured by subspace V.

    = trace(V^T S_B V) / trace(S_B)
    """
    total = np.trace(S_B)
    if total < 1e-15:
        return float("nan")
    projected = np.trace(V.T @ S_B @ V)
    return float(projected / total)


# ---------------------------------------------------------------------------
# Part A: Naive per-type W₁
# ---------------------------------------------------------------------------


def run_naive_w1(
    modal_files: list[str],
    delta_tau: dict[str, float],
    pca_dim: int,
    n_subsample: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    print("=" * 70)
    print("PART A: Naive per-type W₁ (inter-class separation)")
    print("=" * 70)

    w1_tau: dict[str, float] = {}
    for info_type in INFO_TYPES:
        print(f"\n  --- {info_type} ---")
        result = _load_modal_with_labels(modal_files, info_type)
        if result is None:
            print(f"    No labels found for {info_type}.")
            w1_tau[info_type] = float("nan")
            continue
        X_modal, labels, src_file = result
        print(f"    Using {src_file} ({len(labels)} samples, "
              f"{len(np.unique(labels))} classes)")
        w1 = compute_w1_interclass(
            X_modal, labels, pca_dim=pca_dim,
            n_subsample=n_subsample, rng=rng,
        )
        w1_tau[info_type] = w1
        print(f"    W1^{info_type} = {w1:.4f}")

    rows = []
    w1_vals, delta_vals = [], []
    for info_type in INFO_TYPES:
        w1 = w1_tau.get(info_type, float("nan"))
        delta = delta_tau.get(info_type, float("nan"))
        rows.append({"info_type": info_type, "W1_tau": w1, "delta_tau": delta})
        if not (np.isnan(w1) or np.isnan(delta)):
            w1_vals.append(w1)
            delta_vals.append(delta)

    df = pd.DataFrame(rows)
    if len(w1_vals) >= 3:
        rho, p = spearmanr(w1_vals, delta_vals)
        print(f"\n  Spearman ρ(W1^τ, Δ_τ) = {rho:.4f}  (p = {p:.4f})")
    return df


# ---------------------------------------------------------------------------
# Part B: Directional analysis (variance fraction + dimension-matched W₁)
# ---------------------------------------------------------------------------


def run_directional_analysis(
    modal_files: list[str],
    text_file: str,
    delta_tau: dict[str, float],
    alpha_threshold: float,
    pca_dim: int,
    n_subsample: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Directional analysis using mode alignment eigendecomposition.

    Primary metric: fraction of between-class variance in modality-specific modes.
    Validation: dimension-matched W₁ (PCA both subspaces to same dim).
    """
    print("\n" + "=" * 70)
    print("PART B: Directional analysis (text-aligned vs modality-specific)")
    print("=" * 70)

    # --- Load representations for eigendecomposition ---
    X_modal_full = None
    modal_src = None
    for mf in modal_files:
        try:
            X = load_representations(mf, MODAL_HOOK)
            mask = np.any(X != 0, axis=1)
            X = X[mask].astype(np.float64)
            if len(X) > 100:
                X_modal_full = X
                modal_src = mf
                break
        except Exception:
            continue

    if X_modal_full is None:
        print("  ERROR: no modal file with sufficient adapter_output found.")
        return pd.DataFrame()

    # Determine text hook
    TEXT_HOOK_CANDIDATES = ["llm_hidden_16", "llm_final"]
    text_hook = None
    with h5py.File(text_file, "r") as f:
        text_keys = [k for k in f.keys() if k not in _NON_REP_KEYS]
    for candidate in TEXT_HOOK_CANDIDATES:
        if candidate in text_keys:
            text_hook = candidate
            break
    if text_hook is None:
        print(f"  ERROR: none of {TEXT_HOOK_CANDIDATES} found in {text_file}.")
        return pd.DataFrame()

    X_text_full = load_representations(text_file, text_hook)
    text_mask = np.any(X_text_full != 0, axis=1)
    X_text_full = X_text_full[text_mask].astype(np.float64)

    print(f"  Modal source: {modal_src} ({X_modal_full.shape})")
    print(f"  Text source:  {text_file} [{text_hook}] ({X_text_full.shape})")

    if X_modal_full.shape[1] != X_text_full.shape[1]:
        d_min = min(X_modal_full.shape[1], X_text_full.shape[1])
        print(f"  Dimension mismatch — projecting to d={d_min} via PCA")
        combined = np.concatenate([X_modal_full, X_text_full], axis=0)
        pca = PCA(n_components=d_min)
        pca.fit(combined)
        X_modal_full = pca.transform(X_modal_full)
        X_text_full = pca.transform(X_text_full)

    # --- Eigendecomposition ---
    print("\n  Computing covariance matrices and eigendecomposition ...")
    X_modal_c = X_modal_full - X_modal_full.mean(axis=0, keepdims=True)
    cov_M = (X_modal_c.T @ X_modal_c) / (len(X_modal_c) - 1)
    X_text_c = X_text_full - X_text_full.mean(axis=0, keepdims=True)
    cov_T = (X_text_c.T @ X_text_c) / (len(X_text_c) - 1)

    eigenvalues, eigenvectors = eigendecompose(cov_M)
    alpha = compute_alignment_scores(eigenvalues, eigenvectors, cov_T)

    # --- Partition modes ---
    ta_mask = alpha > alpha_threshold
    ms_mask = alpha <= alpha_threshold
    n_ta = int(ta_mask.sum())
    n_ms = int(ms_mask.sum())
    print(f"  Alpha threshold: {alpha_threshold}")
    print(f"  Text-aligned modes (α > {alpha_threshold}): {n_ta}")
    print(f"  Modality-specific modes (α ≤ {alpha_threshold}): {n_ms}")

    V_ta = eigenvectors[:, ta_mask]  # (d, n_ta)
    V_ms = eigenvectors[:, ms_mask]  # (d, n_ms)

    if n_ms == 0:
        print("  WARNING: No modality-specific modes found.")
        return pd.DataFrame()

    # Print top modality-specific modes for reference
    ms_indices = np.where(ms_mask)[0]
    print(f"\n  Modality-specific modes (by eigenvalue):")
    print(f"    {'Mode':>6} {'Eigenvalue':>12} {'Alpha':>8} {'Var %':>8}")
    total_var = eigenvalues.sum()
    for idx in ms_indices:
        print(f"    {idx:>6d} {eigenvalues[idx]:>12.6f} {alpha[idx]:>8.4f} "
              f"{100*eigenvalues[idx]/total_var:>7.2f}%")

    # W₁ target dim: match to n_ms for fair comparison
    w1_target_dim = min(n_ms, pca_dim)
    print(f"\n  W₁ dimension-matched to d={w1_target_dim} for fair comparison")

    # --- Per info type ---
    print("\n  Computing per-type directional metrics ...")
    rows = []
    for info_type in INFO_TYPES:
        print(f"\n    --- {info_type} ---")
        result = _load_modal_with_labels(modal_files, info_type)
        if result is None:
            print(f"      No labels found.")
            rows.append({
                "info_type": info_type,
                "var_frac_MS": float("nan"),
                "var_frac_TA": float("nan"),
                "weighted_alpha": float("nan"),
                "W1_TA_matched": float("nan"),
                "W1_MS_matched": float("nan"),
                "ratio_MS_TA_var": float("nan"),
                "ratio_MS_TA_w1": float("nan"),
                "delta_tau": delta_tau.get(info_type, float("nan")),
            })
            continue

        X_typed, labels, src_file = result
        n_samples = len(labels)
        n_classes = len(np.unique(labels))
        print(f"      Source: {src_file} ({n_samples} samples, {n_classes} classes)")

        # --- Between-class scatter ---
        S_B = compute_between_class_scatter(X_typed, labels)
        frac_ms = variance_fraction_in_subspace(S_B, V_ms)
        frac_ta = variance_fraction_in_subspace(S_B, V_ta)

        # --- Weighted alignment score ---
        # α̃_τ = Σ_k c_k * α_k, where c_k = u_k^T S_B u_k / trace(S_B)
        # Tells us: how text-aligned is this type's class-separating variation?
        trace_SB = np.trace(S_B)
        if trace_SB > 1e-15:
            proj_SB = eigenvectors.T @ S_B @ eigenvectors  # (K, K)
            c_k = np.diag(proj_SB) / trace_SB  # per-mode contribution
            weighted_alpha = float(np.sum(c_k * alpha))
        else:
            c_k = np.zeros_like(alpha)
            weighted_alpha = float("nan")

        print(f"      Between-class var fraction in MS modes: {frac_ms:.4f}")
        print(f"      Between-class var fraction in TA modes: {frac_ta:.4f}")
        print(f"      Weighted alignment α̃_τ:                {weighted_alpha:.4f}")

        # Top contributing modes for this info type
        top_modes = np.argsort(c_k)[::-1][:5]
        print(f"      Top modes: {', '.join(f'k={k}(c={c_k[k]:.3f},α={alpha[k]:.3f})' for k in top_modes)}")

        # --- Dimension-matched W₁ ---
        X_ta = X_typed @ V_ta  # (n, n_ta)
        X_ms = X_typed @ V_ms  # (n, n_ms)

        w1_ta = compute_w1_interclass(
            X_ta, labels, pca_dim=w1_target_dim,
            n_subsample=n_subsample, rng=rng,
        )
        w1_ms = compute_w1_interclass(
            X_ms, labels, pca_dim=w1_target_dim,
            n_subsample=n_subsample, rng=rng,
        )

        ratio_var = frac_ms / frac_ta if frac_ta > 1e-10 else float("nan")
        ratio_w1 = w1_ms / w1_ta if w1_ta > 0 else float("nan")
        delta = delta_tau.get(info_type, float("nan"))

        print(f"      W₁(TA, d={w1_target_dim}): {w1_ta:.4f}")
        print(f"      W₁(MS, d={w1_target_dim}): {w1_ms:.4f}")
        print(f"      Variance ratio (MS/TA): {ratio_var:.4f}")
        print(f"      W₁ ratio (MS/TA):       {ratio_w1:.4f}")
        print(f"      Probe Δ_τ:              {delta:+.4f}")

        rows.append({
            "info_type": info_type,
            "var_frac_MS": frac_ms,
            "var_frac_TA": frac_ta,
            "weighted_alpha": weighted_alpha,
            "W1_TA_matched": w1_ta,
            "W1_MS_matched": w1_ms,
            "ratio_MS_TA_var": ratio_var,
            "ratio_MS_TA_w1": ratio_w1,
            "delta_tau": delta,
        })

    df = pd.DataFrame(rows)

    # --- Correlations ---
    print("\n  " + "-" * 50)
    print("  CORRELATION ANALYSIS")
    print("  " + "-" * 50)

    correlations = {}
    valid = df.dropna(subset=["var_frac_MS", "delta_tau"])
    if len(valid) >= 3:
        # Variance fraction vs |delta|
        rho_var, p_var = spearmanr(
            valid["var_frac_MS"].values,
            np.abs(valid["delta_tau"].values),
        )
        print(f"  ρ(var_frac_MS, |Δ_τ|) = {rho_var:.4f}  (p = {p_var:.4f})")
        correlations["var_frac_MS_vs_abs_delta"] = (rho_var, p_var)

        # Variance ratio vs |delta|
        rho_ratio, p_ratio = spearmanr(
            valid["ratio_MS_TA_var"].values,
            np.abs(valid["delta_tau"].values),
        )
        print(f"  ρ(ratio_var, |Δ_τ|)   = {rho_ratio:.4f}  (p = {p_ratio:.4f})")
        correlations["ratio_var_vs_abs_delta"] = (rho_ratio, p_ratio)

    valid_w1 = df.dropna(subset=["ratio_MS_TA_w1", "delta_tau"])
    if len(valid_w1) >= 3:
        rho_w1, p_w1 = spearmanr(
            valid_w1["ratio_MS_TA_w1"].values,
            np.abs(valid_w1["delta_tau"].values),
        )
        print(f"  ρ(ratio_w1, |Δ_τ|)    = {rho_w1:.4f}  (p = {p_w1:.4f})")
        correlations["ratio_w1_vs_abs_delta"] = (rho_w1, p_w1)

    # Weighted alignment: higher α̃_τ → less degradation → negative ρ with Δ
    valid_alpha = df.dropna(subset=["weighted_alpha", "delta_tau"])
    if len(valid_alpha) >= 3:
        rho_alpha, p_alpha = spearmanr(
            valid_alpha["weighted_alpha"].values,
            valid_alpha["delta_tau"].values,
        )
        print(f"  ρ(α̃_τ, Δ_τ)          = {rho_alpha:.4f}  (p = {p_alpha:.4f})")
        correlations["weighted_alpha_vs_signed_delta"] = (rho_alpha, p_alpha)

        rho_alpha_abs, p_alpha_abs = spearmanr(
            valid_alpha["weighted_alpha"].values,
            np.abs(valid_alpha["delta_tau"].values),
        )
        print(f"  ρ(α̃_τ, |Δ_τ|)        = {rho_alpha_abs:.4f}  (p = {p_alpha_abs:.4f})")
        correlations["weighted_alpha_vs_abs_delta"] = (rho_alpha_abs, p_alpha_abs)

    # Signed delta with var fraction
    if len(valid) >= 3:
        rho_signed, p_signed = spearmanr(
            valid["var_frac_MS"].values,
            valid["delta_tau"].values,
        )
        print(f"  ρ(var_frac_MS, Δ_τ)   = {rho_signed:.4f}  (p = {p_signed:.4f})")
        correlations["var_frac_MS_vs_signed_delta"] = (rho_signed, p_signed)

    # Store metadata
    df.attrs["n_ta"] = n_ta
    df.attrs["n_ms"] = n_ms
    df.attrs["alpha_threshold"] = alpha_threshold
    df.attrs["w1_target_dim"] = w1_target_dim
    df.attrs["correlations"] = correlations

    return df


# ---------------------------------------------------------------------------
# Compute probe degradation
# ---------------------------------------------------------------------------


def compute_delta_tau(
    probe_df: pd.DataFrame,
    retention_col: str,
) -> dict[str, float]:
    """Degradation from encoder_output to llm_final (positive = loss)."""
    mean_retention = (
        probe_df.groupby(["hook_point", "info_type"])[retention_col]
        .mean()
        .reset_index()
    )
    delta_tau: dict[str, float] = {}
    for info_type in INFO_TYPES:
        subset = mean_retention[mean_retention["info_type"] == info_type]
        enc_rows = subset[subset["hook_point"] == "encoder_output"]
        fin_rows = subset[subset["hook_point"] == "llm_final"]
        if enc_rows.empty or fin_rows.empty:
            delta_tau[info_type] = float("nan")
            continue
        r_enc = enc_rows[retention_col].values[0]
        r_fin = fin_rows[retention_col].values[0]
        delta_tau[info_type] = r_enc - r_fin  # positive = degradation
    return delta_tau


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-information-type decomposition: naive W₁ and "
        "directional analysis (text-aligned vs modality-specific)."
    )
    parser.add_argument(
        "--modal-files", required=True, nargs="+",
        help="HDF5 files with modal representations (one per dataset).",
    )
    parser.add_argument(
        "--text-file", required=True,
        help="HDF5 file with text baseline representations.",
    )
    parser.add_argument(
        "--probe-results", required=True, nargs="+",
        help="CSV file(s) from probe training (script 03).",
    )
    parser.add_argument(
        "--output-dir", default="results/exp1/per_type/",
        help="Directory for output.",
    )
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--n-subsample", type=int, default=1000)
    parser.add_argument("--alpha-threshold", type=float, default=0.5,
                        help="Threshold for mode partition (default: 0.5).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load probe results
    # ------------------------------------------------------------------
    print("[1/4] Loading probe results ...")
    dfs = []
    for p in args.probe_results:
        df = pd.read_csv(p)
        dfs.append(df)
        print(f"  Loaded {len(df)} rows from {p}")
    probe_df = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(probe_df)} rows")
    print(f"  Info types: {sorted(probe_df['info_type'].unique())}")

    if "retention_ratio" in probe_df.columns:
        retention_col = "retention_ratio"
    elif "accuracy" in probe_df.columns:
        retention_col = "accuracy"
    else:
        print("ERROR: probe CSV must contain 'retention_ratio' or 'accuracy'.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Compute probe degradation
    # ------------------------------------------------------------------
    print("\n[2/4] Computing probe degradation Δ_τ (encoder → llm_final) ...")
    delta_tau = compute_delta_tau(probe_df, retention_col)
    for info_type in INFO_TYPES:
        d = delta_tau.get(info_type, float("nan"))
        if not np.isnan(d):
            print(f"  {info_type}: Δ_τ = {d:+.4f}")
        else:
            print(f"  {info_type}: Δ_τ = NaN")

    # ------------------------------------------------------------------
    # Part A: Naive per-type W₁
    # ------------------------------------------------------------------
    print(f"\n[3/4] Running naive per-type W₁ ...")
    naive_df = run_naive_w1(
        modal_files=args.modal_files,
        delta_tau=delta_tau,
        pca_dim=args.pca_dim,
        n_subsample=args.n_subsample,
        rng=rng,
    )

    # ------------------------------------------------------------------
    # Part B: Directional analysis
    # ------------------------------------------------------------------
    print(f"\n[4/4] Running directional analysis ...")
    dir_df = run_directional_analysis(
        modal_files=args.modal_files,
        text_file=args.text_file,
        delta_tau=delta_tau,
        alpha_threshold=args.alpha_threshold,
        pca_dim=args.pca_dim,
        n_subsample=args.n_subsample,
        rng=rng,
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    naive_csv = output_dir / "per_type_naive.csv"
    naive_df.to_csv(naive_csv, index=False)
    print(f"\nNaive results saved to {naive_csv}")

    if not dir_df.empty:
        dir_csv = output_dir / "per_type_directional.csv"
        dir_df.to_csv(dir_csv, index=False)
        print(f"Directional results saved to {dir_csv}")

        # Build summary
        correlations = dir_df.attrs.get("correlations", {})
        summary = {
            "alpha_threshold": dir_df.attrs.get("alpha_threshold", args.alpha_threshold),
            "n_text_aligned_modes": dir_df.attrs.get("n_ta", -1),
            "n_modality_specific_modes": dir_df.attrs.get("n_ms", -1),
            "w1_target_dim": dir_df.attrs.get("w1_target_dim", -1),
        }
        for corr_name, (rho, p) in correlations.items():
            summary[f"rho_{corr_name}"] = float(rho) if not np.isnan(rho) else None
            summary[f"p_{corr_name}"] = float(p) if not np.isnan(p) else None

        # Naive Spearman
        naive_valid = naive_df.dropna(subset=["W1_tau", "delta_tau"])
        if len(naive_valid) >= 3:
            rho_n, p_n = spearmanr(
                naive_valid["W1_tau"].values,
                naive_valid["delta_tau"].values,
            )
            summary["rho_naive_w1_vs_delta"] = float(rho_n)
            summary["p_naive_w1_vs_delta"] = float(p_n)

        summary_path = output_dir / "per_type_summary.json"
        with open(summary_path, "w") as fp:
            json.dump(summary, fp, indent=2)
        print(f"Summary saved to {summary_path}")

    # ------------------------------------------------------------------
    # Final combined table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMBINED RESULTS")
    print("=" * 70)

    header = (f"{'Type':<10} {'Naive W₁':>10} {'α̃_τ':>8} {'VarFrac MS':>11} "
              f"{'W₁(TA)':>8} {'W₁(MS)':>8} {'Δ_τ':>8}")
    print(f"\n{header}")
    print("-" * len(header))

    def _f(v: float) -> str:
        return f"{v:.4f}" if not np.isnan(v) else "NaN"

    for info_type in INFO_TYPES:
        nr = naive_df[naive_df["info_type"] == info_type]
        nw1 = nr["W1_tau"].values[0] if len(nr) else float("nan")

        if not dir_df.empty:
            dr = dir_df[dir_df["info_type"] == info_type]
            vf_ms = dr["var_frac_MS"].values[0] if len(dr) else float("nan")
            w_alpha = dr["weighted_alpha"].values[0] if len(dr) else float("nan")
            w1_ta = dr["W1_TA_matched"].values[0] if len(dr) else float("nan")
            w1_ms = dr["W1_MS_matched"].values[0] if len(dr) else float("nan")
        else:
            vf_ms = w_alpha = w1_ta = w1_ms = float("nan")

        delta = delta_tau.get(info_type, float("nan"))
        print(f"  {info_type:<8} {_f(nw1):>10} {_f(w_alpha):>8} {_f(vf_ms):>11} "
              f"{_f(w1_ta):>8} {_f(w1_ms):>8} {delta:>+8.4f}")

    print("-" * len(header))
    print()


if __name__ == "__main__":
    main()

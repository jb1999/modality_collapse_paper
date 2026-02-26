#!/usr/bin/env python3
"""Linear probing script for modality collapse experiments.

Trains linear probes (logistic regression) on extracted representations to
measure information retention across the model pipeline:

    lexical >> emotion >> speaker >> acoustic

For each (hook_point x info_type) combination, trains an sklearn
LogisticRegression classifier on z-score normalised representations with
5-fold random seeding and reports accuracy, F1 (macro), chance level, and
retention ratio = (accuracy - chance) / (1 - chance).

Usage:
    python scripts/03_run_probes.py \\
        --representations data/representations/ultravox_cremad.h5 \\
        --config configs/experiment1.yaml \\
        --output-dir results/exp1/probes/

    python scripts/03_run_probes.py \\
        --representations data/representations/ultravox_cremad.h5 \\
        --output-dir results/exp1/probes/ \\
        --n-seeds 3
"""

from __future__ import annotations

import modality_collapse.utils.env  # noqa: F401  # set HF_HOME early

import argparse
import json
import logging
import sys
import warnings
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from modality_collapse.extraction.storage import load_metadata, load_representations  # noqa: E402
from modality_collapse.utils.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Expected hierarchy ordering: text-correlated info retained best.
EXPECTED_HIERARCHY = ["lexical", "emotion", "speaker", "acoustic"]

# Label dataset keys stored inside the HDF5 file.
LABEL_KEYS = {
    # Speech info types
    "lexical": "transcript",
    "emotion": "emotion",
    "speaker": "speaker_id",
    "acoustic": "sound_class",
    # Vision info types
    "object_category": "object_category",
    "super_category": "super_category",
    "answer": "answer",
    "question_type": "question_type",
    # Non-textual vision info types
    "object_count": "object_count",
    "avg_obj_size": "avg_obj_size",
    "spatial_spread": "spatial_spread",
}

# Fallback label keys (tried if primary key not found).
# Allows vision datasets (caption) to work with the same "lexical" info type.
LABEL_KEYS_FALLBACK: dict[str, list[str]] = {
    "lexical": ["caption", "question"],
}


# ---------------------------------------------------------------------------
# HDF5 label loading helpers
# ---------------------------------------------------------------------------


def _load_string_dataset(h5_path: str, dataset_name: str) -> np.ndarray | None:
    """Load a string/bytes dataset from an HDF5 file.

    Returns None if the dataset does not exist.
    """
    with h5py.File(h5_path, "r") as f:
        if dataset_name not in f:
            return None
        raw = f[dataset_name][:]

    # h5py may return bytes; decode to str.
    if raw.dtype.kind == "O" or raw.dtype.kind == "S":
        raw = np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in raw])
    else:
        raw = np.array([str(x) for x in raw])
    return raw


def _load_labels_for_info_type(
    h5_path: str,
    info_type: str,
    n_samples: int,
) -> tuple[np.ndarray | None, LabelEncoder | None]:
    """Load and encode labels for *info_type* from the HDF5 file.

    For ``"lexical"`` labels (transcripts), the top 50 most frequent words
    are extracted and a per-word binary probe is NOT used here -- instead
    we encode words as multi-class targets.  Each transcript is assigned
    the class of its most salient (frequent-but-not-stop) word.

    Returns ``(encoded_labels, label_encoder)`` or ``(None, None)`` when
    the info type is unavailable.
    """
    label_key = LABEL_KEYS.get(info_type)
    if label_key is None:
        return None, None

    raw_labels = _load_string_dataset(h5_path, label_key)

    # Try fallback keys if primary key not found (e.g. "caption" for vision).
    if raw_labels is None and info_type in LABEL_KEYS_FALLBACK:
        for fallback_key in LABEL_KEYS_FALLBACK[info_type]:
            raw_labels = _load_string_dataset(h5_path, fallback_key)
            if raw_labels is not None:
                logger.info(
                    "Using fallback label key '%s' for info_type=%s",
                    fallback_key,
                    info_type,
                )
                break

    if raw_labels is None:
        return None, None

    # Trim to n_samples (HDF5 may have pre-allocated extra rows).
    raw_labels = raw_labels[:n_samples]

    if info_type == "lexical":
        return _encode_lexical_labels(raw_labels)

    # Filter out empty / missing labels.
    valid_mask = np.array([bool(l.strip()) for l in raw_labels])
    if valid_mask.sum() < 20:
        logger.warning(
            "Too few valid labels for info_type=%s (%d found). Skipping.",
            info_type,
            valid_mask.sum(),
        )
        return None, None

    le = LabelEncoder()
    encoded = le.fit_transform(raw_labels[valid_mask])

    # Filter out classes with fewer than 2 members (StratifiedShuffleSplit
    # requires at least 2 per class for train/test splitting).
    class_counts = np.bincount(encoded)
    rare_classes = set(np.where(class_counts < 2)[0])
    if rare_classes:
        n_rare = len(rare_classes)
        logger.info(
            "Filtering %d rare classes (< 2 members) from info_type=%s",
            n_rare,
            info_type,
        )
        # Mark rare-class samples as invalid.
        rare_mask = np.isin(encoded, list(rare_classes))
        encoded[rare_mask] = -1
        # Re-encode remaining classes to consecutive integers.
        keep_mask = encoded >= 0
        if keep_mask.sum() < 20:
            logger.warning(
                "Too few valid samples after rare-class filtering (%d). Skipping.",
                keep_mask.sum(),
            )
            return None, None
        kept_labels = raw_labels[valid_mask][keep_mask]
        le2 = LabelEncoder()
        re_encoded = le2.fit_transform(kept_labels)
        # Build new valid_mask combining both filters.
        valid_indices = np.where(valid_mask)[0][keep_mask]
        full_encoded = np.full(len(raw_labels), -1, dtype=np.int64)
        full_encoded[valid_indices] = re_encoded
        return full_encoded, le2

    # Return full-size arrays with -1 for invalid indices.
    full_encoded = np.full(len(raw_labels), -1, dtype=np.int64)
    full_encoded[valid_mask] = encoded
    return full_encoded, le


def _encode_lexical_labels(
    transcripts: np.ndarray,
    top_k: int = 50,
) -> tuple[np.ndarray | None, LabelEncoder | None]:
    """Encode transcript labels as multi-class: assign each sample the
    class of the most-frequent-word it contains (from the top-K vocabulary).

    Samples that contain no top-K word are marked -1 (excluded from probing).
    """
    # Tokenise: lowercase, split on whitespace, strip punctuation.
    import re

    word_counter: Counter = Counter()
    tokenised: list[list[str]] = []
    for t in transcripts:
        words = re.findall(r"[a-z]+", t.lower())
        tokenised.append(words)
        word_counter.update(words)

    # Take the top-K most frequent words.
    top_words = [w for w, _ in word_counter.most_common(top_k)]
    if len(top_words) < 5:
        logger.warning("Fewer than 5 distinct words in transcripts; skipping lexical probes.")
        return None, None

    word_to_idx = {w: i for i, w in enumerate(top_words)}

    # Assign each sample the label of the first top-K word it contains.
    labels = np.full(len(transcripts), -1, dtype=np.int64)
    for i, words in enumerate(tokenised):
        for w in words:
            if w in word_to_idx:
                labels[i] = word_to_idx[w]
                break

    le = LabelEncoder()
    le.classes_ = np.array(top_words)

    n_valid = (labels >= 0).sum()
    if n_valid < 50:
        logger.warning("Only %d samples matched top-%d words; skipping lexical.", n_valid, top_k)
        return None, None

    logger.info("Lexical: %d / %d samples assigned to top-%d word classes.", n_valid, len(labels), top_k)
    return labels, le


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


def run_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_seeds: int,
    train_ratio: float,
    weight_decay: float,
    max_iter: int,
) -> list[dict]:
    """Train a linear probe with multiple random train/test splits.

    Args:
        X: Feature matrix of shape ``(n_samples, hidden_dim)``.
        y: Integer label vector of shape ``(n_samples,)``.
        n_seeds: Number of random train/test splits.
        train_ratio: Fraction of data used for training.
        weight_decay: L2 regularisation strength (sklearn C = 1/(wd*N)).
        max_iter: Maximum solver iterations.

    Returns:
        List of result dicts, one per seed.
    """
    # Filter out classes with fewer than 2 members (StratifiedShuffleSplit
    # requires at least 2 per class).
    class_counts = np.bincount(y)
    rare_classes = np.where(class_counts < 2)[0]
    if len(rare_classes) > 0:
        keep_mask = ~np.isin(y, rare_classes)
        X = X[keep_mask]
        y = y[keep_mask]
        # Re-encode to consecutive integers.
        le_re = LabelEncoder()
        y = le_re.fit_transform(y)

    n_classes = len(np.unique(y))
    chance = 1.0 / n_classes

    n_train = int(len(y) * train_ratio)
    C_val = 1.0 / (weight_decay * n_train) if weight_decay > 0 else 1e6

    results = []

    sss = StratifiedShuffleSplit(
        n_splits=n_seeds,
        train_size=train_ratio,
        random_state=42,
    )

    for seed_idx, (train_idx, test_idx) in enumerate(sss.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Z-score normalise (fit on train, transform both).
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train logistic regression.
        clf = LogisticRegression(
            max_iter=max_iter,
            C=C_val,
            solver="lbfgs",
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0.0)
        retention = (acc - chance) / (1.0 - chance) if chance < 1.0 else 0.0

        results.append(
            {
                "seed": seed_idx,
                "accuracy": float(acc),
                "f1_macro": float(f1),
                "chance_level": float(chance),
                "retention_ratio": float(retention),
                "n_classes": n_classes,
                "n_train": len(y_train),
                "n_test": len(y_test),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Discovery: list available hook points and labels in an HDF5 file
# ---------------------------------------------------------------------------


def discover_hdf5_contents(h5_path: str) -> tuple[list[str], list[str]]:
    """Return (hook_point_names, label_dataset_names) present in the HDF5 file."""
    hook_points = []
    label_datasets = []
    label_dataset_set = set(LABEL_KEYS.values())
    for fallbacks in LABEL_KEYS_FALLBACK.values():
        label_dataset_set.update(fallbacks)
    # Non-representation datasets to skip when listing hook points.
    skip_keys = {"sample_ids", "text"} | label_dataset_set

    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            if key in label_dataset_set:
                label_datasets.append(key)
            elif key not in skip_keys:
                hook_points.append(key)

    return hook_points, label_datasets


# ---------------------------------------------------------------------------
# Hierarchy check
# ---------------------------------------------------------------------------


def check_hierarchy(summary_df: pd.DataFrame) -> bool:
    """Check whether the retention ratio ordering matches the expected hierarchy.

    The hierarchy is: lexical >> emotion >> speaker >> acoustic.
    We check strict ordering of mean retention ratios across hook points.

    Returns True if PASS, False if FAIL.
    """
    # Aggregate retention ratios by info_type.
    means = (
        summary_df.groupby("info_type")["retention_ratio"]
        .mean()
        .to_dict()
    )

    # Only check types that are present.
    present = [t for t in EXPECTED_HIERARCHY if t in means]
    if len(present) < 2:
        print("\n[Hierarchy check] Cannot evaluate: fewer than 2 info types present.")
        print(f"  Present types: {present}")
        return False

    is_ordered = True
    for i in range(len(present) - 1):
        a, b = present[i], present[i + 1]
        r_a, r_b = means[a], means[b]
        ok = r_a > r_b
        symbol = ">>" if ok else "<="
        status = "OK" if ok else "VIOLATION"
        print(f"  {a} ({r_a:.4f}) {symbol} {b} ({r_b:.4f})  [{status}]")
        if not ok:
            is_ordered = False

    return is_ordered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train linear probes on extracted representations to measure "
            "information retention across the model pipeline."
        ),
    )
    parser.add_argument(
        "--representations",
        required=True,
        help="Path to the HDF5 file produced by script 01 (representations + labels).",
    )
    parser.add_argument(
        "--config",
        default="configs/experiment1.yaml",
        help="Path to the experiment YAML config (default: configs/experiment1.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/exp1/probes/",
        help="Directory to write probe results (default: results/exp1/probes/).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="Override the number of random seeds from config.",
    )
    parser.add_argument(
        "--hook-points",
        nargs="*",
        default=None,
        help="Subset of hook points to probe (default: all found in HDF5).",
    )
    parser.add_argument(
        "--info-types",
        nargs="*",
        default=None,
        help="Subset of info types to probe (default: all with available labels).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Override max_iter for LogisticRegression.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Override model_name in output filenames (e.g. 'ultravox_speaker_lora').",
    )
    args = parser.parse_args()

    # ---- Logging ----------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    h5_path = args.representations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load config ------------------------------------------------------
    cfg = load_config(args.config)
    probing_cfg = cfg.get("probing", {})

    n_seeds = args.n_seeds or probing_cfg.get("n_seeds", 5)
    train_ratio = float(probing_cfg.get("train_ratio", 0.8))
    weight_decay = float(probing_cfg.get("weight_decay", 1e-4))
    max_iter = args.max_iter or int(probing_cfg.get("epochs", 100))
    config_info_types = list(probing_cfg.get("info_types", list(LABEL_KEYS.keys())))

    # ---- Load HDF5 metadata -----------------------------------------------
    logger.info("Loading metadata from %s", h5_path)
    metadata = load_metadata(h5_path)
    model_name = metadata.get("model_name", metadata.get("model", "unknown"))
    if args.tag:
        model_name = args.tag
    dataset_name = metadata.get("dataset_name", metadata.get("dataset", "unknown"))
    n_samples_stored = int(metadata.get("n_samples", 0))

    logger.info(
        "HDF5 metadata: model=%s, dataset=%s, n_samples=%d",
        model_name,
        dataset_name,
        n_samples_stored,
    )

    # ---- Discover what is in the HDF5 file --------------------------------
    hook_points_found, label_datasets_found = discover_hdf5_contents(h5_path)
    logger.info("Hook points found: %s", hook_points_found)
    logger.info("Label datasets found: %s", label_datasets_found)

    # Determine which hook points and info types to probe.
    if args.hook_points:
        hook_points = [hp for hp in args.hook_points if hp in hook_points_found]
    else:
        hook_points = hook_points_found

    if args.info_types:
        info_types = args.info_types
    else:
        info_types = config_info_types

    if not hook_points:
        logger.error("No hook points to probe. Exiting.")
        sys.exit(1)

    logger.info("Will probe hook points: %s", hook_points)
    logger.info("Will probe info types:  %s", info_types)
    logger.info(
        "Settings: n_seeds=%d, train_ratio=%.2f, weight_decay=%.1e, max_iter=%d",
        n_seeds,
        train_ratio,
        weight_decay,
        max_iter,
    )

    # ---- Pre-load labels for each info type --------------------------------
    labels_cache: dict[str, tuple[np.ndarray, LabelEncoder]] = {}
    for info_type in info_types:
        encoded, le = _load_labels_for_info_type(h5_path, info_type, n_samples_stored)
        if encoded is not None and le is not None:
            labels_cache[info_type] = (encoded, le)
            n_valid = (encoded >= 0).sum()
            n_classes = len(le.classes_)
            logger.info(
                "  %s: %d valid samples, %d classes",
                info_type,
                n_valid,
                n_classes,
            )
        else:
            logger.info("  %s: not available in this HDF5 file.", info_type)

    if not labels_cache:
        logger.error("No labels found for any info type. Exiting.")
        sys.exit(1)

    # ---- Run probes -------------------------------------------------------
    all_results: list[dict] = []
    n_total = len(hook_points) * len(labels_cache)
    task_idx = 0

    for hp in hook_points:
        logger.info("Loading representations for hook_point=%s ...", hp)
        X_all = load_representations(h5_path, hp)
        # Trim to actual sample count.
        X_all = X_all[:n_samples_stored]

        # Convert to float32 for sklearn.
        X_all = X_all.astype(np.float32)

        for info_type, (y_encoded, le) in labels_cache.items():
            task_idx += 1
            logger.info(
                "[%d/%d] Probing %s x %s ...",
                task_idx,
                n_total,
                hp,
                info_type,
            )

            # Select only valid samples (y >= 0).
            valid_mask = y_encoded >= 0
            X = X_all[valid_mask]
            y = y_encoded[valid_mask]

            if len(y) < 20:
                logger.warning(
                    "  Skipping %s x %s: only %d valid samples.",
                    hp,
                    info_type,
                    len(y),
                )
                continue

            # Run probes across seeds.
            seed_results = run_probe(
                X=X,
                y=y,
                n_seeds=n_seeds,
                train_ratio=train_ratio,
                weight_decay=weight_decay,
                max_iter=max_iter,
            )

            for res in seed_results:
                res["model"] = model_name
                res["dataset"] = dataset_name
                res["hook_point"] = hp
                res["info_type"] = info_type
                all_results.append(res)

            # Log mean accuracy for this combination.
            mean_acc = np.mean([r["accuracy"] for r in seed_results])
            mean_ret = np.mean([r["retention_ratio"] for r in seed_results])
            logger.info(
                "  => mean accuracy=%.4f, mean retention_ratio=%.4f",
                mean_acc,
                mean_ret,
            )

    if not all_results:
        logger.error("No probe results generated. Exiting.")
        sys.exit(1)

    # ---- Save results to CSV ----------------------------------------------
    results_df = pd.DataFrame(all_results)

    # Reorder columns for readability.
    col_order = [
        "model",
        "dataset",
        "hook_point",
        "info_type",
        "seed",
        "accuracy",
        "f1_macro",
        "chance_level",
        "retention_ratio",
        "n_classes",
        "n_train",
        "n_test",
    ]
    col_order = [c for c in col_order if c in results_df.columns]
    results_df = results_df[col_order]

    csv_path = output_dir / f"probe_results_{model_name}_{dataset_name}.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved detailed results to %s", csv_path)

    # Also save as JSON for programmatic consumption.
    json_path = output_dir / f"probe_results_{model_name}_{dataset_name}.json"
    results_df.to_json(json_path, orient="records", indent=2)
    logger.info("Saved detailed results to %s", json_path)

    # ---- Print summary table ----------------------------------------------
    print("\n" + "=" * 100)
    print("PROBE RESULTS SUMMARY")
    print(f"Model: {model_name}  |  Dataset: {dataset_name}")
    print("=" * 100)

    summary = (
        results_df.groupby(["hook_point", "info_type"])
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_retention=("retention_ratio", "mean"),
            std_retention=("retention_ratio", "std"),
            mean_f1=("f1_macro", "mean"),
            chance=("chance_level", "first"),
            n_classes=("n_classes", "first"),
        )
        .reset_index()
    )

    # Print in tabular format.
    print(
        f"\n{'Hook Point':<20} {'Info Type':<12} {'Accuracy':>12} "
        f"{'Retention':>12} {'F1 (macro)':>12} {'Chance':>8} {'Classes':>8}"
    )
    print("-" * 100)
    for _, row in summary.iterrows():
        print(
            f"{row['hook_point']:<20} {row['info_type']:<12} "
            f"{row['mean_accuracy']:>8.4f} +/- {row['std_accuracy']:.4f}"
            f"{row['mean_retention']:>8.4f} +/- {row['std_retention']:.4f}"
            f"{row['mean_f1']:>12.4f} "
            f"{row['chance']:>8.4f} "
            f"{row['n_classes']:>8d}"
        )

    # Save summary CSV.
    summary_csv = output_dir / f"probe_summary_{model_name}_{dataset_name}.csv"
    summary.to_csv(summary_csv, index=False)
    logger.info("Saved summary to %s", summary_csv)

    # ---- Hierarchy check ---------------------------------------------------
    print("\n" + "=" * 100)
    print("Hierarchy check: Information retention ordering")
    print(f"Expected ordering: {' >> '.join(EXPECTED_HIERARCHY)}")
    print("=" * 100)

    passed = check_hierarchy(results_df)

    if passed:
        print("\n  *** RESULT: PASS ***")
        print("  The information hierarchy is consistent with expected ordering.")
    else:
        print("\n  *** RESULT: FAIL ***")
        print("  The information hierarchy does NOT match expected ordering.")
        print("  Review the results above.")

    print("=" * 100)
    print()

    # Hierarchy check is informational only; results are already saved.
    sys.exit(0)


if __name__ == "__main__":
    main()

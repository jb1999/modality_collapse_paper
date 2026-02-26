#!/usr/bin/env python3
"""Generate formatted tables (LaTeX + terminal) from experiment results.

Supports multi-model results (Ultravox, Qwen2-Audio, etc.) — all models'
data is loaded and shown side by side.

Produces:
  - Table 1: Probe accuracy across hook points x info types (all models/datasets)
  - Table 2: Information retention through the LLM (key finding, per model)
  - Table 3: Cross-dataset Lipschitz comparison (per model)
  - Table 4: Mode alignment summary (per model)

Usage:
    python scripts/08_generate_tables.py --results-dir results/exp1/
"""

from __future__ import annotations

import modality_collapse.utils.env  # noqa: F401  # set HF_HOME early

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INFO_TYPES = [
    "lexical", "emotion", "speaker", "acoustic",
    "object_category", "super_category", "answer", "question_type",
]
HOOK_POINTS = ["encoder_output", "adapter_output", "llm_hidden_16", "llm_final"]
HOOK_SHORT = {
    "encoder_output": "Encoder",
    "adapter_output": "Adapter",
    "llm_hidden_16": "LLM-H16",
    "llm_final": "LLM-Final",
}

MODEL_SHORT = {
    "ultravox": "UVox",
    "qwen2audio": "Q2A",
    "llava": "LLaVA",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_all_probes(probes_dir: Path) -> pd.DataFrame:
    """Load and concatenate all probe result CSVs."""
    csvs = sorted(probes_dir.glob("probe_results_*.csv"))
    if not csvs:
        return pd.DataFrame()
    dfs = [pd.read_csv(c) for c in csvs]
    return pd.concat(dfs, ignore_index=True)


def load_lipschitz_summaries(lipschitz_dir: Path) -> list[dict]:
    """Load all lipschitz summary JSONs."""
    jsons = sorted(lipschitz_dir.glob("lipschitz_summary_*.json"))
    results = []
    for j in jsons:
        with open(j) as f:
            results.append(json.load(f))
    return results


# ---------------------------------------------------------------------------
# Table 1: Probe accuracy
# ---------------------------------------------------------------------------


def generate_table1(probe_df: pd.DataFrame, tables_dir: Path) -> None:
    """Table 1: Probe accuracy across hook points x info types, per model."""
    if probe_df.empty:
        print("  SKIP Table 1: no probe data.")
        return

    stats = (
        probe_df.groupby(["model", "dataset", "hook_point", "info_type"])["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )

    models = sorted(stats["model"].unique())

    # Terminal
    print("\n" + "=" * 110)
    print("TABLE 1: Probe Accuracy (mean +/- std)")
    print("=" * 110)

    header = f"{'Model':<8} {'Type (Dataset)':<22}"
    for hp in HOOK_POINTS:
        header += f"  {HOOK_SHORT[hp]:>14}"
    print(header)
    print("-" * 110)

    for model in models:
        model_stats = stats[stats["model"] == model]
        combos = (
            model_stats.groupby(["dataset", "info_type"])
            .first()
            .reset_index()[["dataset", "info_type"]]
        )
        for _, row in combos.iterrows():
            ds, it = row["dataset"], row["info_type"]
            ms = MODEL_SHORT.get(model, model[:5])
            label = f"{it} ({ds[:5]})"
            line = f"  {ms:<6} {label:<22}"
            for hp in HOOK_POINTS:
                sub = model_stats[
                    (model_stats["dataset"] == ds)
                    & (model_stats["info_type"] == it)
                    & (model_stats["hook_point"] == hp)
                ]
                if sub.empty:
                    line += f"  {'--':>14}"
                else:
                    m, s = sub["mean"].values[0], sub["std"].values[0]
                    line += f"  {m:.3f}+/-{s:.3f}".rjust(14)
            print(line)
        if model != models[-1]:
            print()

    print("=" * 110)

    # LaTeX
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Probe accuracy across processing stages for each model. "
                 r"Linear probes (logistic regression) trained on frozen representations "
                 r"at each hook point. Mean $\pm$ std over 5 seeds.}")
    latex.append(r"\label{tab:probe_accuracy}")
    latex.append(r"\begin{tabular}{ll" + "c" * len(HOOK_POINTS) + "}")
    latex.append(r"\toprule")
    hp_headers = [HOOK_SHORT[hp] for hp in HOOK_POINTS]
    latex.append("Model & Info Type & " + " & ".join(hp_headers) + r" \\")
    latex.append(r"\midrule")

    for model in models:
        model_stats = stats[stats["model"] == model]
        combos = (
            model_stats.groupby(["dataset", "info_type"])
            .first()
            .reset_index()[["dataset", "info_type"]]
        )
        ms = MODEL_SHORT.get(model, model)
        first_row = True
        for _, row in combos.iterrows():
            ds, it = row["dataset"], row["info_type"]
            label = f"{it} ({ds[:5]})"
            model_cell = ms if first_row else ""
            cells = [model_cell, label]
            for hp in HOOK_POINTS:
                sub = model_stats[
                    (model_stats["dataset"] == ds)
                    & (model_stats["info_type"] == it)
                    & (model_stats["hook_point"] == hp)
                ]
                if sub.empty:
                    cells.append("--")
                else:
                    m, s = sub["mean"].values[0], sub["std"].values[0]
                    cells.append(f"${m:.3f} \\pm {s:.3f}$")
            latex.append(" & ".join(cells) + r" \\")
            first_row = False
        if model != models[-1]:
            latex.append(r"\midrule")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    path = tables_dir / "table1_probe_accuracy.tex"
    path.write_text("\n".join(latex))
    print(f"\n  LaTeX saved to {path}")


# ---------------------------------------------------------------------------
# Table 2: Information retention (key finding)
# ---------------------------------------------------------------------------


def generate_table2(probe_df: pd.DataFrame, tables_dir: Path) -> None:
    """Table 2: Information retention through the LLM, per model.

    Shows adapter accuracy, llm_final accuracy, chance level,
    final/chance ratio, and above-chance retention percentage.
    """
    if probe_df.empty:
        print("  SKIP Table 2: no probe data.")
        return

    stats = (
        probe_df.groupby(["model", "dataset", "hook_point", "info_type"])[
            ["accuracy", "chance_level"]
        ]
        .mean()
        .reset_index()
    )

    models = sorted(stats["model"].unique())

    combos = [
        # Speech
        ("librispeech", "lexical"),
        ("cremad", "emotion"),
        ("librispeech", "speaker"),
        ("cremad", "speaker"),
        ("esc50", "acoustic"),
        # Vision
        ("coco", "lexical"),
        ("coco", "object_category"),
        ("coco", "super_category"),
        ("gqa", "lexical"),
        ("gqa", "answer"),
        ("gqa", "question_type"),
    ]

    # Terminal
    print("\n" + "=" * 105)
    print("TABLE 2: Information Retention Through the LLM (Key Finding)")
    print("=" * 105)
    print(
        f"  {'Model':<8} {'Type (Dataset)':<22} {'Adapter':>8} {'LLM-Final':>10} "
        f"{'Chance':>7} {'Final/Chance':>13} {'Retained%':>10}"
    )
    print("-" * 105)

    rows_for_latex = []
    for model in models:
        model_stats = stats[stats["model"] == model]
        ms = MODEL_SHORT.get(model, model[:5])
        for ds, it in combos:
            label = f"{it} ({ds[:5]})"
            adp = model_stats[
                (model_stats["dataset"] == ds)
                & (model_stats["info_type"] == it)
                & (model_stats["hook_point"] == "adapter_output")
            ]
            fin = model_stats[
                (model_stats["dataset"] == ds)
                & (model_stats["info_type"] == it)
                & (model_stats["hook_point"] == "llm_final")
            ]
            if adp.empty or fin.empty:
                continue

            adp_acc = adp["accuracy"].values[0]
            fin_acc = fin["accuracy"].values[0]
            chance = adp["chance_level"].values[0]
            ratio = fin_acc / chance
            adp_above = adp_acc - chance
            fin_above = fin_acc - chance
            retained = 100 * fin_above / adp_above if adp_above > 0.001 else float("nan")

            print(
                f"  {ms:<8} {label:<22} {adp_acc:>8.3f} {fin_acc:>10.3f} "
                f"{chance:>7.3f} {ratio:>12.1f}x {retained:>9.0f}%"
            )
            rows_for_latex.append((ms, label, adp_acc, fin_acc, chance, ratio, retained))
        if model != models[-1]:
            print()

    print("-" * 105)
    print("  Key: Non-text info retained but INACCESSIBLE to frozen decoder.")
    print("=" * 105)

    # LaTeX
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(
        r"\caption{Information retention through the frozen LLM. "
        r"Non-text information (emotion, speaker, acoustic) is carried through "
        r"all transformer layers, "
        r"demonstrating that modality collapse is a \emph{decoding} failure, "
        r"not an encoding failure. Pattern consistent across models.}"
    )
    latex.append(r"\label{tab:retention}")
    latex.append(r"\begin{tabular}{llccccc}")
    latex.append(r"\toprule")
    latex.append(
        r"Model & Info Type & Adapter & LLM-Final & Chance & "
        r"Final/Chance & Retained\% \\"
    )
    latex.append(r"\midrule")

    prev_model = None
    for ms, label, adp_acc, fin_acc, chance, ratio, retained in rows_for_latex:
        if prev_model is not None and ms != prev_model:
            latex.append(r"\midrule")
        ret_str = f"{retained:.0f}\\%" if not np.isnan(retained) else "--"
        model_cell = ms if ms != prev_model else ""
        latex.append(
            f"{model_cell} & {label} & ${adp_acc:.3f}$ & ${fin_acc:.3f}$ & "
            f"${chance:.3f}$ & ${ratio:.1f}\\times$ & {ret_str} \\\\"
        )
        prev_model = ms

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    path = tables_dir / "table2_retention.tex"
    path.write_text("\n".join(latex))
    print(f"\n  LaTeX saved to {path}")


# ---------------------------------------------------------------------------
# Table 3: Cross-dataset Lipschitz
# ---------------------------------------------------------------------------


def generate_table3(lipschitz_summaries: list[dict], tables_dir: Path) -> None:
    """Table 3: Cross-model, cross-dataset Lipschitz comparison."""
    if not lipschitz_summaries:
        print("  SKIP Table 3: no Lipschitz summaries.")
        return

    content_map = {
        "librispeech": "Clean read speech",
        "cremad": "Emotional acted speech",
        "esc50": "Environmental sounds",
    }

    # Sort by model then dataset
    summaries = sorted(lipschitz_summaries, key=lambda s: (s.get("model", ""), s.get("dataset", "")))

    # Terminal
    print("\n" + "=" * 100)
    print("TABLE 3: Cross-Model, Cross-Dataset Lipschitz Comparison")
    print("=" * 100)
    print(
        f"  {'Model':<12} {'Dataset':<14} {'N':>5} {'L_log mean':>11} {'L_log p95':>10} "
        f"{'p95/p5':>7} {'Content':>20}"
    )
    print("-" * 100)

    rows_for_latex = []
    prev_model = None
    for s in summaries:
        model = s.get("model", "unknown")
        ds = s.get("dataset", "unknown")
        n = s.get("n_computed", s.get("n_valid", 0))
        mean = s.get("L_log_mean", float("nan"))
        p95 = s.get("L_log_p95", float("nan"))
        p5 = s.get("L_log_p5", 1e-10)
        variation = p95 / max(p5, 1e-10)
        content = content_map.get(ds, ds)
        ms = MODEL_SHORT.get(model, model[:5])

        if prev_model is not None and model != prev_model:
            print()
        print(
            f"  {ms:<12} {ds:<14} {n:>5} {mean:>11.3f} {p95:>10.3f} "
            f"{variation:>7.1f}x {content:>20}"
        )
        rows_for_latex.append((ms, ds, n, mean, p95, variation, content))
        prev_model = model

    print("=" * 100)

    # LaTeX
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(
        r"\caption{Lipschitz constant $L_{\log}$ of the frozen LLM decoder "
        r"across models and datasets. Qwen2-Audio's LLM backbone is "
        r"$\sim$18$\times$ smoother than Ultravox's, consistent with "
        r"different training regimes.}"
    )
    latex.append(r"\label{tab:lipschitz}")
    latex.append(r"\begin{tabular}{llcccl}")
    latex.append(r"\toprule")
    latex.append(
        r"Model & Dataset & $N$ & $L_{\log}$ (mean) & $L_{\log}$ (p95) & Content \\"
    )
    latex.append(r"\midrule")

    prev_ms = None
    for ms, ds, n, mean, p95, variation, content in rows_for_latex:
        if prev_ms is not None and ms != prev_ms:
            latex.append(r"\midrule")
        model_cell = ms if ms != prev_ms else ""
        latex.append(f"{model_cell} & {ds} & {n} & ${mean:.3f}$ & ${p95:.3f}$ & {content} \\\\")
        prev_ms = ms

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    path = tables_dir / "table3_lipschitz.tex"
    path.write_text("\n".join(latex))
    print(f"\n  LaTeX saved to {path}")


# ---------------------------------------------------------------------------
# Table 4: Mode alignment summary
# ---------------------------------------------------------------------------


def generate_table4(mode_dir: Path, tables_dir: Path) -> None:
    """Table 4: Mode alignment — top eigenmodes with alpha scores, per model."""
    # Find all model-specific mode alignment files
    csvs = sorted(mode_dir.glob("mode_alignment_*_results.csv"))
    if not csvs:
        # Fallback to old non-tagged file
        fallback = mode_dir / "mode_alignment_results.csv"
        if fallback.exists():
            csvs = [fallback]
        else:
            print(f"  SKIP Table 4: no mode alignment data in {mode_dir}")
            return

    for csv_path in csvs:
        # Extract model tag from filename
        stem = csv_path.stem  # e.g. "mode_alignment_ultravox_results"
        parts = stem.replace("mode_alignment_", "").replace("_results", "")
        model_tag = parts if parts != "results" else "unknown"
        ms = MODEL_SHORT.get(model_tag, model_tag)

        df = pd.read_csv(csv_path)
        total_var = df["eigenvalue"].sum()

        # Show top 10 modes by eigenvalue
        top = df.head(10).copy()
        top["var_pct"] = 100 * top["eigenvalue"] / total_var

        n_ms = int((df["alignment_score"] <= 0.5).sum())
        n_ta = int((df["alignment_score"] > 0.5).sum())
        ms_var = df.loc[df["alignment_score"] <= 0.5, "eigenvalue"].sum()
        ms_var_pct = 100 * ms_var / total_var

        # Top modes concentration
        top1_pct = 100 * df.iloc[0]["eigenvalue"] / total_var if len(df) > 0 else 0
        top2_cum = (
            100 * df.iloc[:2]["eigenvalue"].sum() / total_var if len(df) > 1 else top1_pct
        )

        # Terminal
        print(f"\n{'=' * 80}")
        print(f"TABLE 4: Mode Alignment — {ms} (Top 10 Eigenmodes)")
        print(f"{'=' * 80}")
        print(f"  {'Mode':>5} {'Eigenvalue':>11} {'Var %':>7} {'alpha':>8}  Alignment")
        print("-" * 80)

        for _, row in top.iterrows():
            k = int(row["mode_index"])
            ev = row["eigenvalue"]
            vp = row["var_pct"]
            a = row["alignment_score"]
            if a < 0.1:
                atag = "modality-specific"
            elif a < 0.5:
                atag = "weakly MS"
            elif a < 2.0:
                atag = "text-aligned"
            else:
                atag = "text-enriched"
            print(f"  {k:>5d} {ev:>11.4f} {vp:>6.1f}% {a:>8.4f}  {atag}")

        print("-" * 80)
        print(f"  Mode 0: {top1_pct:.1f}% of variance (top 2: {top2_cum:.1f}%)")
        print(
            f"  MS modes (alpha <= 0.5): {n_ms}, carrying {ms_var_pct:.1f}% of variance"
        )
        print(
            f"  TA modes (alpha > 0.5):  {n_ta}, carrying {100 - ms_var_pct:.1f}% of variance"
        )
        print(f"{'=' * 80}")

    # LaTeX — combined table for all models
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(
        r"\caption{Mode alignment profile: top eigenmodes ranked by "
        r"eigenvalue for each model. $\tilde{\alpha}(u_k) = u_k^\top "
        r"\Sigma_T u_k / \lambda_k$ measures text-alignment. "
        r"Both models show dominant modes with $\tilde{\alpha} \approx 0$ "
        r"(purely modality-specific), but Qwen2-Audio's Mode~0 concentrates "
        r"78\% of variance vs Ultravox's 51\%.}"
    )
    latex.append(r"\label{tab:mode_alignment}")
    latex.append(r"\begin{tabular}{lrccl}")
    latex.append(r"\toprule")
    latex.append(r"Model & Mode & Eigenvalue & Var\% & $\tilde{\alpha}$ \\")
    latex.append(r"\midrule")

    for csv_path in csvs:
        stem = csv_path.stem
        parts = stem.replace("mode_alignment_", "").replace("_results", "")
        model_tag = parts if parts != "results" else "unknown"
        ms = MODEL_SHORT.get(model_tag, model_tag)

        df = pd.read_csv(csv_path)
        total_var = df["eigenvalue"].sum()
        top = df.head(5).copy()
        top["var_pct"] = 100 * top["eigenvalue"] / total_var

        first_row = True
        for _, row in top.iterrows():
            k = int(row["mode_index"])
            ev = row["eigenvalue"]
            vp = row["var_pct"]
            a = row["alignment_score"]
            a_str = f"${a:.4f}$"
            if a < 0.1:
                a_str = r"\textbf{" + f"${a:.4f}$" + "}"
            model_cell = ms if first_row else ""
            latex.append(
                f"{model_cell} & {k} & ${ev:.2f}$ & ${vp:.1f}\\%$ & {a_str} \\\\"
            )
            first_row = False

        if csv_path != csvs[-1]:
            latex.append(r"\midrule")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    path = tables_dir / "table4_mode_alignment.tex"
    path.write_text("\n".join(latex))
    print(f"\n  LaTeX saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate formatted tables for the modality collapse paper."
    )
    parser.add_argument(
        "--results-dir",
        default="results/exp1/",
        help="Root results directory (default: results/exp1/).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    probe_df = load_all_probes(results_dir / "probes")
    lipschitz_summaries = load_lipschitz_summaries(results_dir / "lipschitz")

    # Table 1: Probe accuracy
    print("\n" + "#" * 70)
    print("# TABLE 1: Probe Accuracy")
    print("#" * 70)
    generate_table1(probe_df, tables_dir)

    # Table 2: Information retention (key finding)
    print("\n" + "#" * 70)
    print("# TABLE 2: Information Retention (Key Finding)")
    print("#" * 70)
    generate_table2(probe_df, tables_dir)

    # Table 3: Cross-dataset Lipschitz
    print("\n" + "#" * 70)
    print("# TABLE 3: Cross-Dataset Lipschitz")
    print("#" * 70)
    generate_table3(lipschitz_summaries, tables_dir)

    # Table 4: Mode alignment
    print("\n" + "#" * 70)
    print("# TABLE 4: Mode Alignment")
    print("#" * 70)
    generate_table4(results_dir / "mode_alignment", tables_dir)

    print(f"\nAll tables saved to {tables_dir}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# run_prismatic_pipeline.sh — Prismatic model experiments.
#
# Prismatic models (prismatic_dinov2, prismatic_siglip) need a separate venv
# with the prismatic-vlms package.  Run this AFTER run_full_pipeline.sh.
#
# Prerequisites:
#   - run_full_pipeline.sh completed (need vicuna text baseline + COCO labels)
#   - Prismatic venv exists at ~/venvs/prismatic with prismatic-vlms installed
#
# Usage:
#   UV_PROJECT_ENVIRONMENT=~/venvs/prismatic bash scripts/run_prismatic_pipeline.sh
#   UV_PROJECT_ENVIRONMENT=~/venvs/prismatic bash scripts/run_prismatic_pipeline.sh --force
#
# The Prismatic venv has packages (prismatic-vlms, torchvision, timm) that are
# NOT in pyproject.toml.  We use `uv run --no-sync` to prevent uv from removing
# them.  See README.md for venv setup instructions.

set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

RUN="uv run --no-sync python"
R="data/representations"
RESULTS="results/exp1"
DEVICE="${DEVICE:-cuda:0}"

FORCE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

run() {
    local output="$1"; shift
    if [[ $FORCE -eq 1 ]] || [[ ! -e "$output" ]]; then
        echo ">>> $*"
        "$@"
    else
        echo "--- Skipping (exists: $output)"
    fi
}

echo ""
echo "============================================================"
echo "  Prismatic Pipeline"
echo "============================================================"
echo ""

# -- 01: Extract prismatic representations (GPU) -------------------------
run "$R/prismatic_dinov2_coco.h5" $RUN scripts/01_extract_representations.py --model prismatic_dinov2 --dataset coco --device "$DEVICE"
run "$R/prismatic_siglip_coco.h5" $RUN scripts/01_extract_representations.py --model prismatic_siglip --dataset coco --device "$DEVICE"

# -- 02: Vicuna text baseline (should already exist from main pipeline) ---
# The vicuna model needs to be downloaded locally.  If vicuna_from_llava_coco.h5
# already exists from run_full_pipeline.sh, this will be skipped.
run "$R/vicuna_from_llava_coco.h5" $RUN scripts/02_extract_text_baseline.py --source-file "$R/llava_coco.h5" --text-model vicuna --device "$DEVICE"

# -- 10: Add non-textual vision labels to H5 files (CPU) ------------------
# Adds object_count, avg_obj_size, spatial_spread from COCO annotations.
run "$R/.vision_labels_done" $RUN scripts/10_extract_vision_labels.py
touch "$R/.vision_labels_done"

# -- 03: Probes (CPU — works with any venv, just reads HDF5) -------------
VISION_TYPES="lexical object_category super_category object_count avg_obj_size spatial_spread"
run "$RESULTS/probes/probe_summary_prismatic_dinov2_coco.csv" $RUN scripts/03_run_probes.py --representations "$R/prismatic_dinov2_coco.h5" --config configs/experiment1.yaml --output-dir "$RESULTS/probes/" --info-types $VISION_TYPES
run "$RESULTS/probes/probe_summary_prismatic_siglip_coco.csv" $RUN scripts/03_run_probes.py --representations "$R/prismatic_siglip_coco.h5" --config configs/experiment1.yaml --output-dir "$RESULTS/probes/" --info-types $VISION_TYPES

# -- 04: Lipschitz estimation (GPU) --------------------------------------
run "$RESULTS/lipschitz/lipschitz_summary_prismatic_dinov2_coco.json" $RUN scripts/04_estimate_lipschitz.py --model prismatic_dinov2 --representations "$R/prismatic_dinov2_coco.h5" --device "$DEVICE"
run "$RESULTS/lipschitz/lipschitz_summary_prismatic_siglip_coco.json" $RUN scripts/04_estimate_lipschitz.py --model prismatic_siglip --representations "$R/prismatic_siglip_coco.h5" --device "$DEVICE"

# -- 12: Causal ablation (GPU) -------------------------------------------
run "$RESULTS/causal_ablation/causal_ablation_prismatic_dinov2_coco.json" $RUN scripts/12_causal_ablation.py --model prismatic_dinov2
run "$RESULTS/causal_ablation/causal_ablation_prismatic_siglip_coco.json" $RUN scripts/12_causal_ablation.py --model prismatic_siglip

# -- 07: Mode alignment (CPU) ---------------------------------------------
run "$RESULTS/mode_alignment/mode_alignment_prismatic_dinov2_vicuna_summary.csv" \
    $RUN scripts/07_mode_alignment.py \
        --modal-file "$R/prismatic_dinov2_coco.h5" \
        --text-file  "$R/vicuna_from_llava_coco.h5" \
        --tag prismatic_dinov2_vicuna \
        --output-dir "$RESULTS/mode_alignment/"

run "$RESULTS/mode_alignment/mode_alignment_prismatic_siglip_vicuna_summary.csv" \
    $RUN scripts/07_mode_alignment.py \
        --modal-file "$R/prismatic_siglip_coco.h5" \
        --text-file  "$R/vicuna_from_llava_coco.h5" \
        --tag prismatic_siglip_vicuna \
        --output-dir "$RESULTS/mode_alignment/"

echo ""
echo "============================================================"
echo "  Prismatic pipeline complete."
echo "  Run: uv run python scripts/verify_results.py"
echo "============================================================"

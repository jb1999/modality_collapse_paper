#!/usr/bin/env bash
# run_full_pipeline.sh — Full experiment pipeline for modality collapse.
#
# Usage:
#   bash scripts/run_full_pipeline.sh              # skip existing outputs
#   bash scripts/run_full_pipeline.sh --force       # re-run everything
#   bash scripts/run_full_pipeline.sh --phase 3     # run from phase 3
#
# Environment variables:
#   DATA_ROOT  — path to a directory containing raw/ (and optionally
#                representations/, labels/).  Symlinks data/{raw,representations,labels}
#                to the corresponding subdirs under DATA_ROOT.
#   DEVICE     — GPU device (default: cuda:0)
#
# Notes:
#   - Prismatic models (prismatic_dinov2, prismatic_siglip) need a separate
#     venv (~/venvs/prismatic).  They are NOT included here.
#   - Vicuna text baseline needs a local model checkout.

set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

# ── DATA_ROOT symlinks ───────────────────────────────────────────────────────
if [[ -n "${DATA_ROOT:-}" ]]; then
    mkdir -p data
    for subdir in raw representations labels; do
        if [[ -d "$DATA_ROOT/$subdir" ]] && [[ ! -e "data/$subdir" ]]; then
            echo ">>> Symlinking data/$subdir → $DATA_ROOT/$subdir"
            ln -s "$DATA_ROOT/$subdir" "data/$subdir"
        fi
    done
fi

RUN="uv run python"
R="data/representations"
RESULTS="results/exp1"
DEVICE="${DEVICE:-cuda:0}"

FORCE=0
START_PHASE=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --phase) START_PHASE="$2"; shift 2 ;;
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

phase() {
    echo ""
    echo "============================================================"
    echo "  Phase $1: $2"
    echo "============================================================"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Extraction  (GPU, ~3 hrs)
# ═══════════════════════════════════════════════════════════════════════════════
if [[ $START_PHASE -le 1 ]]; then
phase 1 "Extraction"

# -- 01: Multimodal representations ----------------------------------------
# Ultravox × speech datasets
run "$R/ultravox_librispeech.h5"  $RUN scripts/01_extract_representations.py --model ultravox    --dataset librispeech --device "$DEVICE"
run "$R/ultravox_cremad.h5"       $RUN scripts/01_extract_representations.py --model ultravox    --dataset cremad      --device "$DEVICE"
run "$R/ultravox_esc50.h5"        $RUN scripts/01_extract_representations.py --model ultravox    --dataset esc50       --device "$DEVICE"

# Qwen2-Audio × speech datasets
run "$R/qwen2audio_librispeech.h5" $RUN scripts/01_extract_representations.py --model qwen2audio --dataset librispeech --device "$DEVICE"
run "$R/qwen2audio_cremad.h5"      $RUN scripts/01_extract_representations.py --model qwen2audio --dataset cremad      --device "$DEVICE"
run "$R/qwen2audio_esc50.h5"       $RUN scripts/01_extract_representations.py --model qwen2audio --dataset esc50       --device "$DEVICE"

# LLaVA × COCO
run "$R/llava_coco.h5"            $RUN scripts/01_extract_representations.py --model llava       --dataset coco        --device "$DEVICE"

# -- 10: Vision labels (CPU) -----------------------------------------------
run "data/labels/coco_vision_labels.h5"  $RUN scripts/10_extract_vision_labels.py

# -- 02: Text baselines through Llama-3.1-8B --------------------------------
# Only for sources that contain text (transcript or caption).
# ESC-50 has NO text — no text baseline possible.
run "$R/llama_from_ultravox_librispeech.h5"  $RUN scripts/02_extract_text_baseline.py --source-file "$R/ultravox_librispeech.h5"  --device "$DEVICE"
run "$R/llama_from_ultravox_cremad.h5"       $RUN scripts/02_extract_text_baseline.py --source-file "$R/ultravox_cremad.h5"       --device "$DEVICE"
run "$R/llama_from_llava_coco.h5"            $RUN scripts/02_extract_text_baseline.py --source-file "$R/llava_coco.h5"            --device "$DEVICE"

fi  # phase 1

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Analysis  (CPU + GPU, ~1 hr)
# ═══════════════════════════════════════════════════════════════════════════════
if [[ $START_PHASE -le 2 ]]; then
phase 2 "Analysis"

# -- 03: Linear probes -----------------------------------------------------
run "$RESULTS/probes/probe_summary_ultravox_librispeech.csv"   $RUN scripts/03_run_probes.py --representations "$R/ultravox_librispeech.h5"   --config configs/experiment1.yaml --output-dir "$RESULTS/probes/"
run "$RESULTS/probes/probe_summary_ultravox_cremad.csv"        $RUN scripts/03_run_probes.py --representations "$R/ultravox_cremad.h5"        --config configs/experiment1.yaml --output-dir "$RESULTS/probes/"
run "$RESULTS/probes/probe_summary_ultravox_esc50.csv"         $RUN scripts/03_run_probes.py --representations "$R/ultravox_esc50.h5"         --config configs/experiment1.yaml --output-dir "$RESULTS/probes/"
run "$RESULTS/probes/probe_summary_qwen2audio_librispeech.csv" $RUN scripts/03_run_probes.py --representations "$R/qwen2audio_librispeech.h5" --config configs/experiment1.yaml --output-dir "$RESULTS/probes/"
run "$RESULTS/probes/probe_summary_qwen2audio_cremad.csv"      $RUN scripts/03_run_probes.py --representations "$R/qwen2audio_cremad.h5"      --config configs/experiment1.yaml --output-dir "$RESULTS/probes/"
run "$RESULTS/probes/probe_summary_qwen2audio_esc50.csv"       $RUN scripts/03_run_probes.py --representations "$R/qwen2audio_esc50.h5"       --config configs/experiment1.yaml --output-dir "$RESULTS/probes/"
VISION_TYPES="lexical object_category super_category object_count avg_obj_size spatial_spread"
run "$RESULTS/probes/probe_summary_llava_coco.csv"             $RUN scripts/03_run_probes.py --representations "$R/llava_coco.h5"             --config configs/experiment1.yaml --output-dir "$RESULTS/probes/" --info-types $VISION_TYPES

# -- 04: Lipschitz estimation (GPU) ----------------------------------------
run "$RESULTS/lipschitz/lipschitz_summary_ultravox_librispeech.json"   $RUN scripts/04_estimate_lipschitz.py --model ultravox    --representations "$R/ultravox_librispeech.h5"   --device "$DEVICE"
run "$RESULTS/lipschitz/lipschitz_summary_ultravox_cremad.json"        $RUN scripts/04_estimate_lipschitz.py --model ultravox    --representations "$R/ultravox_cremad.h5"        --device "$DEVICE"
run "$RESULTS/lipschitz/lipschitz_summary_ultravox_esc50.json"         $RUN scripts/04_estimate_lipschitz.py --model ultravox    --representations "$R/ultravox_esc50.h5"         --device "$DEVICE"
run "$RESULTS/lipschitz/lipschitz_summary_qwen2audio_librispeech.json" $RUN scripts/04_estimate_lipschitz.py --model qwen2audio  --representations "$R/qwen2audio_librispeech.h5" --device "$DEVICE"
run "$RESULTS/lipschitz/lipschitz_summary_qwen2audio_cremad.json"      $RUN scripts/04_estimate_lipschitz.py --model qwen2audio  --representations "$R/qwen2audio_cremad.h5"      --device "$DEVICE"
run "$RESULTS/lipschitz/lipschitz_summary_qwen2audio_esc50.json"       $RUN scripts/04_estimate_lipschitz.py --model qwen2audio  --representations "$R/qwen2audio_esc50.h5"       --device "$DEVICE"
run "$RESULTS/lipschitz/lipschitz_summary_llava_coco.json"             $RUN scripts/04_estimate_lipschitz.py --model llava       --representations "$R/llava_coco.h5"             --device "$DEVICE"

# -- 05: Wasserstein distances (CPU) ----------------------------------------
run "$RESULTS/wasserstein/wasserstein_results.csv" \
    $RUN scripts/05_estimate_wasserstein.py \
        --modal-file "$R/ultravox_librispeech.h5" \
        --text-file  "$R/llama_from_ultravox_librispeech.h5" \
        --config configs/experiment1.yaml \
        --output-dir "$RESULTS/wasserstein/"

# -- 07: Mode alignment (CPU) -----------------------------------------------
run "$RESULTS/mode_alignment/mode_alignment_ultravox_summary.csv" \
    $RUN scripts/07_mode_alignment.py \
        --modal-file "$R/ultravox_librispeech.h5" \
        --text-file  "$R/llama_from_ultravox_librispeech.h5" \
        --output-dir "$RESULTS/mode_alignment/"

run "$RESULTS/mode_alignment/mode_alignment_qwen2audio_summary.csv" \
    $RUN scripts/07_mode_alignment.py \
        --modal-file "$R/qwen2audio_librispeech.h5" \
        --text-file  "$R/llama_from_ultravox_librispeech.h5" \
        --output-dir "$RESULTS/mode_alignment/"

run "$RESULTS/mode_alignment/mode_alignment_llava_coco_summary.csv" \
    $RUN scripts/07_mode_alignment.py \
        --modal-file "$R/llava_coco.h5" \
        --text-file  "$R/llama_from_llava_coco.h5" \
        --tag llava_coco \
        --output-dir "$RESULTS/mode_alignment/"

# -- 06: Per-type analysis (CPU) --------------------------------------------
run "$RESULTS/per_type/per_type_summary.json" \
    $RUN scripts/06_per_type_analysis.py \
        --modal-files "$R/ultravox_librispeech.h5" \
                      "$R/ultravox_cremad.h5" \
                      "$R/ultravox_esc50.h5" \
        --text-file   "$R/llama_from_ultravox_librispeech.h5" \
        --probe-results "$RESULTS/probes/probe_results_ultravox_librispeech.csv" \
                        "$RESULTS/probes/probe_results_ultravox_cremad.csv" \
                        "$RESULTS/probes/probe_results_ultravox_esc50.csv" \
        --output-dir "$RESULTS/per_type/"

fi  # phase 2

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Causal evidence  (GPU, ~1 hr)
# ═══════════════════════════════════════════════════════════════════════════════
if [[ $START_PHASE -le 3 ]]; then
phase 3 "Causal evidence"

# -- 11: Gradient projection ------------------------------------------------
run "$RESULTS/gradient_projection/gradient_projection_ultravox_librispeech.json" \
    $RUN scripts/11_gradient_projection.py \
        --model ultravox \
        --modal-file "$R/ultravox_librispeech.h5" \
        --text-file  "$R/llama_from_ultravox_librispeech.h5" \
        --device "$DEVICE"

# -- 12: Causal ablation ----------------------------------------------------
run "$RESULTS/causal_ablation/causal_ablation_ultravox_librispeech.json" \
    $RUN scripts/12_causal_ablation.py --model ultravox

# NOTE: prismatic_dinov2 + prismatic_siglip need UV_PROJECT_ENVIRONMENT=~/venvs/prismatic
# and their representation H5 files.  Skipped here; run separately if needed.

# -- 13: MS swap -------------------------------------------------------------
run "$RESULTS/ms_swap/ms_swap_ultravox_librispeech.json" \
    $RUN scripts/13_ms_swap.py

fi  # phase 3

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — LoRA intervention  (GPU, ~3-4 hrs)
# ═══════════════════════════════════════════════════════════════════════════════
if [[ $START_PHASE -le 4 ]]; then
phase 4 "LoRA intervention"

# -- 14: Train emotion LoRA -------------------------------------------------
run "checkpoints/ultravox_emotion_lora/model.safetensors" \
    $RUN scripts/14_train_speaker_lora.py --device "$DEVICE"

# -- 01: Re-extract with LoRA applied ---------------------------------------
run "$R/ultravox_emotion_lora_cremad.h5" \
    $RUN scripts/01_extract_representations.py \
        --model ultravox --dataset cremad --device "$DEVICE" \
        --checkpoint checkpoints/ultravox_emotion_lora \
        --output-tag ultravox_emotion_lora

# -- 03: Probe LoRA representations -----------------------------------------
run "$RESULTS/probes/probe_summary_ultravox_emotion_lora_cremad.csv" \
    $RUN scripts/03_run_probes.py \
        --representations "$R/ultravox_emotion_lora_cremad.h5" \
        --config configs/experiment1.yaml \
        --output-dir "$RESULTS/probes/" \
        --tag ultravox_emotion_lora

fi  # phase 4

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5 — Tables  (CPU)
# ═══════════════════════════════════════════════════════════════════════════════
if [[ $START_PHASE -le 5 ]]; then
phase 5 "Tables"

run "$RESULTS/tables/table1_probe_accuracy.tex" \
    $RUN scripts/08_generate_tables.py --results-dir "$RESULTS/"

fi  # phase 5

echo ""
echo "============================================================"
echo "  Pipeline complete."
echo "  Run: uv run python scripts/verify_results.py"
echo "============================================================"

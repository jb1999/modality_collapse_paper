#!/usr/bin/env bash
# download_data.sh — Download all datasets required for the experiment pipeline.
#
# Datasets:
#   - LibriSpeech test-clean: auto-downloaded by HuggingFace datasets (no action needed)
#   - CREMA-D:  ~600 MB of WAV files from GitHub
#   - ESC-50:   ~650 MB of WAV files from GitHub
#   - COCO:     ~800 MB val2017 images + ~250 MB annotations
#
# Usage:
#   bash scripts/download_data.sh                  # download all to data/raw/
#   bash scripts/download_data.sh --data-dir /path  # download to custom location
#   bash scripts/download_data.sh --only cremad     # download a single dataset

set -euo pipefail

DATA_DIR="data/raw"
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/download_data.sh [--data-dir DIR] [--only DATASET]"
            echo "  DATASET: cremad | esc50 | coco | all (default)"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: $1 is required but not installed."; exit 1; }; }
need wget
need unzip

should_download() {
    [[ -z "$ONLY" ]] || [[ "$ONLY" == "$1" ]]
}

# ── CREMA-D ──────────────────────────────────────────────────────────────────
if should_download cremad; then
    CREMAD_DIR="$DATA_DIR/CREMA-D"
    if [[ -d "$CREMAD_DIR" ]] && ls "$CREMAD_DIR"/*.wav &>/dev/null; then
        echo "--- CREMA-D already exists at $CREMAD_DIR ($(ls "$CREMAD_DIR"/*.wav | wc -l) wav files)"
    else
        echo ">>> Downloading CREMA-D (~600 MB) ..."
        mkdir -p "$CREMAD_DIR"
        TMPZIP=$(mktemp /tmp/cremad_XXXX.zip)
        wget -q --show-progress -O "$TMPZIP" \
            "https://github.com/CheyneyComputerScience/CREMA-D/archive/refs/heads/master.zip"
        echo ">>> Extracting WAV files ..."
        unzip -q -j "$TMPZIP" "CREMA-D-master/AudioWAV/*" -d "$CREMAD_DIR"
        rm "$TMPZIP"
        echo "    CREMA-D: $(ls "$CREMAD_DIR"/*.wav | wc -l) wav files in $CREMAD_DIR"
    fi
fi

# ── ESC-50 ───────────────────────────────────────────────────────────────────
if should_download esc50; then
    ESC50_DIR="$DATA_DIR/ESC-50"
    if [[ -d "$ESC50_DIR/audio" ]] && [[ -f "$ESC50_DIR/meta/esc50.csv" ]]; then
        echo "--- ESC-50 already exists at $ESC50_DIR"
    else
        echo ">>> Downloading ESC-50 (~650 MB) ..."
        mkdir -p "$ESC50_DIR"
        TMPZIP=$(mktemp /tmp/esc50_XXXX.zip)
        wget -q --show-progress -O "$TMPZIP" \
            "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
        echo ">>> Extracting ..."
        # Extract audio/ and meta/ from the archive
        TMPDIR=$(mktemp -d /tmp/esc50_extract_XXXX)
        unzip -q "$TMPZIP" -d "$TMPDIR"
        cp -r "$TMPDIR"/ESC-50-master/audio "$ESC50_DIR/"
        cp -r "$TMPDIR"/ESC-50-master/meta "$ESC50_DIR/"
        rm -rf "$TMPDIR" "$TMPZIP"
        echo "    ESC-50: $(ls "$ESC50_DIR"/audio/*.wav | wc -l) wav files in $ESC50_DIR"
    fi
fi

# ── COCO val2017 ─────────────────────────────────────────────────────────────
if should_download coco; then
    COCO_DIR="$DATA_DIR/COCO"
    NEED_IMAGES=0
    NEED_ANNOT=0
    [[ -d "$COCO_DIR/val2017" ]] || NEED_IMAGES=1
    [[ -f "$COCO_DIR/annotations/captions_val2017.json" ]] || NEED_ANNOT=1

    if [[ $NEED_IMAGES -eq 0 ]] && [[ $NEED_ANNOT -eq 0 ]]; then
        echo "--- COCO val2017 already exists at $COCO_DIR"
    else
        mkdir -p "$COCO_DIR"

        if [[ $NEED_IMAGES -eq 1 ]]; then
            echo ">>> Downloading COCO val2017 images (~800 MB) ..."
            wget -q --show-progress -O "$COCO_DIR/val2017.zip" \
                "http://images.cocodataset.org/zips/val2017.zip"
            echo ">>> Extracting images ..."
            unzip -q "$COCO_DIR/val2017.zip" -d "$COCO_DIR"
            rm "$COCO_DIR/val2017.zip"
        fi

        if [[ $NEED_ANNOT -eq 1 ]]; then
            echo ">>> Downloading COCO annotations (~250 MB) ..."
            wget -q --show-progress -O "$COCO_DIR/annotations.zip" \
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            echo ">>> Extracting annotations ..."
            unzip -q "$COCO_DIR/annotations.zip" -d "$COCO_DIR"
            rm "$COCO_DIR/annotations.zip"
        fi

        echo "    COCO: $(ls "$COCO_DIR"/val2017/*.jpg | wc -l) images in $COCO_DIR"
    fi
fi

echo ""
echo "============================================================"
echo "  Data download complete."
echo ""
echo "  LibriSpeech test-clean will auto-download on first run."
echo "  To run the pipeline: bash scripts/run_full_pipeline.sh"
echo "============================================================"

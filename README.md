# Modality Collapse in Multimodal LLMs

Code for reproducing the experiments in *"Modality Collapse as Mismatched Decoding: Information-Theoretic Limits of Multimodal LLMs"*. When a frozen LLM processes non-text inputs through an adapter, information is lost not because the adapter fails to encode it, but because the LLM decoder was never trained to extract it. We formalize this as a mismatched decoder problem from information theory and validate it across speech and vision models.

## Setup

```bash
# Install dependencies (requires Python >= 3.10, CUDA 12.1)
pip install uv  # if not installed
uv sync

# Or with pip
pip install -e .
```

### HuggingFace Authentication

Some models require authentication. Set your token:
```bash
huggingface-cli login
```

### Data Download

Download all required datasets (~2.3 GB total):

```bash
bash scripts/download_data.sh
```

This downloads CREMA-D, ESC-50, and COCO val2017 to `data/raw/`. LibriSpeech test-clean is auto-downloaded by HuggingFace on first run.

To download to a custom location: `bash scripts/download_data.sh --data-dir /path/to/data/raw`
To download a single dataset: `bash scripts/download_data.sh --only cremad`

## Full Reproduction

Run the complete experiment pipeline:

```bash
export UV_PROJECT_ENVIRONMENT=~/venvs/modality_collapse_paper  # optional

# Point DATA_ROOT at a directory containing raw/, representations/, labels/
# (a symlink data → DATA_ROOT will be created automatically)
DATA_ROOT=/path/to/data bash scripts/run_full_pipeline.sh
```

The pipeline is idempotent — it skips steps whose outputs already exist. Use `--force` to re-run everything, or `--phase N` to start from phase N.

All scripts use relative `data/` paths. If your raw data lives elsewhere, set `DATA_ROOT` and the pipeline creates a symlink. Alternatively, create the symlink yourself: `ln -s /path/to/data data`.

### Pipeline Phases

| Phase | Scripts | GPU? | Description |
|-------|---------|------|-------------|
| 1 | 01, 02, 10 | Yes | Extract representations from all models |
| 2 | 03-07 | Partial | Probes, Lipschitz, Wasserstein, mode alignment |
| 3 | 11-13 | Yes | Gradient projection, causal ablation, MS swap |
| 4 | 14 | Yes | Train emotion LoRA + re-extract + re-probe |
| 5 | 08 | No | Generate tables |

### Prismatic Models (Separate Venv)

Prismatic vision models (prismatic_dinov2, prismatic_siglip) require the [`prismatic-vlms`](https://github.com/TRI-ML/prismatic-vlms) package, which adds `torchvision` and `timm` as dependencies. These are **not** in `pyproject.toml` because the main pipeline does not need them. Prismatic therefore uses its own venv.

**Setting up the Prismatic venv:**

```bash
# Create the venv (Python 3.12 recommended)
uv venv --python 3.12 ~/venvs/prismatic

# Install the main project into it
uv pip install --python ~/venvs/prismatic/bin/python -e .

# Install prismatic-vlms from GitHub (brings in torchvision, timm)
uv pip install --python ~/venvs/prismatic/bin/python \
    "prismatic @ git+https://github.com/TRI-ML/prismatic-vlms.git"
```

> **Warning:** Do **not** run `uv sync` against the Prismatic venv (e.g., `UV_PROJECT_ENVIRONMENT=~/venvs/prismatic uv sync`). This will remove `prismatic-vlms`, `torchvision`, and `timm` because they are not in `pyproject.toml`. If you accidentally do this, re-run the `uv pip install` commands above to restore them.

**Running the Prismatic pipeline** (after the main pipeline completes):

```bash
UV_PROJECT_ENVIRONMENT=~/venvs/prismatic bash scripts/run_prismatic_pipeline.sh
```

The script uses `uv run --no-sync` internally to prevent `uv` from removing the manually installed Prismatic packages.

This extracts Prismatic representations, adds non-textual vision labels, runs probes (all vision info types), Lipschitz estimation, causal ablation, and mode alignment for both Prismatic variants.

## Verification

After **both** pipelines complete, verify all results match expected values:

```bash
uv run python scripts/verify_results.py
```

This checks 19 metrics across probes, Lipschitz estimates, causal ablation, mode alignment, MS swap, and gradient projection against expected values with tolerances. All 19 must pass.

## Hardware Requirements

- **GPU**: Single GPU with 24GB VRAM (tested on RTX 4090 and A100-80GB)
- **RAM**: 32GB recommended
- **Disk**: ~50GB for model caches + ~20GB for extracted representations
- Models are loaded one at a time to fit in 24GB VRAM

## Models

| Model | Type | Parameters |
|-------|------|------------|
| Ultravox v0.6 | Speech-LLM | 8B (Llama-3.1-8B backbone) |
| Qwen2-Audio | Speech-LLM | 7B |
| LLaVA v1.5 | Vision-LLM | 7B (Vicuna backbone) |
| Prismatic (DINOv2, SigLIP) | Vision-LLM | 7B (Vicuna backbone) |
| Llama 3.1 8B | Text baseline | 8B |

## Project Structure

```
src/modality_collapse/       # Python package
  models/                    # Model-specific extractors
  extraction/                # Hooks, HDF5 storage, pooling
  probing/                   # Linear probe training/eval
  metrics/                   # Wasserstein, CKA, MMD, Lipschitz
  data/                      # Dataset loaders
  utils/                     # Config, device, environment
scripts/                     # Numbered experiment scripts
configs/                     # Experiment configuration
tests/                       # Regression expected values
```

## License

MIT

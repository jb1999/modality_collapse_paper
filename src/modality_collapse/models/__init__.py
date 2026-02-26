"""Model registry and modality-specific extractors.

Public API
----------
- :func:`get_model_config` / :func:`list_models` -- query the registry
- :class:`UltravoxExtractor`   -- speech (Whisper + Llama-3.1-8B)
- :class:`Qwen2AudioExtractor` -- speech (Qwen2 audio encoder + Qwen2-7B)
- :class:`LlavaExtractor`      -- vision (CLIP + Vicuna-7B)
- :class:`PrismaticExtractor`  -- vision (DINOv2/SigLIP + Vicuna-7B)
- :class:`TextBaselineExtractor` -- text-only (Llama-3.1-8B)
"""

from .registry import ModelConfig, get_model_config, list_models
from .ultravox import UltravoxExtractor
from .qwen2audio import Qwen2AudioExtractor
from .llava import LlavaExtractor
from .text_baseline import TextBaselineExtractor

# Prismatic requires a separate venv (~/venvs/prismatic) with Python 3.12.
# Import lazily to avoid ImportError in the main venv.
try:
    from .prismatic import PrismaticExtractor
except ImportError:
    PrismaticExtractor = None  # type: ignore[misc, assignment]

__all__ = [
    "ModelConfig",
    "get_model_config",
    "list_models",
    "UltravoxExtractor",
    "Qwen2AudioExtractor",
    "LlavaExtractor",
    "PrismaticExtractor",
    "TextBaselineExtractor",
]

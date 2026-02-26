"""Model registry mapping short names to HuggingFace paths and hook points.

Each entry stores enough metadata to load the model, attach activation
hooks at the right module paths, and preprocess inputs for that modality.
Hook points are best-guess defaults; run the inspection script
(``scripts/inspect_model.py``) on an actual checkpoint to verify.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for a single model in the registry.

    Attributes:
        name: Short identifier used throughout the codebase (e.g. "ultravox").
        hf_path: HuggingFace Hub model path or local directory.
        model_class: Name of the ``transformers`` class to instantiate.
        processor_class: Name of the ``transformers`` processor / tokenizer class.
        hook_points: Mapping from logical names (e.g. "adapter_output") to
            the ``nn.Module`` attribute path inside the loaded model.
        modality: Primary input modality -- "speech", "vision", or "text".
        hidden_dim: Dimensionality of the LLM hidden states.
    """

    name: str
    hf_path: str
    model_class: str
    processor_class: str
    hook_points: dict[str, str] = field(default_factory=dict)
    modality: str = "text"
    hidden_dim: int = 4096


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelConfig] = {
    # ------------------------------------------------------------------
    # Speech: Ultravox  (Llama-3.1-8B backbone + Whisper audio tower)
    # ------------------------------------------------------------------
    "ultravox": ModelConfig(
        name="ultravox",
        hf_path="fixie-ai/ultravox-v0_6-llama-3_1-8b",
        model_class="UltravoxModel",
        processor_class="UltravoxProcessor",
        modality="speech",
        hidden_dim=4096,
        hook_points={
            "encoder_output": "audio_tower.layer_norm",
            "adapter_output": "multi_modal_projector.ln_post",
            "llm_hidden_16": "language_model.model.layers.16",
            "llm_final": "language_model.model.norm",
        },
    ),
    # ------------------------------------------------------------------
    # Speech: Qwen2-Audio  (Qwen2-7B backbone + audio encoder)
    # ------------------------------------------------------------------
    "qwen2audio": ModelConfig(
        name="qwen2audio",
        hf_path="Qwen/Qwen2-Audio-7B-Instruct",
        model_class="Qwen2AudioForConditionalGeneration",
        processor_class="AutoProcessor",
        modality="speech",
        hidden_dim=4096,
        hook_points={
            "encoder_output": "audio_tower.layer_norm",
            "adapter_output": "multi_modal_projector.linear",
            "llm_hidden_16": "language_model.model.layers.16",
            "llm_final": "language_model.model.norm",
        },
    ),
    # ------------------------------------------------------------------
    # Vision: LLaVA-v1.5  (Vicuna-7B backbone + CLIP vision tower)
    # ------------------------------------------------------------------
    "llava": ModelConfig(
        name="llava",
        hf_path="llava-hf/llava-1.5-7b-hf",
        model_class="LlavaForConditionalGeneration",
        processor_class="AutoProcessor",
        modality="vision",
        hidden_dim=4096,
        hook_points={
            "encoder_output": "model.vision_tower.vision_model.post_layernorm",
            "adapter_output": "model.multi_modal_projector",
            "llm_hidden_16": "model.language_model.layers.16",
            "llm_final": "model.language_model.norm",
        },
    ),
    # ------------------------------------------------------------------
    # Vision: Prismatic DINOv2  (Vicuna-7B + DINOv2, NO text alignment)
    # ------------------------------------------------------------------
    "prismatic_dinov2": ModelConfig(
        name="prismatic_dinov2",
        hf_path="TRI-ML/prismatic-vlms",
        model_class="PrismaticVLM",
        processor_class="None",
        modality="vision",
        hidden_dim=4096,
        hook_points={
            "encoder_output": "vision_backbone",
            "adapter_output": "projector.projector.2",
            "llm_hidden_16": "llm_backbone.llm.model.layers.16",
            "llm_final": "llm_backbone.llm.model.norm",
        },
    ),
    # ------------------------------------------------------------------
    # Vision: Prismatic SigLIP  (Vicuna-7B + SigLIP, text-aligned control)
    # ------------------------------------------------------------------
    "prismatic_siglip": ModelConfig(
        name="prismatic_siglip",
        hf_path="TRI-ML/prismatic-vlms",
        model_class="PrismaticVLM",
        processor_class="None",
        modality="vision",
        hidden_dim=4096,
        hook_points={
            "encoder_output": "vision_backbone",
            "adapter_output": "projector.projector.2",
            "llm_hidden_16": "llm_backbone.llm.model.layers.16",
            "llm_final": "llm_backbone.llm.model.norm",
        },
    ),
    # ------------------------------------------------------------------
    # Text baseline: Llama-3.1-8B  (no adapter / encoder)
    # ------------------------------------------------------------------
    "llama": ModelConfig(
        name="llama",
        hf_path="meta-llama/Llama-3.1-8B",
        model_class="AutoModelForCausalLM",
        processor_class="AutoTokenizer",
        modality="text",
        hidden_dim=4096,
        hook_points={
            "llm_hidden_16": "model.layers.16",
            "llm_final": "model.norm",
        },
    ),
    # ------------------------------------------------------------------
    # Text baseline: Vicuna-7B-v1.5  (Prismatic's LLM backbone)
    # ------------------------------------------------------------------
    "vicuna": ModelConfig(
        name="vicuna",
        hf_path="lmsys/vicuna-7b-v1.5",
        model_class="AutoModelForCausalLM",
        processor_class="AutoTokenizer",
        modality="text",
        hidden_dim=4096,
        hook_points={
            "llm_hidden_16": "model.layers.16",
            "llm_final": "model.norm",
        },
    ),
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_model_config(name: str) -> ModelConfig:
    """Return the :class:`ModelConfig` for *name*.

    Args:
        name: A key in :data:`MODEL_REGISTRY` (e.g. ``"ultravox"``).

    Returns:
        The corresponding :class:`ModelConfig`.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(
            f"Unknown model '{name}'. Available models: {available}"
        )
    return MODEL_REGISTRY[name]


def list_models() -> list[str]:
    """Return a sorted list of all registered model names."""
    return sorted(MODEL_REGISTRY)

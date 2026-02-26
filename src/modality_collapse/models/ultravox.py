"""Ultravox model loader and audio preprocessor.

Ultravox (fixie-ai) pairs a Whisper audio tower with a Llama-3.1-8B
language model connected via a multi-modal projector.  This module
handles loading the checkpoint and converting raw audio waveforms into
the dict of tensors that the model's forward pass expects.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from .registry import get_model_config, ModelConfig

logger = logging.getLogger(__name__)


class UltravoxExtractor:
    """Handles loading Ultravox and preprocessing audio for extraction.

    Typical usage::

        ext = UltravoxExtractor(device=torch.device("cuda:0"))
        model, processor = ext.load()
        inputs = ext.preprocess(audio_array, sample_rate)
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        checkpoint_path: str | None = None,
    ) -> None:
        self.config: ModelConfig = get_model_config("ultravox")
        self.device = device
        self.dtype = dtype
        self.checkpoint_path = checkpoint_path
        self.model: Any | None = None
        self.processor: Any | None = None
        self._tokenizer: Any | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> tuple[Any, Any]:
        """Load Ultravox via ``transformers.pipeline``.

        Ultravox uses custom model code (``trust_remote_code=True``) that
        registers under ``AutoModel`` via its ``auto_map``.  Loading through
        ``pipeline()`` is the proven approach (used in interspeech2026).

        The ``_init_weights`` attribute may be missing from newer
        ``transformers`` — we patch it in if needed.

        Returns:
            A ``(model, processor)`` tuple.  The model is in eval mode.
        """
        import transformers
        import transformers.modeling_utils as _mu

        # Patch _init_weights if missing (Ultravox custom code needs it)
        if not hasattr(_mu, "_init_weights"):
            _mu._init_weights = True
            logger.info("Patched transformers.modeling_utils._init_weights")

        logger.info("Loading %s via pipeline ...", self.config.hf_path)
        pipe = transformers.pipeline(
            model=self.config.hf_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=self.dtype,
        )

        self.model = pipe.model

        # Apply LoRA checkpoint if provided
        if self.checkpoint_path is not None:
            self._apply_checkpoint(self.checkpoint_path)

        self.model.eval()

        # Extract processor and tokenizer from pipeline
        self.processor = getattr(pipe, "feature_extractor", None)
        if self.processor is None:
            self.processor = getattr(pipe, "processor", None)
        if self.processor is None:
            # Fallback: load processor separately
            self.processor = transformers.AutoProcessor.from_pretrained(
                self.config.hf_path, trust_remote_code=True
            )

        self._tokenizer = getattr(pipe, "tokenizer", None)
        if self._tokenizer is None:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.config.hf_path, trust_remote_code=True
            )

        logger.info(
            "Loaded %s on %s (%s).",
            self.config.name,
            self.device,
            self.dtype,
        )
        return self.model, self.processor

    # ------------------------------------------------------------------
    # Checkpoint / LoRA loading
    # ------------------------------------------------------------------

    def _apply_checkpoint(self, checkpoint_path: str) -> None:
        """Apply fine-tuned checkpoint weights, with LoRA support."""
        from pathlib import Path
        from safetensors.torch import load_file

        ckpt = Path(checkpoint_path)
        weights = load_file(ckpt / "model.safetensors")
        has_lora = any("lora" in k.lower() for k in weights)

        if has_lora:
            logger.info("Applying LoRA weights from %s", checkpoint_path)
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model.language_model = get_peft_model(
                self.model.language_model, lora_config
            )

        state = self.model.state_dict()
        n_updated = 0
        for key, value in weights.items():
            if key in state and state[key].shape == value.shape:
                state[key] = value.to(state[key].dtype)
                n_updated += 1
        self.model.load_state_dict(state)
        logger.info(
            "Loaded %d/%d checkpoint tensors (LoRA=%s).",
            n_updated, len(weights), has_lora,
        )

        # Merge LoRA into base weights so hook paths remain valid
        if has_lora:
            self.model.language_model = self.model.language_model.merge_and_unload()
            logger.info("Merged LoRA weights into base model.")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _build_text_prompt(self) -> str:
        """Build a minimal text prompt with ``<|audio|>`` placeholder.

        Ultravox requires a text prompt containing the ``<|audio|>`` token.
        We use a neutral prompt that doesn't bias the model toward any
        particular task, since we only need hidden-state representations.
        """
        if self._tokenizer is None:
            # Fallback if tokenizer not available
            return "<|audio|>"

        messages = [
            {"role": "user", "content": "<|audio|>"},
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    def preprocess(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> dict[str, torch.Tensor]:
        """Convert a single audio waveform into model-ready inputs.

        Ultravox requires both audio and a text prompt containing the
        ``<|audio|>`` placeholder token.

        Args:
            audio: 1-D float array of audio samples.
            sr: Sample rate of *audio* in Hz.

        Returns:
            Dict of tensors suitable for ``model(**inputs)``.
        """
        if self.processor is None:
            raise RuntimeError("Call .load() before .preprocess().")

        text = self._build_text_prompt()
        inputs = self.processor(
            text=text,
            audio=audio,
            sampling_rate=sr,
            return_tensors="pt",
        )
        return {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

    def preprocess_batch(
        self,
        audios: list[np.ndarray],
        srs: list[int],
    ) -> dict[str, torch.Tensor]:
        """Convert a batch of audio waveforms into model-ready inputs.

        Args:
            audios: List of 1-D float arrays (one per utterance).
            srs: Corresponding sample rates.

        Returns:
            Batched dict of tensors with padding applied.
        """
        if self.processor is None:
            raise RuntimeError("Call .load() before .preprocess_batch().")

        text = self._build_text_prompt()
        texts = [text] * len(audios)
        inputs = self.processor(
            text=texts,
            audio=audios,
            sampling_rate=srs[0],
            return_tensors="pt",
            padding=True,
        )
        return {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

"""Qwen2-Audio model loader and audio preprocessor.

Qwen2-Audio-7B-Instruct pairs a Whisper-style audio encoder with the
Qwen2-7B language model via a multi-modal projector.  This module
handles loading the checkpoint and converting raw audio waveforms into
the dict of tensors that the model's forward pass expects.

The processor requires both a text prompt (with ``<|AUDIO|>`` token via
chat template) and audio data.  The text prompt is constructed
automatically using the processor's ``apply_chat_template`` method.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from .registry import get_model_config, ModelConfig

logger = logging.getLogger(__name__)


class Qwen2AudioExtractor:
    """Handles loading Qwen2-Audio and preprocessing audio for extraction.

    Typical usage::

        ext = Qwen2AudioExtractor(device=torch.device("cuda:0"))
        model, processor = ext.load()
        inputs = ext.preprocess(audio_array, sample_rate)
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.config: ModelConfig = get_model_config("qwen2audio")
        self.device = device
        self.dtype = dtype
        self.model: Any | None = None
        self.processor: Any | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> tuple[Any, Any]:
        """Load the Qwen2-Audio model and processor from HuggingFace.

        Uses ``Qwen2AudioForConditionalGeneration`` and ``AutoProcessor``.

        Returns:
            A ``(model, processor)`` tuple.  The model is placed on
            ``self.device`` in eval mode with ``self.dtype`` precision.
        """
        import transformers

        # -- processor --------------------------------------------------
        self.processor = transformers.AutoProcessor.from_pretrained(
            self.config.hf_path,
            trust_remote_code=True,
        )

        # -- model ------------------------------------------------------
        try:
            model_cls = getattr(transformers, self.config.model_class)
        except AttributeError:
            logger.warning(
                "%s not found in transformers, falling back to AutoModel.",
                self.config.model_class,
            )
            model_cls = transformers.AutoModel

        self.model = model_cls.from_pretrained(
            self.config.hf_path,
            dtype=self.dtype,
            device_map={"": self.device},
            trust_remote_code=True,
        )
        self.model.eval()

        logger.info(
            "Loaded %s on %s (%s).",
            self.config.name,
            self.device,
            self.dtype,
        )
        return self.model, self.processor

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _build_text_prompt(self) -> str:
        """Build a minimal text prompt with ``<|AUDIO|>`` placeholder.

        Qwen2-Audio requires a text prompt containing the audio token
        placeholder.  We use a neutral prompt via the chat template so
        the model processes audio without task-specific bias.
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "placeholder"},
                    {"type": "text", "text": "Describe this audio."},
                ],
            },
        ]
        return self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

    def preprocess(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> dict[str, torch.Tensor]:
        """Convert a single audio waveform into model-ready inputs.

        Qwen2-Audio processor requires both a text prompt (with
        ``<|AUDIO|>`` token) and the audio data.  The parameter name
        is ``audio=`` (singular), accepting a single array or a list.

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
            audio=[audio],
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

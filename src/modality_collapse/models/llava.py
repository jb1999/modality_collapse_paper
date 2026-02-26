"""LLaVA model loader and image preprocessor.

LLaVA-v1.5-7b pairs a CLIP vision tower with a Vicuna-7B language
model connected via a multi-modal projector.  This module handles
loading the checkpoint and converting PIL images into the dict of
tensors that the model's forward pass expects.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from PIL import Image

from .registry import get_model_config, ModelConfig

logger = logging.getLogger(__name__)


class LlavaExtractor:
    """Handles loading LLaVA and preprocessing images for extraction.

    Typical usage::

        ext = LlavaExtractor(device=torch.device("cuda:0"))
        model, processor = ext.load()
        inputs = ext.preprocess(pil_image)
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.config: ModelConfig = get_model_config("llava")
        self.device = device
        self.dtype = dtype
        self.model: Any | None = None
        self.processor: Any | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> tuple[Any, Any]:
        """Load the LLaVA model and processor from HuggingFace.

        Uses ``LlavaForConditionalGeneration`` and ``AutoProcessor``.

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
            torch_dtype=self.dtype,
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

    def preprocess(
        self,
        image: Image.Image,
        prompt: str = "",
    ) -> dict[str, torch.Tensor]:
        """Convert a single PIL image into model-ready inputs.

        Args:
            image: A PIL Image (RGB).
            prompt: Optional text prompt to accompany the image.  LLaVA
                expects a prompt containing the ``<image>`` placeholder;
                if *prompt* is empty a minimal default is used.

        Returns:
            Dict of tensors suitable for ``model(**inputs)``.
        """
        if self.processor is None:
            raise RuntimeError("Call .load() before .preprocess().")

        if not prompt:
            prompt = "<image>\nDescribe this image."

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def preprocess_batch(
        self,
        images: list[Image.Image],
        prompts: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Convert a batch of PIL images into model-ready inputs.

        Args:
            images: List of PIL Images (RGB).
            prompts: Optional list of text prompts (one per image).
                If *None*, a default prompt is used for each image.

        Returns:
            Batched dict of tensors with padding applied.
        """
        if self.processor is None:
            raise RuntimeError("Call .load() before .preprocess_batch().")

        if prompts is None:
            prompts = ["<image>\nDescribe this image."] * len(images)

        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

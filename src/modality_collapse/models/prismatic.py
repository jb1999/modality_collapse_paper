"""Prismatic VLM model loader and image preprocessor.

Prismatic VLMs (TRI-ML) pair a timm vision backbone (DINOv2, SigLIP, etc.)
with a Vicuna-7B LLM connected via a GELU-MLP projector.

Requires the ``prismatic`` package installed from
``https://github.com/TRI-ML/prismatic-vlms`` and a separate venv
(``~/venvs/prismatic``) due to dependency conflicts with the main project.

Two variants are of primary interest:
  - ``dinov2-224px+7b``: DINOv2 encoder (NO text alignment)
  - ``siglip-224px+7b``: SigLIP encoder (text-aligned, control)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from PIL import Image

from .registry import get_model_config, ModelConfig

logger = logging.getLogger(__name__)


class PrismaticExtractor:
    """Handles loading Prismatic VLMs and preprocessing images.

    Uses the native ``prismatic`` API (``from prismatic import load``).

    Typical usage::

        ext = PrismaticExtractor(device=torch.device("cuda:0"),
                                 variant="dinov2-224px+7b")
        model, processor = ext.load()
        inputs = ext.preprocess(pil_image)
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        variant: str = "dinov2-224px+7b",
    ) -> None:
        # Use the registry config for hook points etc.
        config_name = "prismatic_dinov2" if "dinov2" in variant else "prismatic_siglip"
        self.config: ModelConfig = get_model_config(config_name)
        self.device = device
        self.dtype = dtype
        self.variant = variant
        self.model: Any | None = None
        self.image_transform: Any | None = None
        self.tokenizer: Any | None = None

    def load(self) -> tuple[Any, Any]:
        """Load the Prismatic VLM.

        Returns ``(model, None)`` — no separate processor object,
        preprocessing is handled via ``self.preprocess()``.
        """
        from prismatic import load as prismatic_load

        self.model = prismatic_load(self.variant)
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

        # Cache the image transform and tokenizer for preprocessing.
        self.image_transform = self.model.vision_backbone.get_image_transform()
        self.tokenizer = self.model.llm_backbone.tokenizer

        logger.info(
            "Loaded Prismatic %s on %s (%s).",
            self.variant,
            self.device,
            self.dtype,
        )
        return self.model, None

    def preprocess(
        self,
        image: Image.Image,
        prompt: str = "",
    ) -> dict[str, torch.Tensor]:
        """Convert a single PIL image into model-ready inputs.

        Prismatic's forward pass expects ``input_ids`` and ``pixel_values``.
        """
        if self.model is None:
            raise RuntimeError("Call .load() before .preprocess().")

        if not prompt:
            prompt = "Describe this image."

        # Build prompt using the model's prompt builder (Vicuna chat template).
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()

        # Tokenize text.
        input_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
        ).input_ids.to(self.device)

        # Transform image.
        pixel_values = self.image_transform(image.convert("RGB"))
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        pixel_values = pixel_values.to(self.device, dtype=self.dtype)

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

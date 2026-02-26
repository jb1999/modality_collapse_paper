"""Text baseline loader and tokenizer.

This serves as the text-only control in modality collapse experiments.
There is no adapter or encoder -- inputs are tokenized text fed directly
to the frozen LLM, giving us the "text law" reference distribution
against which non-text modalities are compared.

Supports standalone text models (Llama-3.1-8B) and extracting the LLM
backbone from multimodal models (e.g. Qwen2-Audio).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .registry import get_model_config, ModelConfig

logger = logging.getLogger(__name__)


class TextBaselineExtractor:
    """Handles loading a text-only LLM and tokenizing text for extraction.

    Typical usage::

        ext = TextBaselineExtractor(device=torch.device("cuda:0"))
        model, tokenizer = ext.load()
        inputs = ext.preprocess("The quick brown fox jumps.")

    For extracting the LLM backbone from a multimodal model::

        ext = TextBaselineExtractor(
            device=torch.device("cuda:0"),
            model_name="qwen2audio",
            multimodal=True,
        )
        model, tokenizer = ext.load()
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        model_name: str = "llama",
        multimodal: bool = False,
    ) -> None:
        self.model_name = model_name
        self.multimodal = multimodal
        self.config: ModelConfig = get_model_config(model_name)
        self.device = device
        self.dtype = dtype
        self.model: Any | None = None
        self.processor: Any | None = None  # tokenizer, kept as "processor" for API parity
        self._full_model: Any | None = None  # only used in multimodal mode

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> tuple[Any, Any]:
        """Load the model and tokenizer from HuggingFace.

        In standalone mode, loads a CausalLM directly.  In multimodal
        mode, loads the full multimodal model and extracts the LLM
        backbone and tokenizer.

        Returns:
            A ``(model, tokenizer)`` tuple.  The model is placed on
            ``self.device`` in eval mode with ``self.dtype`` precision.
        """
        if self.multimodal:
            return self._load_multimodal()
        return self._load_standalone()

    def _load_standalone(self) -> tuple[Any, Any]:
        """Load a standalone text-only CausalLM."""
        import transformers

        # -- tokenizer --------------------------------------------------
        self.processor = transformers.AutoTokenizer.from_pretrained(
            self.config.hf_path,
            trust_remote_code=True,
        )
        if self.processor.pad_token is None:
            self.processor.pad_token = self.processor.eos_token

        # -- model ------------------------------------------------------
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
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

    def _load_multimodal(self) -> tuple[Any, Any]:
        """Load a multimodal model and extract the LLM backbone.

        The full model is loaded, then the ``language_model`` submodule
        is extracted for text-only processing.  The tokenizer is obtained
        from the multimodal processor.
        """
        import transformers

        # -- processor (has tokenizer) -----------------------------------
        proc = transformers.AutoProcessor.from_pretrained(
            self.config.hf_path,
            trust_remote_code=True,
        )

        # -- full model --------------------------------------------------
        try:
            model_cls = getattr(transformers, self.config.model_class)
        except AttributeError:
            model_cls = transformers.AutoModel

        self._full_model = model_cls.from_pretrained(
            self.config.hf_path,
            dtype=self.dtype,
            device_map={"": self.device},
            trust_remote_code=True,
        )
        self._full_model.eval()

        # Extract the tokenizer from the multimodal processor.
        if hasattr(proc, "tokenizer"):
            self.processor = proc.tokenizer
        else:
            self.processor = proc

        if self.processor.pad_token is None:
            self.processor.pad_token = self.processor.eos_token

        # The model we expose for hooking is the full model (hooks use
        # paths like language_model.model.layers.16).
        self.model = self._full_model

        logger.info(
            "Loaded %s (multimodal, LLM backbone) on %s (%s).",
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
        text: str,
        max_length: int = 512,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a single text string into model-ready inputs.

        Args:
            text: The input string.
            max_length: Maximum number of tokens; longer inputs are
                truncated from the right.

        Returns:
            Dict of tensors (``input_ids``, ``attention_mask``) suitable
            for ``model(**inputs)``.
        """
        if self.processor is None:
            raise RuntimeError("Call .load() before .preprocess().")

        inputs = self.processor(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def preprocess_batch(
        self,
        texts: list[str],
        max_length: int = 512,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a batch of text strings into model-ready inputs.

        Args:
            texts: List of input strings.
            max_length: Maximum number of tokens per string.

        Returns:
            Batched dict of tensors with padding applied.
        """
        if self.processor is None:
            raise RuntimeError("Call .load() before .preprocess_batch().")

        inputs = self.processor(
            texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

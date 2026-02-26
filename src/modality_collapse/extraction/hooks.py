"""Representation extraction via PyTorch forward hooks."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class RepresentationExtractor:
    """Attach forward hooks to named modules and collect activations.

    Args:
        model: The model to hook into.
        hook_points: Mapping from logical names (e.g. ``"encoder_output"``)
            to dot-separated module paths (e.g. ``"audio_tower.encoder"``).
        pool_strategy: How to reduce the sequence dimension.
            ``"none"``  -- keep all tokens (seq_len, hidden_dim).
            ``"mean"``  -- mean over the sequence dimension.
            ``"last"``  -- take the last token.

    Usage::

        extractor = RepresentationExtractor(model, {"enc": "audio_tower.encoder"})
        with extractor:
            model(inputs)
            acts = extractor.get_activations()
    """

    def __init__(
        self,
        model: nn.Module,
        hook_points: dict[str, str],
        pool_strategy: Literal["none", "mean", "last"] = "none",
    ) -> None:
        self.model = model
        self.hook_points = hook_points
        self.pool_strategy = pool_strategy

        self._activations: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHook] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_module(self, path: str) -> nn.Module:
        """Navigate a dot-separated path to find a submodule.

        Args:
            path: Dot-separated module path (e.g. ``"layer.0.attention"``).

        Returns:
            The target ``nn.Module``.

        Raises:
            AttributeError: If any segment of the path is not found.
        """
        module = self.model
        for attr in path.split("."):
            module = getattr(module, attr)
        return module

    def _hook_fn(self, name: str):
        """Return a hook function that stores the module output.

        The stored tensor is detached and moved to CPU immediately to
        avoid accumulating VRAM across forward passes.
        """

        def hook(
            module: nn.Module,
            input: tuple,
            output,
        ) -> None:
            # Some modules return tuples; take the first element.
            if isinstance(output, tuple):
                output = output[0]

            tensor: torch.Tensor = output.detach().float().cpu()

            # Apply pooling if the tensor is 3-D (batch, seq_len, hidden_dim).
            if tensor.ndim == 3 and self.pool_strategy != "none":
                if self.pool_strategy == "mean":
                    tensor = tensor.mean(dim=1)
                elif self.pool_strategy == "last":
                    tensor = tensor[:, -1, :]

            self._activations[name] = tensor

        return hook

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_hooks(self) -> None:
        """Register forward hooks on all configured hook points."""
        for name, path in self.hook_points.items():
            module = self._get_module(path)
            handle = module.register_forward_hook(self._hook_fn(name))
            self._handles.append(handle)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def get_activations(self) -> dict[str, torch.Tensor]:
        """Return stored activations and clear the internal buffer.

        Returns:
            Mapping from logical hook-point names to tensors (on CPU).
        """
        activations = dict(self._activations)
        self._activations.clear()
        return activations

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "RepresentationExtractor":
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_hooks()

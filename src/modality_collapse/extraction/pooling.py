"""Pooling strategies for variable-length sequences."""

import torch


def mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean pool over the sequence dimension, respecting an attention mask.

    Args:
        hidden_states: Tensor of shape ``(batch, seq_len, hidden_dim)``.
        attention_mask: Optional tensor of shape ``(batch, seq_len)`` with
            1 for real tokens and 0 for padding. If ``None``, all positions
            are treated as valid.

    Returns:
        Tensor of shape ``(batch, hidden_dim)``.
    """
    if attention_mask is None:
        return hidden_states.mean(dim=1)

    # Expand mask to (batch, seq_len, 1) for broadcasting.
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def last_token_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Take the hidden state of the last non-padding token.

    Args:
        hidden_states: Tensor of shape ``(batch, seq_len, hidden_dim)``.
        attention_mask: Optional tensor of shape ``(batch, seq_len)`` with
            1 for real tokens and 0 for padding. If ``None``, the last
            position in the sequence is used.

    Returns:
        Tensor of shape ``(batch, hidden_dim)``.
    """
    if attention_mask is None:
        return hidden_states[:, -1, :]

    # Index of the last non-padding token per batch element.
    # attention_mask.sum(dim=1) gives the count of real tokens;
    # subtract 1 for 0-based indexing.
    last_indices = attention_mask.sum(dim=1).long() - 1  # (batch,)
    last_indices = last_indices.clamp(min=0)

    batch_size = hidden_states.size(0)
    return hidden_states[torch.arange(batch_size, device=hidden_states.device), last_indices]

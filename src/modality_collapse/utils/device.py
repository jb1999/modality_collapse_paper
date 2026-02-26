"""GPU detection and VRAM monitoring utilities."""

import gc
import logging

import torch

logger = logging.getLogger(__name__)


def get_device(device_str: str = "auto") -> torch.device:
    """Parse a device string and return a torch.device.

    When device_str is "auto", selects the GPU with the most free memory.
    Falls back to CPU if no CUDA device is available.

    Args:
        device_str: One of "auto", "cpu", "cuda", "cuda:0", "cuda:1", etc.

    Returns:
        torch.device for the selected device.
    """
    if device_str == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        return torch.device("cpu")

    if device_str == "auto":
        # Pick the GPU with the most free memory.
        best_idx = 0
        best_free = 0
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_free = free
                best_idx = i
        device = torch.device(f"cuda:{best_idx}")
        logger.info(
            "Auto-selected %s (%.1f GB free).",
            device,
            best_free / (1024**3),
        )
        return device

    # Direct specification like "cuda" or "cuda:1".
    return torch.device(device_str)


def log_vram(device: torch.device, label: str = "") -> None:
    """Print current VRAM usage for a CUDA device.

    Args:
        device: The torch device to query.
        label: Optional label printed alongside the stats.
    """
    if device.type != "cuda":
        return

    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    free, total = torch.cuda.mem_get_info(device)
    free_gb = free / (1024**3)
    total_gb = total / (1024**3)

    prefix = f"[{label}] " if label else ""
    logger.info(
        "%sVRAM on %s: %.2f GB allocated, %.2f GB reserved, "
        "%.2f / %.2f GB free/total",
        prefix,
        device,
        allocated,
        reserved,
        free_gb,
        total_gb,
    )


def clear_gpu(device: torch.device) -> None:
    """Run garbage collection and clear the CUDA cache.

    Args:
        device: The torch device to clear. No-op if not a CUDA device.
    """
    if device.type != "cuda":
        return

    gc.collect()
    torch.cuda.empty_cache()

"""HDF5 save/load utilities for extracted representations."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def create_hdf5(
    path: str,
    hook_names: list[str],
    hidden_dims: dict[str, int],
    max_samples: int,
    metadata: dict | None = None,
) -> None:
    """Create an HDF5 file with pre-allocated datasets for each hook point.

    Each dataset has shape ``(max_samples, hidden_dim)`` and dtype float32.
    Metadata is stored as HDF5 root-level attributes.

    Args:
        path: File path for the new HDF5 file.
        hook_names: List of logical hook-point names.
        hidden_dims: Mapping from hook name to hidden dimension size.
        max_samples: Maximum number of samples the file can hold.
        metadata: Optional dict of metadata to store as HDF5 attributes.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        for name in hook_names:
            dim = hidden_dims[name]
            f.create_dataset(
                name,
                shape=(max_samples, dim),
                dtype="float32",
                chunks=(min(256, max_samples), dim),
            )

        if metadata is not None:
            for key, value in metadata.items():
                # HDF5 attributes don't support arbitrary Python objects;
                # serialise dicts/lists as JSON strings.
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                f.attrs[key] = value


def append_to_hdf5(
    path: str,
    hook_name: str,
    data: np.ndarray,
    current_idx: int,
) -> int:
    """Append a batch of representations to an HDF5 dataset.

    Args:
        path: Path to the HDF5 file.
        hook_name: Name of the dataset to write into.
        data: Array of shape ``(batch_size, hidden_dim)`` to append.
        current_idx: Row index at which to start writing.

    Returns:
        Updated index (``current_idx + batch_size``).
    """
    batch_size = data.shape[0]

    with h5py.File(path, "a") as f:
        ds = f[hook_name]
        end_idx = current_idx + batch_size
        ds[current_idx:end_idx] = data
        f.flush()

    return end_idx


def load_representations(
    path: str,
    hook_name: str,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    """Load representations for a hook point from an HDF5 file.

    Args:
        path: Path to the HDF5 file.
        hook_name: Name of the dataset to read.
        indices: Optional array of row indices to load. If ``None``,
            the entire dataset is returned.

    Returns:
        NumPy array of shape ``(n_samples, hidden_dim)``.
    """
    with h5py.File(path, "r") as f:
        ds = f[hook_name]
        if indices is not None:
            # h5py fancy indexing requires a sorted index array.
            sorted_order = np.argsort(indices)
            sorted_indices = indices[sorted_order]
            data = ds[sorted_indices]
            # Restore original ordering.
            inverse_order = np.empty_like(sorted_order)
            inverse_order[sorted_order] = np.arange(len(sorted_order))
            data = data[inverse_order]
        else:
            data = ds[:]
    return data


def load_metadata(path: str) -> dict:
    """Load metadata attributes from an HDF5 file.

    Values that were stored as JSON strings are deserialised back to
    Python objects.

    Args:
        path: Path to the HDF5 file.

    Returns:
        Dictionary of metadata key-value pairs.
    """
    metadata: dict = {}
    with h5py.File(path, "r") as f:
        for key, value in f.attrs.items():
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    pass
            metadata[key] = value
    return metadata

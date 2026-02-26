#!/usr/bin/env python3
"""Extract non-textual vision labels from COCO annotations and write to HDF5.

Computes three non-textual visual attributes from COCO instance annotations:
  - object_count: number of annotated instances per image (binned: 1/2-4/5-10/11+)
  - avg_obj_size: mean bbox area / image area (binned by tertile: small/medium/large)
  - spatial_spread: std dev of bbox centers from image center (binned by tertile)

These are "non-textual" because captions typically don't describe counts, scale,
or spatial layout — they name objects and actions.

Usage:
    python scripts/10_extract_vision_labels.py
"""

from __future__ import annotations

import modality_collapse.utils.env  # noqa: F401

import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ANNOTATIONS_PATH = Path("data/raw/COCO/annotations/instances_val2017.json")
VISION_H5_FILES = [
    Path("data/representations/llava_coco.h5"),
    Path("data/representations/prismatic_dinov2_coco.h5"),
    Path("data/representations/prismatic_siglip_coco.h5"),
]


def compute_vision_labels(annotations_path: Path) -> dict[int, dict[str, str]]:
    """Compute non-textual labels for each image from COCO annotations.

    Returns:
        Dict mapping image_id -> {object_count, avg_obj_size, spatial_spread}.
    """
    print(f"Loading annotations from {annotations_path} ...")
    with open(annotations_path) as f:
        coco = json.load(f)

    # Build image dimension lookup
    image_dims: dict[int, tuple[int, int]] = {}  # image_id -> (width, height)
    for img in coco["images"]:
        image_dims[img["id"]] = (img["width"], img["height"])

    # Collect per-image annotation data
    per_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        per_image[ann["image_id"]].append(ann)

    # Compute raw values for all images that have at least 1 annotation
    raw_counts: dict[int, int] = {}
    raw_sizes: dict[int, float] = {}
    raw_spreads: dict[int, float] = {}

    for img_id, anns in per_image.items():
        if img_id not in image_dims:
            continue
        w, h = image_dims[img_id]
        img_area = w * h
        if img_area == 0:
            continue

        # Object count
        raw_counts[img_id] = len(anns)

        # Average object size (bbox area / image area)
        bbox_areas = []
        centers_x = []
        centers_y = []
        for ann in anns:
            bx, by, bw, bh = ann["bbox"]  # [x, y, width, height]
            bbox_areas.append(bw * bh / img_area)
            centers_x.append((bx + bw / 2) / w)
            centers_y.append((by + bh / 2) / h)

        raw_sizes[img_id] = float(np.mean(bbox_areas))

        # Spatial spread: std dev of distances from center
        if len(anns) >= 2:
            cx = np.array(centers_x) - 0.5
            cy = np.array(centers_y) - 0.5
            dists = np.sqrt(cx**2 + cy**2)
            raw_spreads[img_id] = float(np.std(dists))
        else:
            raw_spreads[img_id] = 0.0

    # Also handle images with 0 annotations
    for img_id in image_dims:
        if img_id not in per_image or len(per_image[img_id]) == 0:
            raw_counts[img_id] = 0
            raw_sizes[img_id] = 0.0
            raw_spreads[img_id] = 0.0

    print(f"  Computed labels for {len(raw_counts)} images "
          f"({len(per_image)} with annotations)")

    # --- Bin object_count ---
    def bin_count(n: int) -> str:
        if n <= 1:
            return "1"
        elif n <= 4:
            return "2-4"
        elif n <= 10:
            return "5-10"
        else:
            return "11+"

    # --- Bin avg_obj_size and spatial_spread by tertile ---
    # Compute tertile thresholds only on images with annotations
    annotated_ids = sorted(per_image.keys())

    size_vals = np.array([raw_sizes[i] for i in annotated_ids if i in raw_sizes])
    size_t1, size_t2 = np.percentile(size_vals, [33.3, 66.7])

    spread_vals = np.array([raw_spreads[i] for i in annotated_ids if i in raw_spreads])
    spread_t1, spread_t2 = np.percentile(spread_vals, [33.3, 66.7])

    print(f"  Size tertiles: {size_t1:.4f}, {size_t2:.4f}")
    print(f"  Spread tertiles: {spread_t1:.4f}, {spread_t2:.4f}")

    def bin_size(v: float) -> str:
        if v <= size_t1:
            return "small"
        elif v <= size_t2:
            return "medium"
        else:
            return "large"

    def bin_spread(v: float) -> str:
        if v <= spread_t1:
            return "concentrated"
        elif v <= spread_t2:
            return "moderate"
        else:
            return "dispersed"

    # Build final label dict
    labels: dict[int, dict[str, str]] = {}
    for img_id in raw_counts:
        labels[img_id] = {
            "object_count": bin_count(raw_counts[img_id]),
            "avg_obj_size": bin_size(raw_sizes[img_id]),
            "spatial_spread": bin_spread(raw_spreads[img_id]),
        }

    # Print distribution
    for key in ["object_count", "avg_obj_size", "spatial_spread"]:
        from collections import Counter
        dist = Counter(v[key] for v in labels.values())
        print(f"  {key} distribution: {dict(sorted(dist.items()))}")

    return labels


def write_labels_to_h5(
    h5_path: Path,
    labels: dict[int, dict[str, str]],
) -> None:
    """Write non-textual vision labels to an existing HDF5 file.

    Matches sample_ids in the HDF5 to image_ids in the labels dict.
    """
    print(f"\nWriting labels to {h5_path} ...")

    with h5py.File(h5_path, "a") as f:
        # Read sample_ids
        raw_ids = f["sample_ids"][:]
        sample_ids = [
            int(x.decode("utf-8") if isinstance(x, bytes) else str(x))
            for x in raw_ids
        ]
        n = len(sample_ids)

        # Match and build label arrays
        matched = 0
        unmatched = 0
        label_arrays: dict[str, list[str]] = {
            "object_count": [],
            "avg_obj_size": [],
            "spatial_spread": [],
        }

        for sid in sample_ids:
            if sid in labels:
                matched += 1
                for key in label_arrays:
                    label_arrays[key].append(labels[sid][key])
            else:
                unmatched += 1
                for key in label_arrays:
                    label_arrays[key].append("")

        print(f"  Matched: {matched}/{n}, Unmatched: {unmatched}")

        # Write datasets
        dt = h5py.special_dtype(vlen=str)
        for key, values in label_arrays.items():
            if key in f:
                del f[key]
            arr = np.array(values, dtype=object)
            f.create_dataset(key, data=arr, dtype=dt)
            n_nonempty = sum(1 for v in values if v)
            print(f"  Wrote {key}: {n_nonempty} non-empty labels")


def main() -> None:
    # Compute labels from annotations
    labels = compute_vision_labels(ANNOTATIONS_PATH)

    # Write to each vision HDF5 file
    for h5_path in VISION_H5_FILES:
        if not h5_path.exists():
            print(f"  SKIP: {h5_path} not found")
            continue
        write_labels_to_h5(h5_path, labels)

    print("\nDone.")


if __name__ == "__main__":
    main()

"""Utilities for converting PanNuke samples into training-ready tensors."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_NAMES = (
    "neoplastic",
    "inflammatory",
    "connective",
    "dead",
    "epithelial",
)

BACKGROUND_LABEL = 0
IGNORE_LABEL = 255


def prepare_pannuke_sample(
    sample: Mapping[str, Any],
    ignore_label: int = IGNORE_LABEL,
) -> dict[str, Any]:
    """Convert one PanNuke sample into PyTorch tensors.

    PanNuke category numbers 0-4 are shifted to 1-5 so that zero can
    represent the background. Pixels shared by multiple nuclei receive
    the ignore label because their ownership is ambiguous.
    """
    image = np.array(sample["image"], dtype=np.uint8, copy=True)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image, received shape {image.shape}.")

    height, width, _ = image.shape
    raw_instances = sample["instances"]
    categories = np.asarray(sample["categories"], dtype=np.int64)

    if len(raw_instances) != len(categories):
        raise ValueError("The number of instance masks must equal the number of categories.")

    if len(raw_instances) == 0:
        instance_masks = np.zeros((0, height, width), dtype=bool)
    else:
        instance_masks = np.stack([np.asarray(mask, dtype=bool) for mask in raw_instances])

    if instance_masks.shape[1:] != (height, width):
        raise ValueError("Instance-mask dimensions do not match the image dimensions.")

    if categories.size and (categories.min() < 0 or categories.max() >= len(CLASS_NAMES)):
        raise ValueError("PanNuke category labels must be between 0 and 4.")

    occupancy = instance_masks.sum(axis=0)
    overlap_mask = occupancy > 1

    semantic_mask = np.full(
        (height, width),
        BACKGROUND_LABEL,
        dtype=np.int64,
    )
    instance_map = np.zeros((height, width), dtype=np.int64)

    for instance_number, (mask, category) in enumerate(
        zip(instance_masks, categories, strict=True),
        start=1,
    ):
        unambiguous_pixels = mask & ~overlap_mask
        semantic_mask[unambiguous_pixels] = int(category) + 1
        instance_map[unambiguous_pixels] = instance_number

    semantic_mask[overlap_mask] = ignore_label

    image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().to(torch.float32) / 255.0

    return {
        "image": image_tensor,
        "semantic_mask": torch.from_numpy(semantic_mask),
        "instance_map": torch.from_numpy(instance_map),
        "instance_masks": torch.from_numpy(instance_masks),
        "categories": torch.from_numpy(categories.copy()),
        "overlap_mask": torch.from_numpy(overlap_mask),
        "tissue": str(sample["tissue"]),
    }


class PanNukeDataset(Dataset):
    """PyTorch wrapper around a loaded PanNuke split."""

    def __init__(self, records: Any) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw_sample = self.records[index]
        prepared_sample = prepare_pannuke_sample(raw_sample)

        tissue_feature = self.records.features.get("tissue")
        tissue_value = raw_sample["tissue"]

        if isinstance(tissue_value, (int, np.integer)) and hasattr(tissue_feature, "int2str"):
            prepared_sample["tissue"] = tissue_feature.int2str(int(tissue_value))

        return prepared_sample

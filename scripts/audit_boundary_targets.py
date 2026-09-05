"""Audit Fold 1 boundary-aware training-target frequencies."""

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

from pancancer_nuclei.data.pannuke import PanNukeDataset
from pancancer_nuclei.data.targets import (
    IGNORE_LABEL,
    create_boundary_target,
)
from pancancer_nuclei.data.transforms import create_validation_transforms

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
CACHE_DIR = "data/raw/huggingface_cache"

OUTPUT_PATH = Path("reports/fold1_boundary_target_audit.json")

CLASS_NAMES = (
    "background",
    "interior",
    "boundary",
)

BOUNDARY_WIDTH = 2
WEIGHT_OFFSET = 1.02


def calculate_class_weights(
    frequencies: np.ndarray,
) -> np.ndarray:
    """Calculate normalized logarithmic class weights."""
    weights = 1.0 / np.log(WEIGHT_OFFSET + frequencies)
    return weights / weights.mean()


def main() -> None:
    """Audit every Fold 1 boundary-aware target."""
    records = load_dataset(
        DATASET_NAME,
        revision=REVISION,
        split="fold1",
        cache_dir=CACHE_DIR,
    )
    dataset = PanNukeDataset(
        records,
        transform=create_validation_transforms(),
    )

    pixel_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    samples_containing_class = np.zeros(
        len(CLASS_NAMES),
        dtype=np.int64,
    )
    ignored_pixels = 0
    total_instances = 0

    print("Auditing Fold 1 boundary targets...")
    print("Samples:", len(dataset))
    print("Boundary width:", BOUNDARY_WIDTH)
    print("------------------------")

    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        instance_masks = sample["instance_masks"].numpy()
        total_instances += len(instance_masks)

        target = create_boundary_target(
            instance_masks,
            boundary_width=BOUNDARY_WIDTH,
        )

        valid_pixels = target != IGNORE_LABEL
        pixel_counts += np.bincount(
            target[valid_pixels],
            minlength=len(CLASS_NAMES),
        )
        ignored_pixels += int((~valid_pixels).sum())

        for class_index in range(len(CLASS_NAMES)):
            if np.any(target == class_index):
                samples_containing_class[class_index] += 1

        completed = sample_index + 1
        if completed % 250 == 0 or completed == len(dataset):
            print(f"Processed {completed}/{len(dataset)}")

    total_valid_pixels = int(pixel_counts.sum())
    frequencies = pixel_counts / total_valid_pixels
    class_weights = calculate_class_weights(frequencies)

    report = {
        "dataset": DATASET_NAME,
        "revision": REVISION,
        "split": "fold1",
        "number_of_samples": len(dataset),
        "number_of_instances": total_instances,
        "boundary_width": BOUNDARY_WIDTH,
        "ignored_overlap_pixels": ignored_pixels,
        "pixel_counts": {
            class_name: int(pixel_counts[class_index])
            for class_index, class_name in enumerate(CLASS_NAMES)
        },
        "pixel_frequencies": {
            class_name: float(frequencies[class_index])
            for class_index, class_name in enumerate(CLASS_NAMES)
        },
        "samples_containing_class": {
            class_name: int(samples_containing_class[class_index])
            for class_index, class_name in enumerate(CLASS_NAMES)
        },
        "class_weights": {
            class_name: float(class_weights[class_index])
            for class_index, class_name in enumerate(CLASS_NAMES)
        },
        "weighting_method": ("Inverse log frequency with offset 1.02, normalized to mean 1."),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("------------------------")

    for class_index, class_name in enumerate(CLASS_NAMES):
        print(
            f"{class_name}: "
            f"{pixel_counts[class_index]:,} pixels "
            f"({100 * frequencies[class_index]:.3f}%), "
            f"weight={class_weights[class_index]:.4f}"
        )

    print("Ignored overlap pixels:", ignored_pixels)
    print("Total nuclei:", total_instances)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

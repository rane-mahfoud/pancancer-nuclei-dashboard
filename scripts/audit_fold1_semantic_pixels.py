"""Audit semantic pixel frequencies in the Fold 1 training split."""

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

from pancancer_nuclei.data.pannuke import PanNukeDataset

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
CACHE_DIR = "data/raw/huggingface_cache"
OUTPUT_PATH = Path("reports/fold1_semantic_pixel_audit.json")

CLASS_NAMES = (
    "background",
    "neoplastic",
    "inflammatory",
    "connective",
    "dead",
    "epithelial",
)


def main() -> None:
    """Count semantic pixels and class presence in Fold 1."""
    records = load_dataset(
        DATASET_NAME,
        revision=REVISION,
        split="fold1",
        cache_dir=CACHE_DIR,
    )
    dataset = PanNukeDataset(records)

    pixel_counts = np.zeros(6, dtype=np.int64)
    samples_containing_class = np.zeros(6, dtype=np.int64)
    ignored_pixels = 0

    for index in range(len(dataset)):
        target = dataset[index]["semantic_mask"].numpy()
        valid_pixels = target != 255
        valid_target = target[valid_pixels]

        pixel_counts += np.bincount(
            valid_target,
            minlength=6,
        )
        ignored_pixels += int((~valid_pixels).sum())

        present_classes = np.unique(valid_target)
        samples_containing_class[present_classes] += 1

        if (index + 1) % 250 == 0:
            print(f"Checked {index + 1}/{len(dataset)} samples")

    total_valid_pixels = int(pixel_counts.sum())
    frequencies = pixel_counts / total_valid_pixels

    report = {
        "dataset": DATASET_NAME,
        "revision": REVISION,
        "split": "fold1",
        "samples": len(dataset),
        "total_valid_pixels": total_valid_pixels,
        "ignored_overlap_pixels": ignored_pixels,
        "pixel_counts": {
            name: int(count)
            for name, count in zip(
                CLASS_NAMES,
                pixel_counts,
                strict=True,
            )
        },
        "pixel_frequencies": {
            name: float(frequency)
            for name, frequency in zip(
                CLASS_NAMES,
                frequencies,
                strict=True,
            )
        },
        "samples_containing_class": {
            name: int(count)
            for name, count in zip(
                CLASS_NAMES,
                samples_containing_class,
                strict=True,
            )
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("------------------------")
    print("Fold 1 samples:", len(dataset))
    print("Ignored overlap pixels:", ignored_pixels)

    for name in CLASS_NAMES:
        count = report["pixel_counts"][name]
        percentage = 100.0 * report["pixel_frequencies"][name]
        sample_count = report["samples_containing_class"][name]

        print(f"{name}: {count:,} pixels ({percentage:.3f}%), present in {sample_count:,} samples")

    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

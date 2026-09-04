import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
CACHE_DIR = Path("data/raw/huggingface_cache")


def count_components(mask):
    rows, columns = np.nonzero(mask)

    if len(rows) == 0:
        return 0

    cropped = mask[
        rows.min() : rows.max() + 1,
        columns.min() : columns.max() + 1,
    ]

    component_count, _ = cv2.connectedComponents(
        cropped.astype(np.uint8),
        connectivity=8,
    )

    return component_count - 1


def add_example(examples, name, sample_id, limit=20):
    if len(examples[name]) < limit:
        examples[name].append(sample_id)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Maximum samples checked per fold; omit for the complete audit.",
)
arguments = parser.parse_args()

dataset = load_dataset(
    DATASET_NAME,
    revision=REVISION,
    cache_dir=str(CACHE_DIR),
)

summary = Counter()
image_dtype_counts = Counter()
mask_dtype_counts = Counter()
image_hashes = defaultdict(list)
instance_areas = []
nuclei_per_patch = []

examples = {
    "wrong_image_shape": [],
    "label_count_mismatch": [],
    "empty_instance": [],
    "disconnected_instance": [],
    "overlapping_instances": [],
}

for split_name in ("fold1", "fold2", "fold3"):
    split = dataset[split_name]
    sample_count = len(split)

    if arguments.limit is not None:
        sample_count = min(arguments.limit, sample_count)

    for index in tqdm(
        range(sample_count),
        desc=f"Geometry audit: {split_name}",
    ):
        sample = split[index]
        sample_id = f"{split_name}:{index}"

        image = np.asarray(sample["image"])
        instances = sample["instances"]
        categories = sample["categories"]

        summary["samples_checked"] += 1
        nuclei_per_patch.append(len(instances))
        image_dtype_counts[str(image.dtype)] += 1

        if image.shape != (256, 256, 3):
            summary["wrong_image_shapes"] += 1
            add_example(examples, "wrong_image_shape", sample_id)

        image_hash = hashlib.sha256(image.tobytes()).hexdigest()
        image_hashes[image_hash].append(sample_id)

        if len(instances) != len(categories):
            summary["label_count_mismatches"] += 1
            add_example(examples, "label_count_mismatch", sample_id)

        if not instances:
            summary["empty_samples"] += 1

        occupancy = np.zeros((256, 256), dtype=np.uint16)

        for instance_number, instance in enumerate(instances):
            mask = np.asarray(instance)
            mask_dtype_counts[str(mask.dtype)] += 1
            summary["instances_checked"] += 1

            if mask.shape != (256, 256):
                summary["wrong_mask_shapes"] += 1
                continue

            binary_mask = mask.astype(bool)
            area = int(binary_mask.sum())
            instance_areas.append(area)

            if area == 0:
                summary["empty_instances"] += 1
                add_example(
                    examples,
                    "empty_instance",
                    f"{sample_id}:instance{instance_number}",
                )
                continue

            occupancy += binary_mask

            component_count = count_components(binary_mask)
            if component_count != 1:
                summary["disconnected_instances"] += 1
                add_example(
                    examples,
                    "disconnected_instance",
                    f"{sample_id}:instance{instance_number}",
                )

            touches_edge = (
                binary_mask[0].any()
                or binary_mask[-1].any()
                or binary_mask[:, 0].any()
                or binary_mask[:, -1].any()
            )
            if touches_edge:
                summary["edge_touching_instances"] += 1

        overlap_pixels = int((occupancy > 1).sum())
        if overlap_pixels > 0:
            summary["samples_with_overlap"] += 1
            summary["overlap_pixels"] += overlap_pixels
            add_example(examples, "overlapping_instances", sample_id)

duplicate_groups = [locations for locations in image_hashes.values() if len(locations) > 1]

summary["exact_duplicate_groups"] = len(duplicate_groups)
summary["samples_in_duplicate_groups"] = sum(len(group) for group in duplicate_groups)

area_array = np.asarray(instance_areas)
nuclei_array = np.asarray(nuclei_per_patch)

audit = {
    "dataset": DATASET_NAME,
    "revision": REVISION,
    "limit_per_fold": arguments.limit,
    "summary": dict(summary),
    "image_dtype_counts": dict(image_dtype_counts),
    "mask_dtype_counts": dict(mask_dtype_counts),
    "instance_area_pixels": {
        "minimum": int(area_array.min()),
        "median": float(np.median(area_array)),
        "mean": float(area_array.mean()),
        "maximum": int(area_array.max()),
    },
    "nuclei_per_patch": {
        "minimum": int(nuclei_array.min()),
        "median": float(np.median(nuclei_array)),
        "mean": float(nuclei_array.mean()),
        "maximum": int(nuclei_array.max()),
    },
    "duplicate_groups": duplicate_groups,
    "examples": examples,
}

output_name = (
    "data_audit_geometry.json" if arguments.limit is None else "data_audit_geometry_smoke.json"
)
output_path = Path("reports") / output_name

output_path.write_text(
    json.dumps(audit, indent=2),
    encoding="utf-8",
)

print("\nGeometry audit completed")
print("------------------------")
print("Samples checked:", summary["samples_checked"])
print("Instances checked:", summary["instances_checked"])
print("Empty samples:", summary["empty_samples"])
print("Empty instances:", summary["empty_instances"])
print("Disconnected instances:", summary["disconnected_instances"])
print("Samples with overlap:", summary["samples_with_overlap"])
print("Exact duplicate groups:", summary["exact_duplicate_groups"])
print("Edge-touching instances:", summary["edge_touching_instances"])
print("Saved to:", output_path)

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from skimage.segmentation import find_boundaries

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
CACHE_DIR = Path("data/raw/huggingface_cache")
OUTPUT_PATH = Path("reports/figures/pannuke_annotation_anomalies.png")

DISCONNECTED_EXAMPLES = [
    ("fold1", 54, 2),
    ("fold1", 66, 29),
    ("fold1", 282, 16),
]

OVERLAP_EXAMPLES = [
    ("fold1", 81),
    ("fold1", 86),
    ("fold1", 119),
]

BACKGROUND = "#fff7fb"
TEXT = "#4c3444"
HIGHLIGHT = np.array([231, 168, 195])
BOUNDARY = np.array([143, 71, 106])
OVERLAP = np.array([196, 63, 131])


def crop_around_mask(image, mask, padding=14):
    rows, columns = np.nonzero(mask)

    row_min = int(rows.min())
    row_max = int(rows.max())
    column_min = int(columns.min())
    column_max = int(columns.max())

    height = row_max - row_min + 1
    width = column_max - column_min + 1

    side_length = min(
        max(image.shape[:2]),
        max(height, width) + 2 * padding,
    )

    row_center = (row_min + row_max) // 2
    column_center = (column_min + column_max) // 2

    row_start = row_center - side_length // 2
    column_start = column_center - side_length // 2

    row_start = max(
        0,
        min(row_start, image.shape[0] - side_length),
    )
    column_start = max(
        0,
        min(column_start, image.shape[1] - side_length),
    )

    return image[
        row_start : row_start + side_length,
        column_start : column_start + side_length,
    ]


def add_mask_overlay(image, mask):
    overlay = image.copy()
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * HIGHLIGHT).astype(np.uint8)

    boundary = find_boundaries(mask, mode="inner")
    overlay[boundary] = BOUNDARY
    return overlay


dataset = load_dataset(
    DATASET_NAME,
    revision=REVISION,
    cache_dir=str(CACHE_DIR),
)

figure, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8),
    facecolor=BACKGROUND,
)

for column, (split_name, sample_index, instance_index) in enumerate(DISCONNECTED_EXAMPLES):
    sample = dataset[split_name][sample_index]
    image = np.asarray(sample["image"])
    mask = np.asarray(sample["instances"][instance_index], dtype=bool)

    component_count, _ = cv2.connectedComponents(
        mask.astype(np.uint8),
        connectivity=8,
    )
    component_count -= 1

    overlay = add_mask_overlay(image, mask)
    cropped = crop_around_mask(overlay, mask)

    axes[0, column].imshow(cropped)
    axes[0, column].set_title(
        f"{split_name}:{sample_index}, instance {instance_index}\n"
        f"{component_count} disconnected components",
        color=TEXT,
    )
    axes[0, column].axis("off")

for column, (split_name, sample_index) in enumerate(OVERLAP_EXAMPLES):
    sample = dataset[split_name][sample_index]
    image = np.asarray(sample["image"])
    instances = sample["instances"]

    occupancy = np.zeros((256, 256), dtype=np.uint16)

    for instance in instances:
        occupancy += np.asarray(instance, dtype=bool)

    overlap_mask = occupancy > 1
    visible_marker = cv2.dilate(
        overlap_mask.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)

    overlay = image.copy()
    overlay[visible_marker] = OVERLAP
    cropped = crop_around_mask(overlay, visible_marker)

    axes[1, column].imshow(cropped)
    axes[1, column].set_title(
        f"{split_name}:{sample_index}\n{int(overlap_mask.sum())} true overlap pixels",
        color=TEXT,
    )
    axes[1, column].axis("off")

figure.suptitle(
    "PanNuke annotation-geometry inspection",
    color=TEXT,
    fontsize=16,
)
figure.text(
    0.5,
    0.02,
    "Top: disconnected instance masks. Bottom: overlap locations; "
    "pink markers enlarged to 3×3 pixels for visibility.",
    ha="center",
    color=TEXT,
)
figure.tight_layout(rect=(0, 0.05, 1, 0.95))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
    facecolor=BACKGROUND,
)
plt.close(figure)

print("Saved:", OUTPUT_PATH)

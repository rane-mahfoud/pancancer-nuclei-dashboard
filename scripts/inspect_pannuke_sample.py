from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from matplotlib.patches import Patch
from skimage.segmentation import find_boundaries

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"

CLASS_NAMES = {
    0: "Neoplastic",
    1: "Inflammatory",
    2: "Connective",
    3: "Dead",
    4: "Epithelial",
}

CLASS_COLORS = {
    0: np.array([230, 25, 75]),
    1: np.array([60, 180, 75]),
    2: np.array([0, 130, 200]),
    3: np.array([245, 130, 48]),
    4: np.array([145, 30, 180]),
}

print("Connecting to the PanNuke mirror...")

dataset = load_dataset(
    DATASET_NAME,
    split="fold1",
    revision=REVISION,
    streaming=True,
)

print("Loading one sample...")
sample = next(iter(dataset))

image = np.asarray(sample["image"])
instances = sample["instances"]
categories = sample["categories"]
tissue_value = sample["tissue"]

if isinstance(tissue_value, int):
    tissue_name = dataset.features["tissue"].int2str(tissue_value)
else:
    tissue_name = str(tissue_value)

assert image.shape == (256, 256, 3)
assert len(instances) == len(categories)

overlay = image.copy()

for instance, category in zip(instances, categories, strict=True):
    mask = np.asarray(instance, dtype=bool)
    color = CLASS_COLORS[category]

    overlay[mask] = (0.65 * overlay[mask] + 0.35 * color).astype(np.uint8)

    boundary = find_boundaries(mask, mode="inner")
    overlay[boundary] = color

legend_items = [
    Patch(
        facecolor=CLASS_COLORS[class_id] / 255,
        label=class_name,
    )
    for class_id, class_name in CLASS_NAMES.items()
]

figure, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image)
axes[0].set_title(f"Original image — {tissue_name}")
axes[0].axis("off")

axes[1].imshow(overlay)
axes[1].set_title(f"Ground-truth nuclei — {len(instances)} instances")
axes[1].axis("off")
axes[1].legend(
    handles=legend_items,
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
)

output_path = Path("reports/figures/pannuke_fold1_sample_overlay.png")
output_path.parent.mkdir(parents=True, exist_ok=True)

figure.tight_layout()
figure.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close(figure)

print(f"Saved verified overlay to: {output_path}")
print("Basic checks passed.")

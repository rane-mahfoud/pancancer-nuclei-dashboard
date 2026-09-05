"""Visualize boundary-aware targets on real PanNuke samples."""

from pathlib import Path

import numpy as np
from datasets import load_dataset
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from matplotlib.patches import Patch

from pancancer_nuclei.data.pannuke import PanNukeDataset
from pancancer_nuclei.data.targets import (
    BOUNDARY_LABEL,
    IGNORE_LABEL,
    INTERIOR_LABEL,
    create_boundary_target,
)
from pancancer_nuclei.data.transforms import create_validation_transforms

plt.switch_backend("Agg")

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
CACHE_DIR = "data/raw/huggingface_cache"

SAMPLE_INDICES = (0, 81, 119)
OUTPUT_PATH = Path("reports/figures/pannuke_boundary_targets.png")

BACKGROUND_COLOR = "#fff7fb"
INTERIOR_COLOR = "#d9c5e8"
BOUNDARY_COLOR = "#d34f83"
IGNORE_COLOR = "#4c3444"
TEXT_COLOR = "#4c3444"

TARGET_COLORS = (
    BACKGROUND_COLOR,
    INTERIOR_COLOR,
    BOUNDARY_COLOR,
    IGNORE_COLOR,
)

TARGET_NAMES = (
    "Background",
    "Nucleus interior",
    "Nucleus boundary",
    "Ignored overlap",
)


def create_overlay(
    image: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Overlay interior, boundary, and ignored pixels on an image."""
    overlay = np.ones(
        (target.shape[0], target.shape[1], 4),
        dtype=np.float32,
    )
    overlay[..., 3] = 0.0

    interior_pixels = target == INTERIOR_LABEL
    boundary_pixels = target == BOUNDARY_LABEL
    ignored_pixels = target == IGNORE_LABEL

    overlay[interior_pixels] = to_rgba(
        INTERIOR_COLOR,
        alpha=0.28,
    )
    overlay[boundary_pixels] = to_rgba(
        BOUNDARY_COLOR,
        alpha=0.95,
    )
    overlay[ignored_pixels] = to_rgba(
        IGNORE_COLOR,
        alpha=1.0,
    )

    return overlay


def main() -> None:
    """Generate original, overlay, and discrete target panels."""
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

    color_map = ListedColormap(TARGET_COLORS)
    color_norm = BoundaryNorm(
        np.arange(-0.5, 4.5, 1.0),
        color_map.N,
    )

    figure, axes = plt.subplots(
        len(SAMPLE_INDICES),
        3,
        figsize=(12, 12),
    )
    figure.patch.set_facecolor(BACKGROUND_COLOR)

    print("Boundary-target inspection:")

    for row, sample_index in enumerate(SAMPLE_INDICES):
        sample = dataset[sample_index]
        image = np.clip(
            sample["image"].permute(1, 2, 0).numpy(),
            0.0,
            1.0,
        )
        instance_masks = sample["instance_masks"].numpy()

        target = create_boundary_target(
            instance_masks,
            boundary_width=2,
        )
        overlay = create_overlay(image, target)

        display_target = target.copy()
        display_target[display_target == IGNORE_LABEL] = 3

        number_of_instances = len(instance_masks)
        interior_pixels = int((target == INTERIOR_LABEL).sum())
        boundary_pixels = int((target == BOUNDARY_LABEL).sum())
        ignored_pixels = int((target == IGNORE_LABEL).sum())

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(
            f"Original | {sample['tissue']}\n{number_of_instances} nuclei",
            color=TEXT_COLOR,
        )

        axes[row, 1].imshow(image)
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(
            "Boundary overlay",
            color=TEXT_COLOR,
        )

        axes[row, 2].imshow(
            display_target,
            cmap=color_map,
            norm=color_norm,
            interpolation="nearest",
        )
        axes[row, 2].set_title(
            f"Training target\ninterior={interior_pixels:,}, boundary={boundary_pixels:,}",
            color=TEXT_COLOR,
        )

        for axis in axes[row]:
            axis.axis("off")

        print(
            f"  fold1:{sample_index} | "
            f"tissue={sample['tissue']} | "
            f"nuclei={number_of_instances} | "
            f"interior={interior_pixels} | "
            f"boundary={boundary_pixels} | "
            f"ignored={ignored_pixels}"
        )

    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor="#d8c5cf",
            label=name,
        )
        for color, name in zip(
            TARGET_COLORS,
            TARGET_NAMES,
            strict=True,
        )
    ]

    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )
    figure.suptitle(
        "PanNuke boundary-aware training targets",
        color=TEXT_COLOR,
        fontsize=16,
    )
    figure.text(
        0.5,
        0.008,
        "Two-pixel boundaries derived from individual ground-truth nucleus masks.",
        ha="center",
        color=TEXT_COLOR,
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.96))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

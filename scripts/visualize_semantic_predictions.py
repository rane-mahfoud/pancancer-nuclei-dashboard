"""Visualize semantic U-Net predictions on Fold 2 samples."""

from pathlib import Path

import matplotlib
import numpy as np
import torch
from datasets import load_dataset

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from pancancer_nuclei.data.pannuke import PanNukeDataset
from pancancer_nuclei.data.transforms import (
    create_validation_transforms,
)
from pancancer_nuclei.models.unet import UNet

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
CACHE_DIR = "data/raw/huggingface_cache"

CHECKPOINT_PATH = Path("models/checkpoints/unet_smoke_best.pt")
OUTPUT_PATH = Path("reports/figures/unet_smoke_predictions.png")
SAMPLE_INDICES = (0, 1, 2)

CLASS_NAMES = (
    "background",
    "neoplastic",
    "inflammatory",
    "connective",
    "dead",
    "epithelial",
)
CLASS_COLORS = (
    "#fff7fb",
    "#d98ba6",
    "#a8c7a5",
    "#8fb8d8",
    "#e8b38e",
    "#b7a0d8",
)


def main() -> None:
    """Generate original, ground-truth, and prediction panels."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    device = torch.device("cuda")
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )
    state_dict = checkpoint["model_state_dict"]

    base_channels = state_dict["encoder_1.layers.0.weight"].shape[0]

    model = UNet(
        input_channels=3,
        number_of_classes=6,
        base_channels=base_channels,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    records = load_dataset(
        DATASET_NAME,
        revision=REVISION,
        split="fold2",
        cache_dir=CACHE_DIR,
    )
    dataset = PanNukeDataset(
        records,
        transform=create_validation_transforms(),
    )

    color_map = ListedColormap(CLASS_COLORS)
    color_norm = BoundaryNorm(
        np.arange(-0.5, 6.5, 1.0),
        color_map.N,
    )

    figure, axes = plt.subplots(
        len(SAMPLE_INDICES),
        3,
        figsize=(11, 10),
    )
    figure.patch.set_facecolor("#fff7fb")

    target_counts = np.zeros(6, dtype=np.int64)
    prediction_counts = np.zeros(6, dtype=np.int64)

    with torch.no_grad():
        for row, sample_index in enumerate(SAMPLE_INDICES):
            sample = dataset[sample_index]
            image = sample["image"]
            target = sample["semantic_mask"]

            logits = model(image.unsqueeze(0).to(device))
            prediction = logits.argmax(dim=1).squeeze(0).cpu()

            valid_pixels = target != 255
            target_counts += np.bincount(
                target[valid_pixels].numpy(),
                minlength=6,
            )
            prediction_counts += np.bincount(
                prediction[valid_pixels].numpy(),
                minlength=6,
            )

            display_image = image.permute(1, 2, 0).numpy()
            display_target = target.clone()
            display_target[display_target == 255] = 0

            axes[row, 0].imshow(display_image)
            axes[row, 0].set_title(f"Original — {sample['tissue']}")

            axes[row, 1].imshow(
                display_target,
                cmap=color_map,
                norm=color_norm,
                interpolation="nearest",
            )
            axes[row, 1].set_title("Ground truth")

            axes[row, 2].imshow(
                prediction,
                cmap=color_map,
                norm=color_norm,
                interpolation="nearest",
            )
            axes[row, 2].set_title("Smoke-model prediction")

            for axis in axes[row]:
                axis.axis("off")

    legend_handles = [
        Patch(color=color, label=name.title())
        for name, color in zip(
            CLASS_NAMES,
            CLASS_COLORS,
            strict=True,
        )
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    figure.suptitle("PanNuke Fold 2 semantic-segmentation smoke check")
    figure.tight_layout(rect=(0, 0.07, 1, 0.96))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    print("Target pixel counts:")
    print(
        dict(
            zip(
                CLASS_NAMES,
                target_counts.tolist(),
                strict=True,
            )
        )
    )
    print("Predicted pixel counts:")
    print(
        dict(
            zip(
                CLASS_NAMES,
                prediction_counts.tolist(),
                strict=True,
            )
        )
    )
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

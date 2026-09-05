"""Visualize semantic U-Net predictions on diverse Fold 2 samples."""

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from pancancer_nuclei.data.pannuke import PanNukeDataset
from pancancer_nuclei.data.transforms import create_validation_transforms
from pancancer_nuclei.models.unet import UNet

plt.switch_backend("Agg")

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
CACHE_DIR = "data/raw/huggingface_cache"

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

BACKGROUND_COLOR = "#fff7fb"
TEXT_COLOR = "#4c3444"
BORDER_COLOR = "#d8c5cf"


def parse_arguments() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(
        description="Visualize semantic U-Net predictions on PanNuke Fold 2."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/checkpoints/semantic_unet_weighted_best.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/semantic_unet_weighted_predictions.png"),
    )
    parser.add_argument(
        "--number-of-samples",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def select_diverse_samples(
    dataset: PanNukeDataset,
    number_of_samples: int,
    seed: int,
) -> list[tuple[int, dict[str, object]]]:
    """Select reproducible non-empty samples from different tissues."""
    generator = np.random.default_rng(seed)
    candidate_indices = generator.permutation(len(dataset))

    selected_samples: list[tuple[int, dict[str, object]]] = []
    selected_tissues: set[str] = set()

    for candidate_index in candidate_indices:
        sample_index = int(candidate_index)
        sample = dataset[sample_index]
        target = sample["semantic_mask"]
        tissue = str(sample["tissue"])

        contains_foreground = bool(((target > 0) & (target != 255)).any().item())

        if not contains_foreground:
            continue

        if tissue in selected_tissues:
            continue

        selected_samples.append((sample_index, sample))
        selected_tissues.add(tissue)

        if len(selected_samples) == number_of_samples:
            break

    if len(selected_samples) < number_of_samples:
        raise RuntimeError("Could not find enough non-empty samples from unique tissues.")

    return selected_samples


def calculate_sample_dice(
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> float:
    """Calculate macro foreground Dice for one sample."""
    valid_pixels = target != 255
    dice_values: list[float] = []

    for class_index in range(1, len(CLASS_NAMES)):
        target_class = (target == class_index) & valid_pixels
        prediction_class = (prediction == class_index) & valid_pixels

        denominator = target_class.sum().item() + prediction_class.sum().item()

        if denominator == 0:
            continue

        intersection = (target_class & prediction_class).sum().item()
        dice_values.append((2.0 * intersection) / denominator)

    if not dice_values:
        return 0.0

    return float(np.mean(dice_values))


def main() -> None:
    """Generate original, ground-truth, and prediction panels."""
    arguments = parse_arguments()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    if not arguments.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {arguments.checkpoint}")

    device = torch.device("cuda")

    checkpoint = torch.load(
        arguments.checkpoint,
        map_location=device,
        weights_only=True,
    )
    state_dict = checkpoint["model_state_dict"]
    base_channels = state_dict["encoder_1.layers.0.weight"].shape[0]

    model = UNet(
        input_channels=3,
        number_of_classes=len(CLASS_NAMES),
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

    selected_samples = select_diverse_samples(
        dataset=dataset,
        number_of_samples=arguments.number_of_samples,
        seed=arguments.seed,
    )

    color_map = ListedColormap(CLASS_COLORS)
    color_norm = BoundaryNorm(
        np.arange(-0.5, len(CLASS_NAMES) + 0.5, 1.0),
        color_map.N,
    )

    figure, axes = plt.subplots(
        len(selected_samples),
        3,
        figsize=(11.5, 16),
    )
    figure.patch.set_facecolor(BACKGROUND_COLOR)

    target_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    prediction_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)

    print("Selected validation samples:")

    with torch.inference_mode():
        for row, (sample_index, sample) in enumerate(selected_samples):
            image = sample["image"]
            target = sample["semantic_mask"]
            tissue = str(sample["tissue"])

            logits = model(image.unsqueeze(0).to(device))
            prediction = logits.argmax(dim=1).squeeze(0).cpu()

            sample_dice = calculate_sample_dice(
                target=target,
                prediction=prediction,
            )

            valid_pixels = target != 255
            target_counts += np.bincount(
                target[valid_pixels].numpy(),
                minlength=len(CLASS_NAMES),
            )
            prediction_counts += np.bincount(
                prediction[valid_pixels].numpy(),
                minlength=len(CLASS_NAMES),
            )

            display_image = np.clip(
                image.permute(1, 2, 0).numpy(),
                0.0,
                1.0,
            )
            display_target = target.clone()
            display_target[display_target == 255] = 0

            axes[row, 0].imshow(display_image)
            axes[row, 0].set_title(
                f"Original | {tissue}",
                color=TEXT_COLOR,
            )

            axes[row, 1].imshow(
                display_target,
                cmap=color_map,
                norm=color_norm,
                interpolation="nearest",
            )
            axes[row, 1].set_title(
                "Ground truth",
                color=TEXT_COLOR,
            )

            axes[row, 2].imshow(
                prediction,
                cmap=color_map,
                norm=color_norm,
                interpolation="nearest",
            )
            axes[row, 2].set_title(
                f"Model prediction | Dice {sample_dice:.3f}",
                color=TEXT_COLOR,
            )

            for axis in axes[row]:
                axis.axis("off")

            print(f"  index={sample_index}, tissue={tissue}, sample Dice={sample_dice:.3f}")

    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor=BORDER_COLOR,
            label=name.title(),
        )
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
        labelcolor=TEXT_COLOR,
        bbox_to_anchor=(0.5, 0.025),
    )
    figure.suptitle(
        "Weighted semantic U-Net | PanNuke Fold 2 validation",
        color=TEXT_COLOR,
        fontsize=16,
    )
    figure.text(
        0.5,
        0.008,
        "Seeded tissue-diverse selection; predictions from the best validation checkpoint.",
        ha="center",
        color=TEXT_COLOR,
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.065, 1, 0.97))

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        arguments.output,
        dpi=200,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    print("\nTarget pixel counts:")
    print(
        dict(
            zip(
                CLASS_NAMES,
                target_counts.tolist(),
                strict=True,
            )
        )
    )

    print("\nPredicted pixel counts:")
    print(
        dict(
            zip(
                CLASS_NAMES,
                prediction_counts.tolist(),
                strict=True,
            )
        )
    )

    print("\nCheckpoint:", arguments.checkpoint)
    print("Saved:", arguments.output)


if __name__ == "__main__":
    main()

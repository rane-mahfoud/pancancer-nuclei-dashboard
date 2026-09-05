"""Check connected-component post-processing on a real PanNuke sample."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from pancancer_nuclei.data.pannuke import PanNukeDataset
from pancancer_nuclei.data.transforms import create_validation_transforms
from pancancer_nuclei.models.unet import UNet
from pancancer_nuclei.postprocessing.connected_components import (
    semantic_to_instances,
)

plt.switch_backend("Agg")

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
CACHE_DIR = "data/raw/huggingface_cache"

CHECKPOINT_PATH = Path("models/checkpoints/semantic_unet_weighted_best.pt")
OUTPUT_PATH = Path("reports/figures/connected_components_example.png")

CLASS_NAMES = (
    "neoplastic",
    "inflammatory",
    "connective",
    "dead",
    "epithelial",
)

CLASS_COLORS = (
    "#d98ba6",
    "#a8c7a5",
    "#8fb8d8",
    "#e8b38e",
    "#b7a0d8",
)

BACKGROUND_COLOR = "#fff7fb"
TEXT_COLOR = "#4c3444"


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(description="Inspect connected-component predictions.")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=1580,
    )
    parser.add_argument(
        "--minimum-area",
        type=int,
        default=10,
    )
    return parser.parse_args()


def convert_to_numpy(value: object) -> np.ndarray:
    """Convert a tensor or array-like object into a NumPy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def count_categories(categories: np.ndarray) -> dict[str, int]:
    """Count instances belonging to each nucleus class."""
    counts = Counter(int(category) for category in categories)

    return {
        class_name: counts.get(class_index, 0) for class_index, class_name in enumerate(CLASS_NAMES)
    }


def draw_instances(
    axis: plt.Axes,
    image: np.ndarray,
    instance_masks: np.ndarray,
    categories: np.ndarray,
    title: str,
) -> None:
    """Draw coloured instance boundaries over a tissue image."""
    axis.imshow(image)

    for instance_mask, category in zip(
        instance_masks,
        categories,
        strict=True,
    ):
        axis.contour(
            instance_mask.astype(np.uint8),
            levels=[0.5],
            colors=["#4c3444"],
            linewidths=3.0,
            alpha=0.75,
        )
        axis.contour(
            instance_mask.astype(np.uint8),
            levels=[0.5],
            colors=[CLASS_COLORS[int(category)]],
            linewidths=1.8,
        )

    axis.set_title(
        f"{title}\n{len(instance_masks)} nuclei",
        color=TEXT_COLOR,
    )
    axis.axis("off")


def main() -> None:
    """Run the model and reconstruct individual predicted nuclei."""
    arguments = parse_arguments()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

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

    sample = dataset[arguments.sample_index]
    image_tensor = sample["image"]
    tissue = str(sample["tissue"])

    with torch.inference_mode():
        logits = model(image_tensor.unsqueeze(0).to(device))
        semantic_prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    prediction = semantic_to_instances(
        semantic_prediction,
        minimum_area=arguments.minimum_area,
    )

    image = np.clip(
        image_tensor.permute(1, 2, 0).numpy(),
        0.0,
        1.0,
    )
    ground_truth_masks = convert_to_numpy(sample["instance_masks"]).astype(bool)
    ground_truth_categories = convert_to_numpy(sample["categories"]).astype(np.int64)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14, 5),
    )
    figure.patch.set_facecolor(BACKGROUND_COLOR)

    axes[0].imshow(image)
    axes[0].set_title(
        f"Original image\n{tissue}",
        color=TEXT_COLOR,
    )
    axes[0].axis("off")

    draw_instances(
        axis=axes[1],
        image=image,
        instance_masks=ground_truth_masks,
        categories=ground_truth_categories,
        title="Ground-truth instances",
    )
    draw_instances(
        axis=axes[2],
        image=image,
        instance_masks=prediction.instance_masks,
        categories=prediction.categories,
        title="Connected-component predictions",
    )

    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor=color,
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
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.suptitle(
        "Semantic prediction converted into individual nuclei",
        color=TEXT_COLOR,
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.93))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    print("Sample index:", arguments.sample_index)
    print("Tissue:", tissue)
    print("Minimum predicted area:", arguments.minimum_area)
    print("Ground-truth nuclei:", len(ground_truth_masks))
    print("Predicted nuclei:", prediction.number_of_instances)
    print(
        "Ground-truth classes:",
        count_categories(ground_truth_categories),
    )
    print(
        "Predicted classes:",
        count_categories(prediction.categories),
    )
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

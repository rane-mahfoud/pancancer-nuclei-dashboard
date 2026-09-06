"""Create deterministic qualitative examples from the locked Fold 3 test set."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from datasets import load_dataset

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from pancancer_nuclei.data.pannuke import PanNukeDataset
from pancancer_nuclei.data.transforms import create_validation_transforms
from pancancer_nuclei.evaluation.panoptic import panoptic_quality
from pancancer_nuclei.models.boundary_unet import BoundaryAwareUNet
from pancancer_nuclei.models.unet import UNet
from pancancer_nuclei.postprocessing.connected_components import (
    semantic_to_instances,
)
from pancancer_nuclei.postprocessing.watershed import (
    boundary_predictions_to_instances,
)

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
CACHE_DIR = "data/raw/huggingface_cache"
E1_CHECKPOINT = Path("models/checkpoints/semantic_unet_weighted_best.pt")
E2_CHECKPOINT = Path("models/checkpoints/boundary_unet_weighted_best.pt")
FINAL_REPORT = Path("reports/fold3_final_evaluation.json")
OUTPUT_REPORT = Path("reports/fold3_qualitative_examples.json")
OUTPUT_FIGURE = Path("reports/figures/fold3_qualitative_examples.png")

SELECTION_SEED = 20260906
TARGET_TISSUES = (
    "Breast",
    "Colon",
    "Kidney",
    "Lung",
    "Skin",
    "Thyroid",
)
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


def load_json(path: Path) -> dict[str, Any]:
    """Load a required JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Required report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_e1_model(device: torch.device) -> UNet:
    """Load the frozen E1 checkpoint."""
    checkpoint = torch.load(
        E1_CHECKPOINT,
        map_location=device,
        weights_only=True,
    )
    state_dict = checkpoint["model_state_dict"]
    base_channels = state_dict["encoder_1.layers.0.weight"].shape[0]
    model = UNet(base_channels=base_channels).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_e2_model(device: torch.device) -> BoundaryAwareUNet:
    """Load the frozen E2 checkpoint."""
    checkpoint = torch.load(
        E2_CHECKPOINT,
        map_location=device,
        weights_only=True,
    )
    state_dict = checkpoint["model_state_dict"]
    base_channels = state_dict["encoder_1.layers.0.weight"].shape[0]
    model = BoundaryAwareUNet(base_channels=base_channels).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def convert_to_numpy(value: object) -> np.ndarray:
    """Convert tensors and array-like values into NumPy arrays."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def masks_to_instance_map(
    masks: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Convert separate instance masks into one labelled map."""
    instance_map = np.zeros(image_shape, dtype=np.int32)
    for instance_id, mask in enumerate(masks, start=1):
        instance_map[mask.astype(bool)] = instance_id
    return instance_map


def category_counts(categories: np.ndarray) -> dict[str, int]:
    """Count instances by PanNuke category."""
    counts = Counter(int(category) for category in categories)
    return {name: counts.get(index, 0) for index, name in enumerate(CLASS_NAMES)}


def tissue_name(records: Any, index: int) -> str:
    """Return a readable tissue label without transforming the image."""
    value = records[index]["tissue"]
    feature = getattr(records, "features", {}).get("tissue")
    if isinstance(value, (int, np.integer)) and hasattr(feature, "int2str"):
        return str(feature.int2str(int(value)))
    return str(value)


def select_indices(records: Any) -> list[tuple[str, int]]:
    """Select one seeded-random sample from each predeclared tissue."""
    candidates: dict[str, list[int]] = defaultdict(list)
    target_set = set(TARGET_TISSUES)
    for index in range(len(records)):
        tissue = tissue_name(records, index)
        if tissue in target_set:
            candidates[tissue].append(index)

    missing = [tissue for tissue in TARGET_TISSUES if not candidates[tissue]]
    if missing:
        raise RuntimeError(f"Fold 3 is missing target tissues: {missing}")

    generator = np.random.default_rng(SELECTION_SEED)
    return [(tissue, int(generator.choice(candidates[tissue]))) for tissue in TARGET_TISSUES]


def draw_instances(
    axis: plt.Axes,
    image: np.ndarray,
    instance_masks: np.ndarray,
    categories: np.ndarray,
    title: str,
) -> None:
    """Draw translucent class fills with high-contrast instance contours."""
    axis.imshow(image)
    overlay = np.zeros((*image.shape[:2], 4), dtype=np.float32)
    for instance_mask, category in zip(
        instance_masks,
        categories,
        strict=True,
    ):
        color = CLASS_COLORS[int(category)]
        overlay[instance_mask.astype(bool)] = to_rgba(color, alpha=0.24)

    axis.imshow(overlay)
    for instance_mask, category in zip(
        instance_masks,
        categories,
        strict=True,
    ):
        color = CLASS_COLORS[int(category)]
        axis.contour(
            instance_mask.astype(np.uint8),
            levels=[0.5],
            colors=[TEXT_COLOR],
            linewidths=2.4,
            alpha=0.85,
        )
        axis.contour(
            instance_mask.astype(np.uint8),
            levels=[0.5],
            colors=[color],
            linewidths=1.35,
        )
    axis.set_title(title, color=TEXT_COLOR, fontsize=9)
    axis.axis("off")


def main() -> None:
    """Generate the fixed-selection qualitative Fold 3 comparison."""
    if OUTPUT_REPORT.exists():
        raise FileExistsError(f"Qualitative report already exists: {OUTPUT_REPORT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")
    for path in (E1_CHECKPOINT, E2_CHECKPOINT, FINAL_REPORT):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    final_report = load_json(FINAL_REPORT)
    if final_report.get("split") != "fold3":
        raise RuntimeError("The final evaluation report is not for Fold 3.")
    e1_area = int(final_report["methods"]["e1"]["minimum_instance_area"])
    e2_configuration = final_report["methods"]["e2"]["configuration"]

    records = load_dataset(
        DATASET_NAME,
        revision=REVISION,
        split="fold3",
        cache_dir=CACHE_DIR,
    )
    selected = select_indices(records)
    dataset = PanNukeDataset(
        records,
        transform=create_validation_transforms(),
    )
    device = torch.device("cuda")
    e1_model = load_e1_model(device)
    e2_model = load_e2_model(device)

    figure, axes = plt.subplots(
        len(selected),
        4,
        figsize=(14, 19),
    )
    figure.patch.set_facecolor(BACKGROUND_COLOR)
    example_records: list[dict[str, Any]] = []

    with torch.inference_mode():
        for row, (expected_tissue, sample_index) in enumerate(selected):
            sample = dataset[sample_index]
            if sample["tissue"] != expected_tissue:
                raise RuntimeError("Tissue decoding changed between raw and prepared data.")
            image_tensor = sample["image"]
            image = np.clip(
                image_tensor.permute(1, 2, 0).numpy(),
                0.0,
                1.0,
            )
            true_masks = convert_to_numpy(sample["instance_masks"]).astype(bool)
            true_categories = convert_to_numpy(sample["categories"]).astype(np.int64)
            image_shape = tuple(image_tensor.shape[-2:])
            true_map = masks_to_instance_map(true_masks, image_shape)

            batch = image_tensor.unsqueeze(0).to(device)
            e1_semantic = e1_model(batch).argmax(dim=1).squeeze(0).cpu().numpy()
            e2_semantic_logits, e2_spatial_logits = e2_model(batch)
            e2_semantic = e2_semantic_logits.argmax(dim=1).squeeze(0).cpu().numpy()
            e2_spatial = torch.softmax(e2_spatial_logits, dim=1).squeeze(0).cpu().numpy()
            e1_prediction = semantic_to_instances(
                e1_semantic,
                minimum_area=e1_area,
            )
            e2_prediction = boundary_predictions_to_instances(
                semantic_mask=e2_semantic,
                spatial_probabilities=e2_spatial,
                seed_threshold=e2_configuration["seed_threshold"],
                minimum_seed_area=e2_configuration["minimum_seed_area"],
                minimum_instance_area=e2_configuration["minimum_instance_area"],
            )
            e1_quality = panoptic_quality(
                true_map,
                e1_prediction.instance_map,
            )
            e2_quality = panoptic_quality(
                true_map,
                e2_prediction.instance_map,
            )

            axes[row, 0].imshow(image)
            axes[row, 0].set_title(
                f"Original | {expected_tissue}\nFold 3 index {sample_index}",
                color=TEXT_COLOR,
                fontsize=9,
            )
            axes[row, 0].axis("off")
            draw_instances(
                axes[row, 1],
                image,
                true_masks,
                true_categories,
                f"Ground truth\n{len(true_masks)} nuclei",
            )
            draw_instances(
                axes[row, 2],
                image,
                e1_prediction.instance_masks,
                e1_prediction.categories,
                (
                    "E1 prediction\n"
                    f"{e1_prediction.number_of_instances} nuclei | "
                    f"bPQ {e1_quality.panoptic_quality:.3f}"
                ),
            )
            draw_instances(
                axes[row, 3],
                image,
                e2_prediction.instance_masks,
                e2_prediction.categories,
                (
                    "E2 prediction\n"
                    f"{e2_prediction.number_of_instances} nuclei | "
                    f"bPQ {e2_quality.panoptic_quality:.3f}"
                ),
            )
            example_records.append(
                {
                    "fold3_index": sample_index,
                    "tissue": expected_tissue,
                    "ground_truth": {
                        "number_of_nuclei": len(true_masks),
                        "class_counts": category_counts(true_categories),
                    },
                    "e1": {
                        "number_of_nuclei": (e1_prediction.number_of_instances),
                        "class_counts": category_counts(e1_prediction.categories),
                        "binary_pq": e1_quality.panoptic_quality,
                        "detection_quality": e1_quality.detection_quality,
                        "segmentation_quality": (e1_quality.segmentation_quality),
                    },
                    "e2": {
                        "number_of_nuclei": (e2_prediction.number_of_instances),
                        "class_counts": category_counts(e2_prediction.categories),
                        "binary_pq": e2_quality.panoptic_quality,
                        "detection_quality": e2_quality.detection_quality,
                        "segmentation_quality": (e2_quality.segmentation_quality),
                    },
                }
            )

    legend_handles = [
        Patch(facecolor=color, edgecolor=color, label=name.title())
        for name, color in zip(CLASS_NAMES, CLASS_COLORS, strict=True)
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.suptitle(
        "PanNuke Fold 3 | Deterministic qualitative comparison",
        color=TEXT_COLOR,
        fontsize=16,
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.975))
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_FIGURE,
        dpi=180,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    qualitative_report = {
        "dataset": DATASET_NAME,
        "revision": REVISION,
        "split": "fold3",
        "selection_protocol": {
            "target_tissues_declared_before_inference": list(TARGET_TISSUES),
            "selection_seed": SELECTION_SEED,
            "rule": (
                "One seeded-random sample from each predeclared tissue; "
                "selection is independent of model performance."
            ),
        },
        "parameters_source": str(FINAL_REPORT),
        "examples": example_records,
    }
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(
        json.dumps(qualitative_report, indent=2),
        encoding="utf-8",
    )

    print("Deterministic Fold 3 qualitative selection:")
    for example in example_records:
        print(
            f"  index={example['fold3_index']}, "
            f"tissue={example['tissue']}, "
            f"E1 bPQ={example['e1']['binary_pq']:.3f}, "
            f"E2 bPQ={example['e2']['binary_pq']:.3f}"
        )
    print("Selection seed:", SELECTION_SEED)
    print("Saved:", OUTPUT_REPORT)
    print("Saved:", OUTPUT_FIGURE)


if __name__ == "__main__":
    main()

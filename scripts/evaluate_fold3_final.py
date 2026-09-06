"""Run the one-shot final E1-versus-E2 evaluation on PanNuke Fold 3."""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from datasets import load_dataset

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from pancancer_nuclei.data.pannuke import PanNukeDataset
from pancancer_nuclei.data.transforms import create_validation_transforms
from pancancer_nuclei.evaluation.panoptic import panoptic_quality
from pancancer_nuclei.models.boundary_unet import BoundaryAwareUNet
from pancancer_nuclei.models.unet import UNet
from pancancer_nuclei.postprocessing.connected_components import (
    InstanceSegmentation,
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
E1_SELECTION_REPORT = Path("reports/instance_threshold_sweep_fold2.json")
E2_SELECTION_REPORT = Path("reports/watershed_parameter_sweep_fold2.json")
OUTPUT_REPORT = Path("reports/fold3_final_evaluation.json")
OUTPUT_OVERVIEW = Path("reports/figures/fold3_final_comparison.png")
OUTPUT_TISSUES = Path("reports/figures/fold3_final_by_tissue.png")

NUMBER_OF_SEMANTIC_CLASSES = 6
IGNORE_LABEL = 255
EXPECTED_E1_MINIMUM_AREA = 100
EXPECTED_E2_CONFIGURATION = {
    "seed_threshold": 0.35,
    "minimum_seed_area": 20,
    "minimum_instance_area": 100,
}

SEMANTIC_CLASS_NAMES = (
    "background",
    "neoplastic",
    "inflammatory",
    "connective",
    "dead",
    "epithelial",
)
INSTANCE_CLASS_NAMES = SEMANTIC_CLASS_NAMES[1:]
BACKGROUND_COLOR = "#fff7fb"
TEXT_COLOR = "#4c3444"
GRID_COLOR = "#e8d7e0"
E1_COLOR = "#b7a0d8"
E2_COLOR = "#d98ba6"
E1_LABEL = "E1 | Semantic U-Net + connected components"
E2_LABEL = "E2 | Boundary U-Net + hybrid separation"


@dataclass
class InstanceAccumulator:
    """Collect instance measurements for one final method."""

    binary_by_tissue: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    multiclass_by_tissue: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    per_class: dict[int, list[float]] = field(
        default_factory=lambda: {index: [] for index in range(len(INSTANCE_CLASS_NAMES))}
    )
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    matched_iou_sum: float = 0.0


def parse_arguments() -> argparse.Namespace:
    """Read non-tuning runtime options."""
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen E1 and E2 pipelines on Fold 3 once."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size; this does not change model predictions.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load a required JSON report."""
    if not path.exists():
        raise FileNotFoundError(f"Required report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_locked_configuration() -> tuple[int, dict[str, Any]]:
    """Load and verify the parameters selected exclusively on Fold 2."""
    e1_report = load_json(E1_SELECTION_REPORT)
    e2_report = load_json(E2_SELECTION_REPORT)
    e1_area = int(e1_report["best_minimum_area"])
    e2_configuration = e2_report["best_configuration"]

    if e1_area != EXPECTED_E1_MINIMUM_AREA:
        raise RuntimeError(
            "The E1 Fold 2 selection changed: "
            f"expected {EXPECTED_E1_MINIMUM_AREA}, found {e1_area}."
        )
    if e2_configuration != EXPECTED_E2_CONFIGURATION:
        raise RuntimeError(
            "The E2 Fold 2 selection changed: "
            f"expected {EXPECTED_E2_CONFIGURATION}, "
            f"found {e2_configuration}."
        )
    return e1_area, e2_configuration


def load_e1_model(device: torch.device) -> UNet:
    """Load the frozen E1 semantic U-Net."""
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
    """Load the frozen E2 boundary-aware U-Net."""
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


def update_semantic_confusion(
    confusion: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Accumulate a true-row, predicted-column semantic confusion matrix."""
    valid = targets != IGNORE_LABEL
    true_labels = targets[valid].to(torch.int64)
    predicted_labels = predictions[valid].to(torch.int64)
    encoded = true_labels * NUMBER_OF_SEMANTIC_CLASSES + predicted_labels
    counts = torch.bincount(
        encoded,
        minlength=NUMBER_OF_SEMANTIC_CLASSES**2,
    )
    confusion += counts.reshape(
        NUMBER_OF_SEMANTIC_CLASSES,
        NUMBER_OF_SEMANTIC_CLASSES,
    ).cpu()


def summarize_semantic(confusion: torch.Tensor) -> dict[str, Any]:
    """Calculate pixel accuracy and Dice scores from one confusion matrix."""
    matrix = confusion.to(torch.float64)
    true_pixels = matrix.sum(dim=1)
    predicted_pixels = matrix.sum(dim=0)
    true_positives = matrix.diag()
    denominator = true_pixels + predicted_pixels
    dice = torch.where(
        denominator > 0,
        2.0 * true_positives / denominator,
        torch.zeros_like(denominator),
    )
    total = matrix.sum()
    pixel_accuracy = float(true_positives.sum() / total) if total > 0 else 0.0
    return {
        "pixel_accuracy": pixel_accuracy,
        "macro_foreground_dice": float(dice[1:].mean()),
        "dice_per_class": {
            name: float(value)
            for name, value in zip(
                SEMANTIC_CLASS_NAMES,
                dice.tolist(),
                strict=True,
            )
        },
        "confusion_matrix": confusion.tolist(),
    }


def convert_to_numpy(value: object) -> np.ndarray:
    """Convert tensors and array-like values to NumPy."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def masks_to_instance_map(
    masks: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Convert separate boolean masks into an integer instance map."""
    instance_map = np.zeros(image_shape, dtype=np.int32)
    for instance_id, mask in enumerate(masks, start=1):
        instance_map[mask.astype(bool)] = instance_id
    return instance_map


def evaluate_instances(
    accumulator: InstanceAccumulator,
    prediction: InstanceSegmentation,
    true_masks: np.ndarray,
    true_categories: np.ndarray,
    tissue: str,
    image_shape: tuple[int, int],
) -> None:
    """Accumulate official-style binary and multiclass PQ for one sample."""
    if len(true_masks) == 0:
        return

    true_binary_map = masks_to_instance_map(true_masks, image_shape)
    binary = panoptic_quality(true_binary_map, prediction.instance_map)
    accumulator.binary_by_tissue[tissue].append(binary.panoptic_quality)
    accumulator.true_positives += binary.true_positives
    accumulator.false_positives += binary.false_positives
    accumulator.false_negatives += binary.false_negatives
    accumulator.matched_iou_sum += binary.matched_iou_sum

    sample_class_pq: list[float] = []
    for class_index in range(len(INSTANCE_CLASS_NAMES)):
        true_selection = true_categories == class_index
        if not np.any(true_selection):
            continue

        predicted_selection = prediction.categories == class_index
        true_class_map = masks_to_instance_map(
            true_masks[true_selection],
            image_shape,
        )
        predicted_class_map = masks_to_instance_map(
            prediction.instance_masks[predicted_selection],
            image_shape,
        )
        class_result = panoptic_quality(
            true_class_map,
            predicted_class_map,
        )
        sample_class_pq.append(class_result.panoptic_quality)
        accumulator.per_class[class_index].append(class_result.panoptic_quality)

    if sample_class_pq:
        accumulator.multiclass_by_tissue[tissue].append(float(np.mean(sample_class_pq)))


def mean_by_tissue(
    measurements: dict[str, list[float]],
) -> tuple[float, dict[str, float]]:
    """Average per sample and then give each tissue equal weight."""
    tissue_values = {
        tissue: float(np.mean(values)) for tissue, values in sorted(measurements.items()) if values
    }
    if not tissue_values:
        return 0.0, {}
    return float(np.mean(list(tissue_values.values()))), tissue_values


def summarize_instances(
    accumulator: InstanceAccumulator,
) -> dict[str, Any]:
    """Create final instance metrics for one method."""
    binary_pq, binary_by_tissue = mean_by_tissue(accumulator.binary_by_tissue)
    multiclass_pq, multiclass_by_tissue = mean_by_tissue(accumulator.multiclass_by_tissue)
    detection_denominator = (
        accumulator.true_positives
        + 0.5 * accumulator.false_positives
        + 0.5 * accumulator.false_negatives
    )
    detection_quality = (
        accumulator.true_positives / detection_denominator if detection_denominator > 0 else 0.0
    )
    segmentation_quality = (
        accumulator.matched_iou_sum / accumulator.true_positives
        if accumulator.true_positives > 0
        else 0.0
    )
    return {
        "binary_pq": binary_pq,
        "multiclass_pq": multiclass_pq,
        "global_detection_quality": detection_quality,
        "global_segmentation_quality": segmentation_quality,
        "per_class_pq": {
            INSTANCE_CLASS_NAMES[index]: (float(np.mean(values)) if values else None)
            for index, values in accumulator.per_class.items()
        },
        "binary_pq_by_tissue": binary_by_tissue,
        "multiclass_pq_by_tissue": multiclass_by_tissue,
        "matched_nuclei": accumulator.true_positives,
        "extra_predicted_nuclei": accumulator.false_positives,
        "missed_nuclei": accumulator.false_negatives,
    }


def relative_change(baseline: float, improved: float) -> float | None:
    """Calculate a relative percentage change when defined."""
    if baseline == 0:
        return None
    return 100.0 * (improved - baseline) / baseline


def comparison_summary(
    e1_semantic: dict[str, Any],
    e2_semantic: dict[str, Any],
    e1_instance: dict[str, Any],
    e2_instance: dict[str, Any],
) -> dict[str, Any]:
    """Calculate final E2-minus-E1 changes without selecting anything."""
    metrics = {
        "semantic_macro_foreground_dice": (
            e1_semantic["macro_foreground_dice"],
            e2_semantic["macro_foreground_dice"],
        ),
        "binary_pq": (e1_instance["binary_pq"], e2_instance["binary_pq"]),
        "multiclass_pq": (
            e1_instance["multiclass_pq"],
            e2_instance["multiclass_pq"],
        ),
    }
    summary = {
        name: {
            "e1": baseline,
            "e2": improved,
            "absolute_change": improved - baseline,
            "relative_change_percent": relative_change(baseline, improved),
        }
        for name, (baseline, improved) in metrics.items()
    }
    summary["detection_counts"] = {
        "additional_matches": (e2_instance["matched_nuclei"] - e1_instance["matched_nuclei"]),
        "change_in_extra_predictions": (
            e2_instance["extra_predicted_nuclei"] - e1_instance["extra_predicted_nuclei"]
        ),
        "change_in_missed_nuclei": (e2_instance["missed_nuclei"] - e1_instance["missed_nuclei"]),
    }
    return summary


def style_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    """Apply the project report style."""
    axis.set_facecolor(BACKGROUND_COLOR)
    axis.grid(axis=grid_axis, color=GRID_COLOR, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=TEXT_COLOR)


def add_bar_labels(axis: plt.Axes, bars: Any) -> None:
    """Add compact numeric labels to bars."""
    axis.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
        fontsize=8,
        color=TEXT_COLOR,
    )


def plot_final_overview(report: dict[str, Any]) -> None:
    """Plot final semantic, instance, and class-level results."""
    e1_semantic = report["results"]["e1"]["semantic"]
    e2_semantic = report["results"]["e2"]["semantic"]
    e1_instance = report["results"]["e1"]["instance"]
    e2_instance = report["results"]["e2"]["instance"]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
        layout="constrained",
    )
    figure.patch.set_facecolor(BACKGROUND_COLOR)
    for axis in axes.ravel():
        style_axis(axis)

    semantic_bars = axes[0, 0].bar(
        (0, 1),
        (
            e1_semantic["macro_foreground_dice"],
            e2_semantic["macro_foreground_dice"],
        ),
        color=(E1_COLOR, E2_COLOR),
        width=0.58,
    )
    axes[0, 0].set_xticks((0, 1), ("E1", "E2"))
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("Dice")
    axes[0, 0].set_title("Semantic foreground Dice")
    add_bar_labels(axes[0, 0], semantic_bars)

    positions = np.arange(2)
    width = 0.34
    e1_instance_bars = axes[0, 1].bar(
        positions - width / 2,
        (e1_instance["binary_pq"], e1_instance["multiclass_pq"]),
        width,
        color=E1_COLOR,
        label="E1",
    )
    e2_instance_bars = axes[0, 1].bar(
        positions + width / 2,
        (e2_instance["binary_pq"], e2_instance["multiclass_pq"]),
        width,
        color=E2_COLOR,
        label="E2",
    )
    axes[0, 1].set_xticks(positions, ("Binary PQ", "Multiclass PQ"))
    axes[0, 1].set_ylim(0, 0.55)
    axes[0, 1].set_ylabel("Panoptic Quality")
    axes[0, 1].set_title("Instance performance")
    axes[0, 1].legend(frameon=False)
    add_bar_labels(axes[0, 1], e1_instance_bars)
    add_bar_labels(axes[0, 1], e2_instance_bars)

    class_positions = np.arange(len(INSTANCE_CLASS_NAMES))
    e1_semantic_values = [e1_semantic["dice_per_class"][name] for name in INSTANCE_CLASS_NAMES]
    e2_semantic_values = [e2_semantic["dice_per_class"][name] for name in INSTANCE_CLASS_NAMES]
    axes[1, 0].bar(
        class_positions - width / 2,
        e1_semantic_values,
        width,
        color=E1_COLOR,
        label="E1",
    )
    axes[1, 0].bar(
        class_positions + width / 2,
        e2_semantic_values,
        width,
        color=E2_COLOR,
        label="E2",
    )
    axes[1, 0].set_xticks(
        class_positions,
        [name.title() for name in INSTANCE_CLASS_NAMES],
        rotation=18,
    )
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel("Dice")
    axes[1, 0].set_title("Semantic Dice by nucleus class")
    axes[1, 0].legend(frameon=False)

    e1_class_values = [e1_instance["per_class_pq"][name] for name in INSTANCE_CLASS_NAMES]
    e2_class_values = [e2_instance["per_class_pq"][name] for name in INSTANCE_CLASS_NAMES]
    axes[1, 1].bar(
        class_positions - width / 2,
        e1_class_values,
        width,
        color=E1_COLOR,
        label="E1",
    )
    axes[1, 1].bar(
        class_positions + width / 2,
        e2_class_values,
        width,
        color=E2_COLOR,
        label="E2",
    )
    axes[1, 1].set_xticks(
        class_positions,
        [name.title() for name in INSTANCE_CLASS_NAMES],
        rotation=18,
    )
    axes[1, 1].set_ylim(0, 0.55)
    axes[1, 1].set_ylabel("Panoptic Quality")
    axes[1, 1].set_title("Instance PQ by nucleus class")
    axes[1, 1].legend(frameon=False)

    figure.suptitle("PanNuke Fold 3 | Locked final evaluation", fontsize=16)
    OUTPUT_OVERVIEW.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_OVERVIEW,
        dpi=200,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def plot_final_tissues(report: dict[str, Any]) -> None:
    """Plot final paired multiclass PQ across Fold 3 tissues."""
    e1_values = report["results"]["e1"]["instance"]["multiclass_pq_by_tissue"]
    e2_values = report["results"]["e2"]["instance"]["multiclass_pq_by_tissue"]
    tissues = sorted(
        set(e1_values) & set(e2_values),
        key=lambda tissue: e2_values[tissue],
    )
    positions = np.arange(len(tissues))
    height = 0.36

    figure, axis = plt.subplots(figsize=(10, 10), layout="constrained")
    figure.patch.set_facecolor(BACKGROUND_COLOR)
    style_axis(axis, grid_axis="x")
    axis.barh(
        positions - height / 2,
        [e1_values[tissue] for tissue in tissues],
        height,
        color=E1_COLOR,
        label=E1_LABEL,
    )
    axis.barh(
        positions + height / 2,
        [e2_values[tissue] for tissue in tissues],
        height,
        color=E2_COLOR,
        label=E2_LABEL,
    )
    axis.set_yticks(positions, tissues)
    axis.set_xlabel("Multiclass Panoptic Quality")
    axis.set_ylabel("Tissue type")
    axis.set_title("PanNuke Fold 3 final instance performance by tissue")
    axis.legend(frameon=False, loc="lower right")
    OUTPUT_TISSUES.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_TISSUES,
        dpi=200,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def print_final_summary(report: dict[str, Any]) -> None:
    """Print the key locked-test results."""
    e1 = report["results"]["e1"]
    e2 = report["results"]["e2"]
    changes = report["comparison"]
    print("------------------------")
    print("LOCKED FOLD 3 FINAL RESULTS")
    print("E1:")
    print(f"  semantic Dice={e1['semantic']['macro_foreground_dice']:.4f}")
    print(f"  bPQ={e1['instance']['binary_pq']:.4f}")
    print(f"  mPQ={e1['instance']['multiclass_pq']:.4f}")
    print("E2:")
    print(f"  semantic Dice={e2['semantic']['macro_foreground_dice']:.4f}")
    print(f"  bPQ={e2['instance']['binary_pq']:.4f}")
    print(f"  mPQ={e2['instance']['multiclass_pq']:.4f}")
    print("E2 minus E1:")
    for name in (
        "semantic_macro_foreground_dice",
        "binary_pq",
        "multiclass_pq",
    ):
        change = changes[name]
        print(
            f"  {name}: {change['absolute_change']:+.4f} "
            f"({change['relative_change_percent']:+.1f}%)"
        )
    counts = changes["detection_counts"]
    print(f"  additional matches: {counts['additional_matches']:+d}")
    print(f"  change in extra predictions: {counts['change_in_extra_predictions']:+d}")
    print(f"  change in missed nuclei: {counts['change_in_missed_nuclei']:+d}")
    print("Fold 3 parameters were not tuned.")
    print("Saved:", OUTPUT_REPORT)
    print("Saved:", OUTPUT_OVERVIEW)
    print("Saved:", OUTPUT_TISSUES)


def main() -> None:
    """Evaluate both frozen pipelines on Fold 3 and save final artifacts."""
    arguments = parse_arguments()
    if arguments.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if OUTPUT_REPORT.exists():
        raise FileExistsError(
            f"Locked result already exists: {OUTPUT_REPORT}. "
            "Refusing to overwrite the final Fold 3 evaluation."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")
    for path in (E1_CHECKPOINT, E2_CHECKPOINT):
        if not path.exists():
            raise FileNotFoundError(f"Required checkpoint not found: {path}")

    e1_area, e2_configuration = validate_locked_configuration()
    print("FINAL LOCKED EVALUATION — NO PARAMETER TUNING")
    print("E1 minimum area selected on Fold 2:", e1_area)
    print("E2 configuration selected on Fold 2:", e2_configuration)

    device = torch.device("cuda")
    e1_model = load_e1_model(device)
    e2_model = load_e2_model(device)
    records = load_dataset(
        DATASET_NAME,
        revision=REVISION,
        split="fold3",
        cache_dir=CACHE_DIR,
    )
    dataset = PanNukeDataset(
        records,
        transform=create_validation_transforms(),
    )
    number_of_samples = len(dataset)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Fold 3 samples:", number_of_samples)
    print("Batch size:", arguments.batch_size)
    print("------------------------")

    semantic_confusions = {
        "e1": torch.zeros(
            NUMBER_OF_SEMANTIC_CLASSES,
            NUMBER_OF_SEMANTIC_CLASSES,
            dtype=torch.int64,
        ),
        "e2": torch.zeros(
            NUMBER_OF_SEMANTIC_CLASSES,
            NUMBER_OF_SEMANTIC_CLASSES,
            dtype=torch.int64,
        ),
    }
    instance_accumulators = {
        "e1": InstanceAccumulator(),
        "e2": InstanceAccumulator(),
    }
    started = time.perf_counter()

    with torch.inference_mode():
        for batch_start in range(0, number_of_samples, arguments.batch_size):
            batch_indices = range(
                batch_start,
                min(batch_start + arguments.batch_size, number_of_samples),
            )
            samples = [dataset[index] for index in batch_indices]
            images = torch.stack([sample["image"] for sample in samples]).to(device)
            targets = torch.stack([sample["semantic_mask"] for sample in samples])

            e1_logits = e1_model(images)
            e2_semantic_logits, e2_spatial_logits = e2_model(images)
            e1_masks = e1_logits.argmax(dim=1).cpu()
            e2_masks = e2_semantic_logits.argmax(dim=1).cpu()
            e2_spatial = torch.softmax(e2_spatial_logits, dim=1).cpu()
            update_semantic_confusion(
                semantic_confusions["e1"],
                e1_masks,
                targets,
            )
            update_semantic_confusion(
                semantic_confusions["e2"],
                e2_masks,
                targets,
            )

            for sample, e1_mask, e2_mask, spatial in zip(
                samples,
                e1_masks.numpy(),
                e2_masks.numpy(),
                e2_spatial.numpy(),
                strict=True,
            ):
                true_masks = convert_to_numpy(sample["instance_masks"]).astype(bool)
                true_categories = convert_to_numpy(sample["categories"]).astype(np.int64)
                tissue = str(sample["tissue"])
                image_shape = tuple(e1_mask.shape)
                e1_prediction = semantic_to_instances(
                    e1_mask,
                    minimum_area=e1_area,
                )
                e2_prediction = boundary_predictions_to_instances(
                    semantic_mask=e2_mask,
                    spatial_probabilities=spatial,
                    seed_threshold=e2_configuration["seed_threshold"],
                    minimum_seed_area=e2_configuration["minimum_seed_area"],
                    minimum_instance_area=e2_configuration["minimum_instance_area"],
                )
                evaluate_instances(
                    instance_accumulators["e1"],
                    e1_prediction,
                    true_masks,
                    true_categories,
                    tissue,
                    image_shape,
                )
                evaluate_instances(
                    instance_accumulators["e2"],
                    e2_prediction,
                    true_masks,
                    true_categories,
                    tissue,
                    image_shape,
                )

            completed = min(
                batch_start + len(samples),
                number_of_samples,
            )
            if completed % 100 == 0 or completed == number_of_samples:
                print(f"Processed {completed}/{number_of_samples}")

    duration_seconds = time.perf_counter() - started
    e1_semantic = summarize_semantic(semantic_confusions["e1"])
    e2_semantic = summarize_semantic(semantic_confusions["e2"])
    e1_instance = summarize_instances(instance_accumulators["e1"])
    e2_instance = summarize_instances(instance_accumulators["e2"])
    report = {
        "dataset": DATASET_NAME,
        "revision": REVISION,
        "split": "fold3",
        "number_of_samples": number_of_samples,
        "protocol": {
            "purpose": "One-shot final evaluation",
            "parameters_selected_on": "fold2",
            "parameters_tuned_on_fold3": False,
            "fold3_results_used_for_model_selection": False,
            "overwrite_protection": True,
        },
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "batch_size": arguments.batch_size,
            "duration_seconds": duration_seconds,
        },
        "methods": {
            "e1": {
                "checkpoint": str(E1_CHECKPOINT),
                "postprocessing": "Per-class connected components",
                "minimum_instance_area": e1_area,
                "parameter_source": str(E1_SELECTION_REPORT),
            },
            "e2": {
                "checkpoint": str(E2_CHECKPOINT),
                "postprocessing": (
                    "Boundary-aware watershed for neoplastic, connective, "
                    "dead, and epithelial nuclei; connected components for "
                    "inflammatory nuclei"
                ),
                "configuration": e2_configuration,
                "parameter_source": str(E2_SELECTION_REPORT),
            },
        },
        "results": {
            "e1": {
                "semantic": e1_semantic,
                "instance": e1_instance,
            },
            "e2": {
                "semantic": e2_semantic,
                "instance": e2_instance,
            },
        },
        "comparison": comparison_summary(
            e1_semantic,
            e2_semantic,
            e1_instance,
            e2_instance,
        ),
    }
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    plot_final_overview(report)
    plot_final_tissues(report)
    print_final_summary(report)


if __name__ == "__main__":
    main()

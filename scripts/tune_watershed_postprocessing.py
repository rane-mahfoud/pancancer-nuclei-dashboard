"""Tune boundary-aware watershed post-processing on PanNuke Fold 2."""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from pancancer_nuclei.data.pannuke import PanNukeDataset
from pancancer_nuclei.data.transforms import create_validation_transforms
from pancancer_nuclei.evaluation.panoptic import panoptic_quality
from pancancer_nuclei.models.boundary_unet import BoundaryAwareUNet
from pancancer_nuclei.postprocessing.connected_components import (
    InstanceSegmentation,
)
from pancancer_nuclei.postprocessing.watershed import (
    boundary_predictions_to_instances,
)

plt.switch_backend("Agg")

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
CACHE_DIR = "data/raw/huggingface_cache"
CHECKPOINT_PATH = Path("models/checkpoints/boundary_unet_weighted_best.pt")

SEED_THRESHOLDS = (0.20, 0.275, 0.35, 0.425, 0.50)
MINIMUM_SEED_AREAS = (10, 20, 40)
MINIMUM_INSTANCE_AREAS = (100, 150)

CLASS_NAMES = (
    "neoplastic",
    "inflammatory",
    "connective",
    "dead",
    "epithelial",
)


@dataclass(frozen=True)
class WatershedConfiguration:
    """One candidate watershed parameter combination."""

    seed_threshold: float
    minimum_seed_area: int
    minimum_instance_area: int


@dataclass
class ResultAccumulator:
    """Store validation measurements for one configuration."""

    binary_by_tissue: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    multiclass_by_tissue: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    per_class: dict[int, list[float]] = field(
        default_factory=lambda: {class_index: [] for class_index in range(len(CLASS_NAMES))}
    )
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


def parse_arguments() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(
        description="Tune boundary-aware watershed parameters on Fold 2."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of validation samples for a smoke check.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def create_configurations() -> tuple[WatershedConfiguration, ...]:
    """Create the fixed, predeclared parameter grid."""
    return tuple(
        WatershedConfiguration(
            seed_threshold=seed_threshold,
            minimum_seed_area=minimum_seed_area,
            minimum_instance_area=minimum_instance_area,
        )
        for seed_threshold in SEED_THRESHOLDS
        for minimum_seed_area in MINIMUM_SEED_AREAS
        for minimum_instance_area in MINIMUM_INSTANCE_AREAS
    )


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


def evaluate_prediction(
    accumulator: ResultAccumulator,
    prediction: InstanceSegmentation,
    true_masks: np.ndarray,
    true_categories: np.ndarray,
    tissue: str,
    image_shape: tuple[int, int],
) -> None:
    """Add one sample's official-style bPQ and mPQ measurements."""
    if len(true_masks) == 0:
        return

    true_binary_map = masks_to_instance_map(true_masks, image_shape)
    binary_result = panoptic_quality(
        true_map=true_binary_map,
        predicted_map=prediction.instance_map,
    )
    accumulator.binary_by_tissue[tissue].append(binary_result.panoptic_quality)
    accumulator.true_positives += binary_result.true_positives
    accumulator.false_positives += binary_result.false_positives
    accumulator.false_negatives += binary_result.false_negatives

    sample_class_pq: list[float] = []
    for class_index in range(len(CLASS_NAMES)):
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
            true_map=true_class_map,
            predicted_map=predicted_class_map,
        )
        sample_class_pq.append(class_result.panoptic_quality)
        accumulator.per_class[class_index].append(class_result.panoptic_quality)

    if sample_class_pq:
        accumulator.multiclass_by_tissue[tissue].append(float(np.mean(sample_class_pq)))


def mean_by_tissue(
    measurements: dict[str, list[float]],
) -> tuple[float, dict[str, float]]:
    """Average per sample, then give every tissue equal weight."""
    tissue_results = {
        tissue: float(np.mean(values)) for tissue, values in sorted(measurements.items()) if values
    }
    if not tissue_results:
        return 0.0, {}
    return float(np.mean(list(tissue_results.values()))), tissue_results


def summarize_configuration(
    configuration: WatershedConfiguration,
    accumulator: ResultAccumulator,
) -> dict[str, object]:
    """Create a JSON-compatible result for one configuration."""
    binary_pq, binary_by_tissue = mean_by_tissue(accumulator.binary_by_tissue)
    multiclass_pq, multiclass_by_tissue = mean_by_tissue(accumulator.multiclass_by_tissue)
    per_class_pq = {
        CLASS_NAMES[class_index]: (float(np.mean(values)) if values else None)
        for class_index, values in accumulator.per_class.items()
    }
    return {
        "seed_threshold": configuration.seed_threshold,
        "minimum_seed_area": configuration.minimum_seed_area,
        "minimum_instance_area": configuration.minimum_instance_area,
        "binary_pq": binary_pq,
        "multiclass_pq": multiclass_pq,
        "per_class_pq": per_class_pq,
        "binary_pq_by_tissue": binary_by_tissue,
        "multiclass_pq_by_tissue": multiclass_by_tissue,
        "matched_nuclei": accumulator.true_positives,
        "extra_predicted_nuclei": accumulator.false_positives,
        "missed_nuclei": accumulator.false_negatives,
    }


def plot_results(
    results: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Plot mPQ heatmaps for both final-area settings."""
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.8),
        sharey=True,
        layout="constrained",
    )
    figure.patch.set_facecolor("#fff7fb")
    color_map = LinearSegmentedColormap.from_list(
        "pastel_pink",
        ("#fff7fb", "#f5dce7", "#eabed0", "#d98ba6"),
    )

    values: list[float] = []
    matrices: list[np.ndarray] = []
    for minimum_instance_area in MINIMUM_INSTANCE_AREAS:
        matrix = np.zeros(
            (len(MINIMUM_SEED_AREAS), len(SEED_THRESHOLDS)),
            dtype=np.float64,
        )
        for row, minimum_seed_area in enumerate(MINIMUM_SEED_AREAS):
            for column, seed_threshold in enumerate(SEED_THRESHOLDS):
                result = next(
                    candidate
                    for candidate in results
                    if candidate["minimum_instance_area"] == minimum_instance_area
                    and candidate["minimum_seed_area"] == minimum_seed_area
                    and candidate["seed_threshold"] == seed_threshold
                )
                matrix[row, column] = float(result["multiclass_pq"])
        matrices.append(matrix)
        values.extend(matrix.ravel().tolist())

    lower = min(values)
    upper = max(values)
    if lower == upper:
        upper = lower + 1.0e-6

    image = None
    for axis, matrix, minimum_instance_area in zip(
        axes,
        matrices,
        MINIMUM_INSTANCE_AREAS,
        strict=True,
    ):
        axis.set_facecolor("#fff7fb")
        image = axis.imshow(
            matrix,
            cmap=color_map,
            vmin=lower,
            vmax=upper,
            aspect="auto",
        )
        axis.set_title(f"Minimum final nucleus area: {minimum_instance_area} pixels")
        axis.set_xticks(range(len(SEED_THRESHOLDS)))
        axis.set_xticklabels([f"{threshold:g}" for threshold in SEED_THRESHOLDS])
        axis.set_yticks(range(len(MINIMUM_SEED_AREAS)))
        axis.set_yticklabels(MINIMUM_SEED_AREAS)
        axis.set_xlabel("Interior seed threshold")

        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.3f}",
                    ha="center",
                    va="center",
                    color="#4c3444",
                    fontweight="bold",
                )

    axes[0].set_ylabel("Minimum seed area (pixels)")
    if image is not None:
        figure.colorbar(
            image,
            ax=axes,
            label="Multiclass PQ",
            shrink=0.82,
            pad=0.04,
        )
    figure.suptitle("Boundary-aware watershed selection | PanNuke Fold 2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def main() -> None:
    """Evaluate the fixed watershed grid on Fold 2."""
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if arguments.limit is not None and arguments.limit < 1:
        raise ValueError("--limit must be at least 1.")

    device = torch.device("cuda")
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )
    state_dict = checkpoint["model_state_dict"]
    base_channels = state_dict["encoder_1.layers.0.weight"].shape[0]
    model = BoundaryAwareUNet(base_channels=base_channels).to(device)
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
    sample_indices = np.arange(len(dataset))
    if arguments.limit is not None:
        generator = np.random.default_rng(42)
        sample_indices = generator.choice(
            sample_indices,
            size=min(arguments.limit, len(dataset)),
            replace=False,
        )
        sample_indices.sort()

    configurations = create_configurations()
    accumulators = {configuration: ResultAccumulator() for configuration in configurations}
    total_samples = len(sample_indices)
    print("Validation samples:", total_samples)
    print("Watershed configurations:", len(configurations))
    print("Seed thresholds:", SEED_THRESHOLDS)
    print("Minimum seed areas:", MINIMUM_SEED_AREAS)
    print("Minimum instance areas:", MINIMUM_INSTANCE_AREAS)
    print("------------------------")

    with torch.inference_mode():
        for batch_start in range(0, total_samples, arguments.batch_size):
            batch_indices = sample_indices[batch_start : batch_start + arguments.batch_size]
            samples = [dataset[int(index)] for index in batch_indices]
            images = torch.stack([sample["image"] for sample in samples]).to(device)
            semantic_logits, spatial_logits = model(images)
            semantic_masks = semantic_logits.argmax(dim=1).cpu().numpy()
            spatial_probabilities = torch.softmax(spatial_logits, dim=1).cpu().numpy()

            for sample, semantic_mask, spatial_probability in zip(
                samples,
                semantic_masks,
                spatial_probabilities,
                strict=True,
            ):
                true_masks = convert_to_numpy(sample["instance_masks"]).astype(bool)
                true_categories = convert_to_numpy(sample["categories"]).astype(np.int64)
                tissue = str(sample["tissue"])
                image_shape = tuple(semantic_mask.shape)

                for configuration in configurations:
                    prediction = boundary_predictions_to_instances(
                        semantic_mask=semantic_mask,
                        spatial_probabilities=spatial_probability,
                        seed_threshold=configuration.seed_threshold,
                        minimum_seed_area=configuration.minimum_seed_area,
                        minimum_instance_area=(configuration.minimum_instance_area),
                    )
                    evaluate_prediction(
                        accumulator=accumulators[configuration],
                        prediction=prediction,
                        true_masks=true_masks,
                        true_categories=true_categories,
                        tissue=tissue,
                        image_shape=image_shape,
                    )

            completed = min(batch_start + len(batch_indices), total_samples)
            if completed % 100 == 0 or completed == total_samples:
                print(f"Processed {completed}/{total_samples}")

    results = [
        summarize_configuration(configuration, accumulators[configuration])
        for configuration in configurations
    ]
    best_result = max(
        results,
        key=lambda result: (
            float(result["multiclass_pq"]),
            float(result["binary_pq"]),
        ),
    )

    suffix = "fold2" if arguments.limit is None else "smoke"
    report_path = Path(f"reports/watershed_parameter_sweep_{suffix}.json")
    figure_path = Path(f"reports/figures/watershed_parameter_sweep_{suffix}.png")
    report = {
        "dataset": DATASET_NAME,
        "revision": REVISION,
        "split": "fold2",
        "number_of_samples": total_samples,
        "checkpoint": str(CHECKPOINT_PATH),
        "selection_rule": "Highest Fold 2 multiclass PQ; binary PQ breaks ties.",
        "parameter_grid": {
            "seed_thresholds": SEED_THRESHOLDS,
            "minimum_seed_areas": MINIMUM_SEED_AREAS,
            "minimum_instance_areas": MINIMUM_INSTANCE_AREAS,
        },
        "best_configuration": {
            "seed_threshold": best_result["seed_threshold"],
            "minimum_seed_area": best_result["minimum_seed_area"],
            "minimum_instance_area": best_result["minimum_instance_area"],
        },
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_results(results, figure_path)

    print("------------------------")
    for result in sorted(
        results,
        key=lambda item: float(item["multiclass_pq"]),
        reverse=True,
    ):
        print(
            f"seed={result['seed_threshold']:.2f}, "
            f"seed area={result['minimum_seed_area']:>2}, "
            f"nucleus area={result['minimum_instance_area']:>3}: "
            f"bPQ={result['binary_pq']:.4f}, "
            f"mPQ={result['multiclass_pq']:.4f}"
        )
    print("------------------------")
    print("Selected configuration:", report["best_configuration"])
    print("Saved report:", report_path)
    print("Saved figure:", figure_path)


if __name__ == "__main__":
    main()

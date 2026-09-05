"""Tune connected-component post-processing on PanNuke Fold 2."""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt

from pancancer_nuclei.data.pannuke import PanNukeDataset
from pancancer_nuclei.data.transforms import create_validation_transforms
from pancancer_nuclei.evaluation.panoptic import panoptic_quality
from pancancer_nuclei.models.unet import UNet
from pancancer_nuclei.postprocessing.connected_components import (
    InstanceSegmentation,
    semantic_to_instances,
)

plt.switch_backend("Agg")

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
CACHE_DIR = "data/raw/huggingface_cache"

CHECKPOINT_PATH = Path("models/checkpoints/semantic_unet_weighted_best.pt")

THRESHOLDS = (1, 5, 10, 20, 30, 50, 75, 100, 150, 200)

CLASS_NAMES = (
    "neoplastic",
    "inflammatory",
    "connective",
    "dead",
    "epithelial",
)


@dataclass
class ThresholdAccumulator:
    """Store validation measurements for one area threshold."""

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
        description="Tune connected-component area threshold on Fold 2."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of validation samples for a smoke check.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    return parser.parse_args()


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
    accumulator: ThresholdAccumulator,
    prediction: InstanceSegmentation,
    true_masks: np.ndarray,
    true_categories: np.ndarray,
    tissue: str,
    image_shape: tuple[int, int],
) -> None:
    """Add one sample's official-style bPQ and mPQ measurements."""
    if len(true_masks) == 0:
        return

    true_binary_map = masks_to_instance_map(
        true_masks,
        image_shape,
    )
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
        class_pq = class_result.panoptic_quality

        sample_class_pq.append(class_pq)
        accumulator.per_class[class_index].append(class_pq)

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

    overall_result = float(np.mean(list(tissue_results.values())))
    return overall_result, tissue_results


def summarize_threshold(
    minimum_area: int,
    accumulator: ThresholdAccumulator,
) -> dict[str, object]:
    """Create a JSON-compatible result for one threshold."""
    binary_pq, binary_by_tissue = mean_by_tissue(accumulator.binary_by_tissue)
    multiclass_pq, multiclass_by_tissue = mean_by_tissue(accumulator.multiclass_by_tissue)

    per_class_pq = {
        CLASS_NAMES[class_index]: (float(np.mean(values)) if values else None)
        for class_index, values in accumulator.per_class.items()
    }

    return {
        "minimum_area": minimum_area,
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
    """Plot bPQ and mPQ against the minimum-area threshold."""
    thresholds = [int(result["minimum_area"]) for result in results]
    binary_scores = [float(result["binary_pq"]) for result in results]
    multiclass_scores = [float(result["multiclass_pq"]) for result in results]

    figure, axis = plt.subplots(figsize=(9, 5.5))
    figure.patch.set_facecolor("#fff7fb")
    axis.set_facecolor("#fff7fb")

    axis.plot(
        thresholds,
        binary_scores,
        color="#d98ba6",
        marker="o",
        linewidth=2.5,
        label="Binary PQ",
    )
    axis.plot(
        thresholds,
        multiclass_scores,
        color="#b7a0d8",
        marker="o",
        linewidth=2.5,
        label="Multiclass PQ",
    )

    axis.set_title(
        "Connected-component threshold selection | Fold 2",
        color="#4c3444",
    )
    axis.set_xlabel("Minimum predicted nucleus area (pixels)")
    axis.set_ylabel("Panoptic Quality")
    axis.grid(color="#e8d7e0", alpha=0.8)
    axis.legend(frameon=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def main() -> None:
    """Evaluate connected-component thresholds on Fold 2."""
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

    sample_indices = np.arange(len(dataset))

    if arguments.limit is not None:
        if arguments.limit < 1:
            raise ValueError("--limit must be at least 1.")

        number_selected = min(arguments.limit, len(dataset))
        generator = np.random.default_rng(42)
        sample_indices = generator.choice(
            sample_indices,
            size=number_selected,
            replace=False,
        )
        sample_indices.sort()

    accumulators = {threshold: ThresholdAccumulator() for threshold in THRESHOLDS}

    total_samples = len(sample_indices)
    print("Validation samples:", total_samples)
    print("Area thresholds:", THRESHOLDS)
    print("------------------------")

    with torch.inference_mode():
        for batch_start in range(
            0,
            total_samples,
            arguments.batch_size,
        ):
            batch_indices = sample_indices[batch_start : batch_start + arguments.batch_size]
            samples = [dataset[int(sample_index)] for sample_index in batch_indices]
            images = torch.stack([sample["image"] for sample in samples]).to(device)

            predicted_semantic_masks = model(images).argmax(dim=1).cpu().numpy()

            for sample, semantic_prediction in zip(
                samples,
                predicted_semantic_masks,
                strict=True,
            ):
                true_masks = convert_to_numpy(sample["instance_masks"]).astype(bool)
                true_categories = convert_to_numpy(sample["categories"]).astype(np.int64)
                tissue = str(sample["tissue"])
                image_shape = tuple(semantic_prediction.shape)

                for threshold in THRESHOLDS:
                    prediction = semantic_to_instances(
                        semantic_prediction,
                        minimum_area=threshold,
                    )
                    evaluate_prediction(
                        accumulator=accumulators[threshold],
                        prediction=prediction,
                        true_masks=true_masks,
                        true_categories=true_categories,
                        tissue=tissue,
                        image_shape=image_shape,
                    )

            completed = min(
                batch_start + len(batch_indices),
                total_samples,
            )

            if completed % 100 == 0 or completed == total_samples:
                print(f"Processed {completed}/{total_samples}")

    results = [
        summarize_threshold(
            minimum_area=threshold,
            accumulator=accumulators[threshold],
        )
        for threshold in THRESHOLDS
    ]

    best_result = max(
        results,
        key=lambda result: (
            float(result["multiclass_pq"]),
            float(result["binary_pq"]),
        ),
    )

    if arguments.limit is None:
        report_path = Path("reports/instance_threshold_sweep_fold2.json")
        figure_path = Path("reports/figures/instance_threshold_sweep_fold2.png")
    else:
        report_path = Path("reports/instance_threshold_sweep_smoke.json")
        figure_path = Path("reports/figures/instance_threshold_sweep_smoke.png")

    report = {
        "dataset": DATASET_NAME,
        "revision": REVISION,
        "split": "fold2",
        "number_of_samples": total_samples,
        "checkpoint": str(CHECKPOINT_PATH),
        "selection_rule": ("Highest Fold 2 multiclass PQ; binary PQ breaks ties."),
        "best_minimum_area": best_result["minimum_area"],
        "results": results,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    plot_results(results, figure_path)

    print("------------------------")

    for result in results:
        print(
            f"Minimum area {result['minimum_area']:>2}: "
            f"bPQ={result['binary_pq']:.4f}, "
            f"mPQ={result['multiclass_pq']:.4f}"
        )

    print("------------------------")
    print("Selected minimum area:", best_result["minimum_area"])
    print("Saved report:", report_path)
    print("Saved figure:", figure_path)


if __name__ == "__main__":
    main()

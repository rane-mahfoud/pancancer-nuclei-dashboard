"""Compare E1 and boundary-aware hybrid instance segmentation on Fold 2."""

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt

CONNECTED_REPORT = Path("reports/instance_threshold_sweep_fold2.json")
WATERSHED_REPORT = Path("reports/watershed_parameter_sweep_fold2.json")
OUTPUT_REPORT = Path("reports/instance_method_comparison.json")
OUTPUT_SUMMARY = Path("reports/figures/instance_method_comparison.png")
OUTPUT_TISSUES = Path("reports/figures/instance_method_comparison_by_tissue.png")

CLASS_NAMES = (
    "neoplastic",
    "inflammatory",
    "connective",
    "dead",
    "epithelial",
)
BACKGROUND_COLOR = "#fff7fb"
TEXT_COLOR = "#4c3444"
E1_COLOR = "#b7a0d8"
E2_COLOR = "#d98ba6"
E1_LABEL = "E1 | Semantic U-Net + connected components"
E2_LABEL = "E2 | Boundary U-Net + hybrid separation"


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON report or fail with a useful path."""
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def selected_connected_result(report: dict[str, Any]) -> dict[str, Any]:
    """Return the selected connected-component result."""
    selected_area = report["best_minimum_area"]
    return next(result for result in report["results"] if result["minimum_area"] == selected_area)


def selected_watershed_result(report: dict[str, Any]) -> dict[str, Any]:
    """Return the selected watershed result."""
    configuration = report["best_configuration"]
    return next(
        result
        for result in report["results"]
        if all(result[key] == value for key, value in configuration.items())
    )


def relative_change(baseline: float, improved: float) -> float | None:
    """Return relative percentage change when the baseline is nonzero."""
    if baseline == 0:
        return None
    return 100.0 * (improved - baseline) / baseline


def style_axis(axis: plt.Axes) -> None:
    """Apply the shared pastel report style."""
    axis.set_facecolor(BACKGROUND_COLOR)
    axis.grid(axis="y", color="#e8d7e0", alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=TEXT_COLOR)


def plot_summary(
    connected: dict[str, Any],
    watershed: dict[str, Any],
) -> None:
    """Plot overall and per-class Fold 2 PQ."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    figure.patch.set_facecolor(BACKGROUND_COLOR)
    for axis in axes:
        style_axis(axis)

    metric_names = ("Binary PQ", "Multiclass PQ")
    connected_overall = (
        connected["binary_pq"],
        connected["multiclass_pq"],
    )
    watershed_overall = (
        watershed["binary_pq"],
        watershed["multiclass_pq"],
    )
    positions = np.arange(len(metric_names))
    width = 0.34
    axes[0].bar(
        positions - width / 2,
        connected_overall,
        width,
        color=E1_COLOR,
        label=E1_LABEL,
    )
    axes[0].bar(
        positions + width / 2,
        watershed_overall,
        width,
        color=E2_COLOR,
        label=E2_LABEL,
    )
    axes[0].set_xticks(positions, metric_names)
    axes[0].set_ylabel("Panoptic Quality")
    axes[0].set_ylim(0, 0.45)
    axes[0].set_title("Overall instance performance")
    axes[0].legend(frameon=False, fontsize=8)

    for offset, values in (
        (-width / 2, connected_overall),
        (width / 2, watershed_overall),
    ):
        for position, value in zip(positions + offset, values, strict=True):
            axes[0].text(
                position,
                value + 0.009,
                f"{value:.3f}",
                ha="center",
                color=TEXT_COLOR,
                fontsize=9,
            )

    class_positions = np.arange(len(CLASS_NAMES))
    connected_classes = [connected["per_class_pq"][name] for name in CLASS_NAMES]
    watershed_classes = [watershed["per_class_pq"][name] for name in CLASS_NAMES]
    axes[1].bar(
        class_positions - width / 2,
        connected_classes,
        width,
        color=E1_COLOR,
        label="E1",
    )
    axes[1].bar(
        class_positions + width / 2,
        watershed_classes,
        width,
        color=E2_COLOR,
        label="E2",
    )
    axes[1].set_xticks(
        class_positions,
        [name.title() for name in CLASS_NAMES],
        rotation=20,
    )
    axes[1].set_ylabel("Panoptic Quality")
    axes[1].set_ylim(0, 0.40)
    axes[1].set_title("Performance by nucleus class")
    axes[1].legend(frameon=False)

    figure.suptitle("PanNuke Fold 2 | Instance-segmentation comparison")
    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_SUMMARY,
        dpi=200,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def plot_tissues(
    connected: dict[str, Any],
    watershed: dict[str, Any],
) -> None:
    """Plot paired mPQ values for every Fold 2 tissue."""
    connected_tissues = connected["multiclass_pq_by_tissue"]
    watershed_tissues = watershed["multiclass_pq_by_tissue"]
    tissues = sorted(
        set(connected_tissues) & set(watershed_tissues),
        key=lambda tissue: watershed_tissues[tissue],
    )
    positions = np.arange(len(tissues))
    height = 0.36

    figure, axis = plt.subplots(figsize=(10, 10), layout="constrained")
    figure.patch.set_facecolor(BACKGROUND_COLOR)
    axis.set_facecolor(BACKGROUND_COLOR)
    axis.grid(axis="x", color="#e8d7e0", alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)

    axis.barh(
        positions - height / 2,
        [connected_tissues[tissue] for tissue in tissues],
        height,
        color=E1_COLOR,
        label=E1_LABEL,
    )
    axis.barh(
        positions + height / 2,
        [watershed_tissues[tissue] for tissue in tissues],
        height,
        color=E2_COLOR,
        label=E2_LABEL,
    )
    axis.set_yticks(positions, tissues)
    axis.set_xlabel("Multiclass Panoptic Quality")
    axis.set_ylabel("Tissue type")
    axis.set_title("PanNuke Fold 2 instance performance by tissue")
    axis.legend(frameon=False, loc="lower right")

    OUTPUT_TISSUES.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_TISSUES,
        dpi=200,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def main() -> None:
    """Create the numerical and visual E1-versus-E2 comparison."""
    connected_report = load_json(CONNECTED_REPORT)
    watershed_report = load_json(WATERSHED_REPORT)
    connected = selected_connected_result(connected_report)
    watershed = selected_watershed_result(watershed_report)

    per_class = {
        class_name: {
            "connected_components": connected["per_class_pq"][class_name],
            "boundary_aware_hybrid": watershed["per_class_pq"][class_name],
            "absolute_change": (
                watershed["per_class_pq"][class_name] - connected["per_class_pq"][class_name]
            ),
            "relative_change_percent": relative_change(
                connected["per_class_pq"][class_name],
                watershed["per_class_pq"][class_name],
            ),
        }
        for class_name in CLASS_NAMES
    }
    tissue_names = sorted(
        set(connected["multiclass_pq_by_tissue"]) & set(watershed["multiclass_pq_by_tissue"])
    )
    per_tissue = {
        tissue: {
            "connected_components": connected["multiclass_pq_by_tissue"][tissue],
            "boundary_aware_hybrid": watershed["multiclass_pq_by_tissue"][tissue],
            "absolute_change": (
                watershed["multiclass_pq_by_tissue"][tissue]
                - connected["multiclass_pq_by_tissue"][tissue]
            ),
        }
        for tissue in tissue_names
    }

    comparison = {
        "dataset": connected_report["dataset"],
        "revision": connected_report["revision"],
        "split": "fold2",
        "fold3_used": False,
        "e1_connected_components": {
            "minimum_area": connected_report["best_minimum_area"],
            "binary_pq": connected["binary_pq"],
            "multiclass_pq": connected["multiclass_pq"],
            "matched_nuclei": connected["matched_nuclei"],
            "extra_predicted_nuclei": connected["extra_predicted_nuclei"],
            "missed_nuclei": connected["missed_nuclei"],
        },
        "e2_boundary_aware_hybrid": {
            "method": (
                "Boundary-aware watershed for neoplastic, connective, dead, "
                "and epithelial nuclei; connected components for "
                "inflammatory nuclei."
            ),
            "configuration": watershed_report["best_configuration"],
            "binary_pq": watershed["binary_pq"],
            "multiclass_pq": watershed["multiclass_pq"],
            "matched_nuclei": watershed["matched_nuclei"],
            "extra_predicted_nuclei": watershed["extra_predicted_nuclei"],
            "missed_nuclei": watershed["missed_nuclei"],
        },
        "overall_change": {
            "binary_pq_absolute": (watershed["binary_pq"] - connected["binary_pq"]),
            "binary_pq_relative_percent": relative_change(
                connected["binary_pq"], watershed["binary_pq"]
            ),
            "multiclass_pq_absolute": (watershed["multiclass_pq"] - connected["multiclass_pq"]),
            "multiclass_pq_relative_percent": relative_change(
                connected["multiclass_pq"], watershed["multiclass_pq"]
            ),
            "additional_matches": (watershed["matched_nuclei"] - connected["matched_nuclei"]),
            "change_in_extra_predictions": (
                watershed["extra_predicted_nuclei"] - connected["extra_predicted_nuclei"]
            ),
            "change_in_missed_nuclei": (watershed["missed_nuclei"] - connected["missed_nuclei"]),
        },
        "per_class": per_class,
        "per_tissue": per_tissue,
    }
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
    plot_summary(connected, watershed)
    plot_tissues(connected, watershed)

    changes = comparison["overall_change"]
    print("E1 connected components:")
    print(f"  bPQ={connected['binary_pq']:.4f}")
    print(f"  mPQ={connected['multiclass_pq']:.4f}")
    print("E2 boundary-aware hybrid:")
    print(f"  bPQ={watershed['binary_pq']:.4f}")
    print(f"  mPQ={watershed['multiclass_pq']:.4f}")
    print("Improvement:")
    print(
        f"  bPQ={changes['binary_pq_absolute']:+.4f} "
        f"({changes['binary_pq_relative_percent']:+.1f}%)"
    )
    print(
        f"  mPQ={changes['multiclass_pq_absolute']:+.4f} "
        f"({changes['multiclass_pq_relative_percent']:+.1f}%)"
    )
    print(f"  additional matches={changes['additional_matches']:+d}")
    print(f"  change in extra predictions={changes['change_in_extra_predictions']:+d}")
    print(f"  change in missed nuclei={changes['change_in_missed_nuclei']:+d}")
    print("Saved:", OUTPUT_REPORT)
    print("Saved:", OUTPUT_SUMMARY)
    print("Saved:", OUTPUT_TISSUES)


if __name__ == "__main__":
    main()

"""Plot Fold 2 instance-segmentation baseline results."""

import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

plt.switch_backend("Agg")

INPUT_PATH = Path("reports/instance_threshold_sweep_fold2.json")
FIGURE_DIRECTORY = Path("reports/figures")

CLASS_ORDER = (
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
GRID_COLOR = "#e8d7e0"


def load_selected_result() -> tuple[dict, dict]:
    """Load the validation-selected threshold result."""
    report = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    selected_area = report["best_minimum_area"]

    selected_result = next(
        result for result in report["results"] if result["minimum_area"] == selected_area
    )
    return report, selected_result


def label_vertical_bars(
    axis: plt.Axes,
    bars: list,
) -> None:
    """Write exact values above vertical bars."""
    for bar in bars:
        value = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            color=TEXT_COLOR,
        )


def plot_class_results(
    report: dict,
    result: dict,
) -> Path:
    """Plot PQ for the five nucleus classes."""
    selected_area = report["best_minimum_area"]
    per_class = result["per_class_pq"]
    values = [per_class[class_name] for class_name in CLASS_ORDER]

    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    figure.patch.set_facecolor(BACKGROUND_COLOR)
    axis.set_facecolor(BACKGROUND_COLOR)

    bars = axis.bar(
        [name.title() for name in CLASS_ORDER],
        values,
        color=CLASS_COLORS,
        edgecolor="#ffffff",
        linewidth=1.5,
    )
    label_vertical_bars(axis, bars)

    axis.set_ylim(0, max(values) * 1.25)
    axis.set_ylabel("Panoptic Quality", color=TEXT_COLOR)
    axis.set_xlabel("Nucleus class", color=TEXT_COLOR)
    axis.set_title(
        "PanNuke Fold 2 PQ by nucleus class\n"
        f"Connected components | minimum area {selected_area} pixels",
        color=TEXT_COLOR,
    )
    axis.grid(
        axis="y",
        color=GRID_COLOR,
        alpha=0.8,
    )
    axis.set_axisbelow(True)
    axis.tick_params(
        axis="x",
        rotation=15,
        colors=TEXT_COLOR,
    )
    axis.tick_params(axis="y", colors=TEXT_COLOR)

    output_path = FIGURE_DIRECTORY / "instance_pq_by_class.png"
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)
    return output_path


def plot_tissue_results(
    report: dict,
    result: dict,
) -> Path:
    """Plot binary and multiclass PQ across tissues."""
    selected_area = report["best_minimum_area"]
    binary_results = result["binary_pq_by_tissue"]
    multiclass_results = result["multiclass_pq_by_tissue"]

    tissues = sorted(
        binary_results,
        key=lambda tissue: multiclass_results.get(tissue, 0.0),
    )
    binary_values = [binary_results[tissue] for tissue in tissues]
    multiclass_values = [multiclass_results.get(tissue, 0.0) for tissue in tissues]

    positions = np.arange(len(tissues))
    bar_height = 0.36

    figure, axis = plt.subplots(figsize=(11, 10.5))
    figure.patch.set_facecolor(BACKGROUND_COLOR)
    axis.set_facecolor(BACKGROUND_COLOR)

    binary_bars = axis.barh(
        positions + bar_height / 2,
        binary_values,
        height=bar_height,
        color="#d98ba6",
        label="Binary PQ",
    )
    multiclass_bars = axis.barh(
        positions - bar_height / 2,
        multiclass_values,
        height=bar_height,
        color="#b7a0d8",
        label="Multiclass PQ",
    )

    axis.bar_label(
        binary_bars,
        labels=[f"{value:.3f}" for value in binary_values],
        padding=3,
        fontsize=8,
        color=TEXT_COLOR,
    )
    axis.bar_label(
        multiclass_bars,
        labels=[f"{value:.3f}" for value in multiclass_values],
        padding=3,
        fontsize=8,
        color=TEXT_COLOR,
    )

    maximum_value = max(binary_values + multiclass_values)
    axis.set_xlim(0, maximum_value * 1.18)
    axis.set_yticks(positions)
    axis.set_yticklabels(tissues)
    axis.set_xlabel("Panoptic Quality", color=TEXT_COLOR)
    axis.set_ylabel("Tissue type", color=TEXT_COLOR)
    axis.set_title(
        "PanNuke Fold 2 instance performance by tissue\n"
        f"Connected components | minimum area {selected_area} pixels",
        color=TEXT_COLOR,
    )
    axis.grid(
        axis="x",
        color=GRID_COLOR,
        alpha=0.8,
    )
    axis.set_axisbelow(True)
    axis.legend(frameon=True, loc="lower right")
    axis.tick_params(colors=TEXT_COLOR)

    output_path = FIGURE_DIRECTORY / "instance_pq_by_tissue.png"
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)
    return output_path


def main() -> None:
    """Generate class-wise and tissue-wise result figures."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Report not found: {INPUT_PATH}")

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    report, selected_result = load_selected_result()

    class_path = plot_class_results(report, selected_result)
    tissue_path = plot_tissue_results(report, selected_result)

    print("Selected minimum area:", report["best_minimum_area"])
    print("Binary PQ:", f"{selected_result['binary_pq']:.4f}")
    print(
        "Multiclass PQ:",
        f"{selected_result['multiclass_pq']:.4f}",
    )
    print("Saved:", class_path)
    print("Saved:", tissue_path)


if __name__ == "__main__":
    main()

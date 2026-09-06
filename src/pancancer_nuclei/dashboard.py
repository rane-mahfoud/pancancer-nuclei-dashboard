"""Prepare locked evaluation reports for the Streamlit dashboard."""

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd

FOREGROUND_CLASSES = (
    "neoplastic",
    "inflammatory",
    "connective",
    "dead",
    "epithelial",
)

METRIC_LABELS = {
    "semantic_macro_foreground_dice": "Semantic foreground Dice",
    "binary_pq": "Binary PQ",
    "multiclass_pq": "Multiclass PQ",
}


def load_json_report(path: Path) -> dict[str, Any]:
    """Load a JSON report and ensure its top level is an object."""
    if not path.exists():
        raise FileNotFoundError(f"Required dashboard report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return report


def validate_final_report(report: dict[str, Any]) -> None:
    """Validate the protocol fields required by the public dashboard."""
    if report.get("split") != "fold3":
        raise ValueError("The dashboard requires a Fold 3 final report.")
    protocol = report.get("protocol", {})
    if protocol.get("parameters_tuned_on_fold3") is not False:
        raise ValueError("Fold 3 must be marked as untuned.")
    if protocol.get("fold3_results_used_for_model_selection") is not False:
        raise ValueError("Fold 3 must not be used for model selection.")
    for method in ("e1", "e2"):
        if method not in report.get("results", {}):
            raise ValueError(f"Missing final results for {method.upper()}.")


def headline_metrics(report: dict[str, Any]) -> pd.DataFrame:
    """Return the three primary locked Fold 3 metrics."""
    validate_final_report(report)
    comparison = report["comparison"]
    rows = []
    for key, label in METRIC_LABELS.items():
        values = comparison[key]
        rows.append(
            {
                "metric_key": key,
                "metric": label,
                "e1": float(values["e1"]),
                "e2": float(values["e2"]),
                "absolute_change": float(values["absolute_change"]),
                "relative_change_percent": float(values["relative_change_percent"]),
            }
        )
    return pd.DataFrame(rows)


def class_comparison(
    report: dict[str, Any],
    task: Literal["semantic", "instance"],
) -> pd.DataFrame:
    """Return E1 and E2 measurements for every nucleus class."""
    validate_final_report(report)
    results = report["results"]
    if task == "semantic":
        e1_values = results["e1"]["semantic"]["dice_per_class"]
        e2_values = results["e2"]["semantic"]["dice_per_class"]
        metric = "Dice"
    elif task == "instance":
        e1_values = results["e1"]["instance"]["per_class_pq"]
        e2_values = results["e2"]["instance"]["per_class_pq"]
        metric = "Panoptic Quality"
    else:
        raise ValueError("task must be either 'semantic' or 'instance'.")

    rows = []
    for class_name in FOREGROUND_CLASSES:
        e1 = float(e1_values[class_name])
        e2 = float(e2_values[class_name])
        rows.append(
            {
                "class": class_name.title(),
                "metric": metric,
                "e1": e1,
                "e2": e2,
                "absolute_change": e2 - e1,
            }
        )
    return pd.DataFrame(rows)


def tissue_comparison(
    report: dict[str, Any],
    metric: Literal["binary_pq", "multiclass_pq"],
) -> pd.DataFrame:
    """Return tissue-level E1 and E2 PQ values and changes."""
    validate_final_report(report)
    if metric not in ("binary_pq", "multiclass_pq"):
        raise ValueError("Unsupported tissue metric.")
    key = f"{metric}_by_tissue"
    results = report["results"]
    e1_values = results["e1"]["instance"][key]
    e2_values = results["e2"]["instance"][key]
    tissues = sorted(set(e1_values) | set(e2_values))
    if set(e1_values) != set(e2_values):
        raise ValueError("E1 and E2 tissue sets do not match.")

    rows = []
    for tissue in tissues:
        e1 = float(e1_values[tissue])
        e2 = float(e2_values[tissue])
        rows.append(
            {
                "tissue": tissue,
                "e1": e1,
                "e2": e2,
                "absolute_change": e2 - e1,
            }
        )
    return pd.DataFrame(rows)


def detection_counts(report: dict[str, Any]) -> pd.DataFrame:
    """Return matched, extra, and missed instance counts."""
    validate_final_report(report)
    results = report["results"]
    fields = (
        ("matched_nuclei", "Matched nuclei"),
        ("extra_predicted_nuclei", "Extra predictions"),
        ("missed_nuclei", "Missed nuclei"),
    )
    rows = []
    for key, label in fields:
        e1 = int(results["e1"]["instance"][key])
        e2 = int(results["e2"]["instance"][key])
        rows.append(
            {
                "measure": label,
                "e1": e1,
                "e2": e2,
                "change": e2 - e1,
            }
        )
    return pd.DataFrame(rows)


def qualitative_comparison(report: dict[str, Any]) -> pd.DataFrame:
    """Flatten fixed qualitative-example summaries for display."""
    examples = report.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("The qualitative report contains no examples.")
    rows = []
    for example in examples:
        e1 = float(example["e1"]["binary_pq"])
        e2 = float(example["e2"]["binary_pq"])
        rows.append(
            {
                "fold3_index": int(example["fold3_index"]),
                "tissue": str(example["tissue"]),
                "ground_truth_nuclei": int(example["ground_truth"]["number_of_nuclei"]),
                "e1_nuclei": int(example["e1"]["number_of_nuclei"]),
                "e2_nuclei": int(example["e2"]["number_of_nuclei"]),
                "e1_binary_pq": e1,
                "e2_binary_pq": e2,
                "absolute_change": e2 - e1,
            }
        )
    return pd.DataFrame(rows)

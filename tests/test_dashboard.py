"""Tests for dashboard report preparation."""

import pytest

from pancancer_nuclei.dashboard import (
    class_comparison,
    detection_counts,
    headline_metrics,
    qualitative_comparison,
    tissue_comparison,
)


def create_final_report() -> dict:
    """Create a compact valid Fold 3 report."""
    semantic_e1 = {
        "dice_per_class": {
            "background": 0.9,
            "neoplastic": 0.4,
            "inflammatory": 0.5,
            "connective": 0.3,
            "dead": 0.1,
            "epithelial": 0.6,
        }
    }
    semantic_e2 = {
        "dice_per_class": {
            "background": 0.9,
            "neoplastic": 0.5,
            "inflammatory": 0.6,
            "connective": 0.4,
            "dead": 0.1,
            "epithelial": 0.7,
        }
    }
    instance_e1 = {
        "per_class_pq": {
            "neoplastic": 0.2,
            "inflammatory": 0.3,
            "connective": 0.2,
            "dead": 0.05,
            "epithelial": 0.1,
        },
        "binary_pq_by_tissue": {"Breast": 0.3, "Colon": 0.2},
        "multiclass_pq_by_tissue": {"Breast": 0.2, "Colon": 0.1},
        "matched_nuclei": 100,
        "extra_predicted_nuclei": 40,
        "missed_nuclei": 60,
    }
    instance_e2 = {
        "per_class_pq": {
            "neoplastic": 0.3,
            "inflammatory": 0.4,
            "connective": 0.25,
            "dead": 0.06,
            "epithelial": 0.2,
        },
        "binary_pq_by_tissue": {"Breast": 0.4, "Colon": 0.3},
        "multiclass_pq_by_tissue": {"Breast": 0.3, "Colon": 0.2},
        "matched_nuclei": 130,
        "extra_predicted_nuclei": 35,
        "missed_nuclei": 30,
    }
    return {
        "split": "fold3",
        "protocol": {
            "parameters_tuned_on_fold3": False,
            "fold3_results_used_for_model_selection": False,
        },
        "results": {
            "e1": {"semantic": semantic_e1, "instance": instance_e1},
            "e2": {"semantic": semantic_e2, "instance": instance_e2},
        },
        "comparison": {
            "semantic_macro_foreground_dice": {
                "e1": 0.38,
                "e2": 0.46,
                "absolute_change": 0.08,
                "relative_change_percent": 21.05,
            },
            "binary_pq": {
                "e1": 0.25,
                "e2": 0.35,
                "absolute_change": 0.10,
                "relative_change_percent": 40.0,
            },
            "multiclass_pq": {
                "e1": 0.15,
                "e2": 0.25,
                "absolute_change": 0.10,
                "relative_change_percent": 66.67,
            },
        },
    }


def test_headline_metrics_preserve_locked_comparison() -> None:
    """Primary dashboard values should come directly from the report."""
    frame = headline_metrics(create_final_report())

    assert frame["metric"].tolist() == [
        "Semantic foreground Dice",
        "Binary PQ",
        "Multiclass PQ",
    ]
    assert frame["e2"].tolist() == pytest.approx([0.46, 0.35, 0.25])


def test_class_and_tissue_comparisons_calculate_changes() -> None:
    """Class and tissue frames should calculate E2-minus-E1 changes."""
    report = create_final_report()
    classes = class_comparison(report, "instance")
    tissues = tissue_comparison(report, "multiclass_pq")

    assert classes.loc[0, "class"] == "Neoplastic"
    assert classes.loc[0, "absolute_change"] == pytest.approx(0.1)
    colon = tissues.loc[tissues["tissue"] == "Colon"].iloc[0]
    assert colon["absolute_change"] == pytest.approx(0.1)


def test_detection_counts_report_directional_changes() -> None:
    """Detection counts should retain whether each count rose or fell."""
    frame = detection_counts(create_final_report())

    assert frame["change"].tolist() == [30, -5, -30]


def test_qualitative_examples_are_flattened() -> None:
    """Fixed qualitative records should become a displayable table."""
    report = {
        "examples": [
            {
                "fold3_index": 12,
                "tissue": "Breast",
                "ground_truth": {"number_of_nuclei": 10},
                "e1": {"number_of_nuclei": 14, "binary_pq": 0.3},
                "e2": {"number_of_nuclei": 11, "binary_pq": 0.4},
            }
        ]
    }

    frame = qualitative_comparison(report)

    assert frame.loc[0, "ground_truth_nuclei"] == 10
    assert frame.loc[0, "absolute_change"] == pytest.approx(0.1)

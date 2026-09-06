"""Tests for the locked Fold 3 evaluation helpers."""

import numpy as np
import pytest
import torch

from pancancer_nuclei.postprocessing.connected_components import (
    InstanceSegmentation,
)
from scripts.evaluate_fold3_final import (
    InstanceAccumulator,
    evaluate_instances,
    summarize_instances,
    summarize_semantic,
    update_semantic_confusion,
)


def test_semantic_summary_ignores_ambiguous_pixels() -> None:
    """Ignored labels must not enter final semantic measurements."""
    confusion = torch.zeros(6, 6, dtype=torch.int64)
    targets = torch.tensor(
        [[[0, 1, 2], [255, 1, 2]]],
        dtype=torch.int64,
    )
    predictions = torch.tensor(
        [[[0, 1, 1], [4, 0, 2]]],
        dtype=torch.int64,
    )

    update_semantic_confusion(confusion, predictions, targets)
    summary = summarize_semantic(confusion)

    assert int(confusion.sum()) == 5
    assert summary["pixel_accuracy"] == pytest.approx(3.0 / 5.0)
    assert summary["dice_per_class"]["background"] == pytest.approx(2.0 / 3.0)
    assert summary["dice_per_class"]["neoplastic"] == pytest.approx(0.5)
    assert summary["dice_per_class"]["inflammatory"] == pytest.approx(2.0 / 3.0)
    assert summary["macro_foreground_dice"] == pytest.approx(7.0 / 30.0)


def test_instance_summary_handles_a_perfect_prediction() -> None:
    """A perfectly matching nucleus should produce perfect PQ components."""
    true_masks = np.zeros((1, 8, 8), dtype=bool)
    true_masks[0, 2:6, 2:6] = True
    true_map = np.zeros((8, 8), dtype=np.int32)
    true_map[true_masks[0]] = 1
    prediction = InstanceSegmentation(
        instance_masks=true_masks.copy(),
        categories=np.asarray([0], dtype=np.int64),
        instance_map=true_map.copy(),
    )
    accumulator = InstanceAccumulator()

    evaluate_instances(
        accumulator=accumulator,
        prediction=prediction,
        true_masks=true_masks,
        true_categories=np.asarray([0], dtype=np.int64),
        tissue="Synthetic",
        image_shape=(8, 8),
    )
    summary = summarize_instances(accumulator)

    assert summary["binary_pq"] == pytest.approx(1.0)
    assert summary["multiclass_pq"] == pytest.approx(1.0)
    assert summary["global_detection_quality"] == pytest.approx(1.0)
    assert summary["global_segmentation_quality"] == pytest.approx(1.0)
    assert summary["per_class_pq"]["neoplastic"] == pytest.approx(1.0)
    assert summary["matched_nuclei"] == 1
    assert summary["extra_predicted_nuclei"] == 0
    assert summary["missed_nuclei"] == 0

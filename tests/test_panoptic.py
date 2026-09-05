"""Tests for panoptic-quality evaluation."""

import numpy as np
import pytest

from pancancer_nuclei.evaluation.panoptic import panoptic_quality


def test_perfect_prediction_has_perfect_pq() -> None:
    """Identical instance maps should receive PQ equal to one."""
    true_map = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 2],
            [0, 0, 0, 2],
        ],
        dtype=np.int32,
    )

    result = panoptic_quality(true_map, true_map.copy())

    assert result.panoptic_quality == pytest.approx(1.0)
    assert result.detection_quality == pytest.approx(1.0)
    assert result.segmentation_quality == pytest.approx(1.0)
    assert result.true_positives == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0


def test_shape_error_reduces_segmentation_quality() -> None:
    """An imperfect matched shape should reduce SQ and PQ."""
    true_map = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    predicted_map = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
        ],
        dtype=np.int32,
    )

    result = panoptic_quality(true_map, predicted_map)

    assert result.detection_quality == pytest.approx(1.0)
    assert result.segmentation_quality == pytest.approx(0.8)
    assert result.panoptic_quality == pytest.approx(0.8)


def test_extra_and_missing_instances_reduce_detection_quality() -> None:
    """False positives and false negatives should reduce DQ."""
    true_map = np.zeros((7, 7), dtype=np.int32)
    true_map[1:3, 1:3] = 1
    true_map[4:6, 4:6] = 2

    predicted_map = np.zeros((7, 7), dtype=np.int32)
    predicted_map[1:3, 1:3] = 1
    predicted_map[1:3, 4:6] = 2

    result = panoptic_quality(true_map, predicted_map)

    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.detection_quality == pytest.approx(0.5)
    assert result.segmentation_quality == pytest.approx(1.0)
    assert result.panoptic_quality == pytest.approx(0.5)


def test_iou_equal_to_threshold_is_not_a_match() -> None:
    """PanNuke matching uses IoU strictly greater than 0.5."""
    true_map = np.array([[1, 1, 0]], dtype=np.int32)
    predicted_map = np.array([[1, 0, 0]], dtype=np.int32)

    result = panoptic_quality(
        true_map,
        predicted_map,
        match_iou=0.5,
    )

    assert result.true_positives == 0
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.panoptic_quality == 0.0


def test_empty_prediction_misses_all_true_instances() -> None:
    """An empty prediction should count every real nucleus as missed."""
    true_map = np.zeros((5, 5), dtype=np.int32)
    true_map[1:3, 1:3] = 1
    predicted_map = np.zeros((5, 5), dtype=np.int32)

    result = panoptic_quality(true_map, predicted_map)

    assert result.true_positives == 0
    assert result.false_positives == 0
    assert result.false_negatives == 1
    assert result.panoptic_quality == 0.0


def test_different_shapes_raise_an_error() -> None:
    """True and predicted maps must use identical image dimensions."""
    true_map = np.zeros((5, 5), dtype=np.int32)
    predicted_map = np.zeros((6, 5), dtype=np.int32)

    with pytest.raises(ValueError):
        panoptic_quality(true_map, predicted_map)

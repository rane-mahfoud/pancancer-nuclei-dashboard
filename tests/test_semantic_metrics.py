import pytest
import torch

from pancancer_nuclei.evaluation.semantic import (
    calculate_semantic_metrics,
    semantic_confusion_matrix,
)

CLASS_NAMES = ("background", "class_a", "class_b")


def test_perfect_predictions_have_perfect_metrics() -> None:
    targets = torch.tensor([[[0, 1], [2, 255]]])
    predictions = torch.full((1, 3, 2, 2), -10.0)

    predictions[0, 0, 0, 0] = 10.0
    predictions[0, 1, 0, 1] = 10.0
    predictions[0, 2, 1, 0] = 10.0

    matrix = semantic_confusion_matrix(
        predictions,
        targets,
        number_of_classes=3,
    )
    metrics = calculate_semantic_metrics(
        matrix,
        class_names=CLASS_NAMES,
    )

    assert metrics["pixel_accuracy"] == pytest.approx(1.0)
    assert metrics["macro_foreground_dice"] == pytest.approx(1.0)
    assert metrics["dice_per_class"]["class_a"] == pytest.approx(1.0)
    assert metrics["iou_per_class"]["class_b"] == pytest.approx(1.0)


def test_metrics_ignore_ambiguous_pixels() -> None:
    targets = torch.tensor([[[0, 1], [1, 255]]])
    predictions = torch.tensor([[[0, 0], [1, 2]]])

    matrix = semantic_confusion_matrix(
        predictions,
        targets,
        number_of_classes=3,
    )
    metrics = calculate_semantic_metrics(
        matrix,
        class_names=CLASS_NAMES,
    )

    assert matrix.sum().item() == 3
    assert metrics["pixel_accuracy"] == pytest.approx(2.0 / 3.0)
    assert metrics["dice_per_class"]["class_a"] == pytest.approx(2.0 / 3.0)
    assert metrics["iou_per_class"]["class_a"] == pytest.approx(0.5)
    assert metrics["dice_per_class"]["class_b"] is None

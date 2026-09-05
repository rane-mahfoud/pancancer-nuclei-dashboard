"""Semantic-segmentation evaluation metrics."""

from collections.abc import Sequence
from typing import Any

import torch

DEFAULT_CLASS_NAMES = (
    "background",
    "neoplastic",
    "inflammatory",
    "connective",
    "dead",
    "epithelial",
)


def semantic_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    number_of_classes: int = 6,
    ignore_label: int = 255,
) -> torch.Tensor:
    """Build a confusion matrix while excluding ignored pixels."""
    if predictions.ndim == targets.ndim + 1:
        if predictions.shape[1] != number_of_classes:
            raise ValueError("Prediction channels must equal number_of_classes.")
        predictions = predictions.argmax(dim=1)
    elif predictions.shape != targets.shape:
        raise ValueError("Predicted labels and targets must have matching shapes.")

    valid_pixels = targets != ignore_label
    valid_targets = targets[valid_pixels].long()
    valid_predictions = predictions[valid_pixels].long()

    if valid_targets.numel() == 0:
        return torch.zeros(
            number_of_classes,
            number_of_classes,
            dtype=torch.int64,
            device=targets.device,
        )

    if valid_targets.min() < 0 or valid_targets.max() >= number_of_classes:
        raise ValueError("Targets contain an invalid class label.")

    if valid_predictions.min() < 0 or valid_predictions.max() >= number_of_classes:
        raise ValueError("Predictions contain an invalid class label.")

    encoded_pairs = valid_targets * number_of_classes + valid_predictions
    counts = torch.bincount(
        encoded_pairs,
        minlength=number_of_classes**2,
    )

    return counts.reshape(number_of_classes, number_of_classes)


def calculate_semantic_metrics(
    confusion_matrix: torch.Tensor,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> dict[str, Any]:
    """Calculate accuracy, Dice, and IoU from a confusion matrix."""
    if confusion_matrix.ndim != 2:
        raise ValueError("The confusion matrix must be two-dimensional.")

    rows, columns = confusion_matrix.shape

    if rows != columns:
        raise ValueError("The confusion matrix must be square.")

    if len(class_names) != rows:
        raise ValueError("The number of class names must match the matrix size.")

    matrix = confusion_matrix.to(torch.float64)
    true_positives = matrix.diagonal()
    false_positives = matrix.sum(dim=0) - true_positives
    false_negatives = matrix.sum(dim=1) - true_positives

    dice_denominator = 2.0 * true_positives + false_positives + false_negatives
    iou_denominator = true_positives + false_positives + false_negatives

    dice_scores = torch.full_like(true_positives, torch.nan)
    iou_scores = torch.full_like(true_positives, torch.nan)

    valid_dice = dice_denominator > 0
    valid_iou = iou_denominator > 0

    dice_scores[valid_dice] = 2.0 * true_positives[valid_dice] / dice_denominator[valid_dice]
    iou_scores[valid_iou] = true_positives[valid_iou] / iou_denominator[valid_iou]

    total_pixels = matrix.sum()
    pixel_accuracy = (true_positives.sum() / total_pixels).item() if total_pixels > 0 else None

    foreground_dice = dice_scores[1:]
    available_foreground = ~torch.isnan(foreground_dice)
    macro_foreground_dice = (
        foreground_dice[available_foreground].mean().item() if available_foreground.any() else None
    )

    def make_class_dictionary(
        values: torch.Tensor,
    ) -> dict[str, float | None]:
        return {
            name: None if torch.isnan(value) else value.item()
            for name, value in zip(class_names, values, strict=True)
        }

    return {
        "pixel_accuracy": pixel_accuracy,
        "macro_foreground_dice": macro_foreground_dice,
        "dice_per_class": make_class_dictionary(dice_scores),
        "iou_per_class": make_class_dictionary(iou_scores),
    }

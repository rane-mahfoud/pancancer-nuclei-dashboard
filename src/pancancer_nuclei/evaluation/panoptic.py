"""Panoptic-quality metrics for nucleus-instance segmentation."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class PanopticQualityResult:
    """Detection, segmentation, and combined panoptic quality."""

    detection_quality: float
    segmentation_quality: float
    panoptic_quality: float
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_iou_sum: float


def _validate_instance_map(
    instance_map: np.ndarray,
    name: str,
) -> np.ndarray:
    """Validate and return a two-dimensional integer instance map."""
    instance_map = np.asarray(instance_map)

    if instance_map.ndim != 2:
        raise ValueError(f"{name} must have shape (height, width).")

    if not np.issubdtype(instance_map.dtype, np.integer):
        raise TypeError(f"{name} must contain integer instance IDs.")

    if np.any(instance_map < 0):
        raise ValueError(f"{name} cannot contain negative instance IDs.")

    return instance_map


def _calculate_pairwise_iou(
    true_map: np.ndarray,
    predicted_map: np.ndarray,
    true_ids: np.ndarray,
    predicted_ids: np.ndarray,
) -> np.ndarray:
    """Calculate IoU between every overlapping instance pair."""
    pairwise_iou = np.zeros(
        (len(true_ids), len(predicted_ids)),
        dtype=np.float64,
    )
    predicted_columns = {
        int(instance_id): column for column, instance_id in enumerate(predicted_ids)
    }

    for true_row, true_id in enumerate(true_ids):
        true_mask = true_map == true_id
        overlapping_ids = np.unique(predicted_map[true_mask])

        for predicted_id in overlapping_ids:
            if predicted_id == 0:
                continue

            predicted_column = predicted_columns[int(predicted_id)]
            predicted_mask = predicted_map == predicted_id

            intersection = np.logical_and(
                true_mask,
                predicted_mask,
            ).sum()
            union = np.logical_or(
                true_mask,
                predicted_mask,
            ).sum()

            pairwise_iou[true_row, predicted_column] = float(intersection) / float(union)

    return pairwise_iou


def panoptic_quality(
    true_map: np.ndarray,
    predicted_map: np.ndarray,
    match_iou: float = 0.5,
) -> PanopticQualityResult:
    """Calculate PQ using one-to-one instance matching.

    Zero represents background. Every positive integer represents one
    individual nucleus. Matches require IoU strictly greater than
    ``match_iou``.
    """
    true_map = _validate_instance_map(true_map, "true_map")
    predicted_map = _validate_instance_map(
        predicted_map,
        "predicted_map",
    )

    if true_map.shape != predicted_map.shape:
        raise ValueError("true_map and predicted_map must have the same shape.")

    if not 0.0 <= match_iou <= 1.0:
        raise ValueError("match_iou must be between 0 and 1.")

    true_ids = np.unique(true_map)
    true_ids = true_ids[true_ids != 0]

    predicted_ids = np.unique(predicted_map)
    predicted_ids = predicted_ids[predicted_ids != 0]

    pairwise_iou = _calculate_pairwise_iou(
        true_map=true_map,
        predicted_map=predicted_map,
        true_ids=true_ids,
        predicted_ids=predicted_ids,
    )

    if len(true_ids) == 0 or len(predicted_ids) == 0:
        matched_iou = np.empty(0, dtype=np.float64)
    elif match_iou >= 0.5:
        matched_rows, matched_columns = np.nonzero(pairwise_iou > match_iou)
        matched_iou = pairwise_iou[
            matched_rows,
            matched_columns,
        ]
    else:
        matched_rows, matched_columns = linear_sum_assignment(-pairwise_iou)
        assigned_iou = pairwise_iou[
            matched_rows,
            matched_columns,
        ]
        matched_iou = assigned_iou[assigned_iou > match_iou]

    true_positives = int(len(matched_iou))
    false_positives = int(len(predicted_ids) - true_positives)
    false_negatives = int(len(true_ids) - true_positives)

    denominator = true_positives + 0.5 * false_positives + 0.5 * false_negatives

    if denominator == 0:
        detection_quality = 0.0
    else:
        detection_quality = true_positives / denominator

    if true_positives == 0:
        segmentation_quality = 0.0
    else:
        segmentation_quality = float(matched_iou.sum()) / true_positives

    combined_quality = detection_quality * segmentation_quality

    return PanopticQualityResult(
        detection_quality=float(detection_quality),
        segmentation_quality=float(segmentation_quality),
        panoptic_quality=float(combined_quality),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        matched_iou_sum=float(matched_iou.sum()),
    )

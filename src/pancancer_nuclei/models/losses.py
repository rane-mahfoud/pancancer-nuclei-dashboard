"""Loss functions for multiclass nucleus segmentation."""

import torch
from torch import nn
from torch.nn import functional as F


def calculate_log_class_weights(
    pixel_counts: torch.Tensor,
    offset: float = 1.02,
) -> torch.Tensor:
    """Create gentle class weights from training pixel counts."""
    if pixel_counts.ndim != 1:
        raise ValueError("pixel_counts must be one-dimensional.")

    if pixel_counts.numel() < 2:
        raise ValueError("At least two classes are required.")

    if (pixel_counts <= 0).any():
        raise ValueError("Every class must have a positive count.")

    if offset <= 1.0:
        raise ValueError("The logarithmic offset must exceed one.")

    counts = pixel_counts.to(torch.float64)
    frequencies = counts / counts.sum()
    weights = 1.0 / torch.log(offset + frequencies)

    # Keep the average weight equal to one.
    weights = weights / weights.mean()

    return weights.to(torch.float32)


class MulticlassDiceLoss(nn.Module):
    """Measure overlap between predicted and true nucleus classes."""

    def __init__(
        self,
        ignore_label: int = 255,
        smooth: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate Dice loss while ignoring ambiguous pixels."""
        number_of_classes = logits.shape[1]
        valid_pixels = targets != self.ignore_label
        safe_targets = targets.masked_fill(~valid_pixels, 0)

        probabilities = torch.softmax(logits, dim=1)
        one_hot_targets = F.one_hot(
            safe_targets,
            num_classes=number_of_classes,
        )
        one_hot_targets = one_hot_targets.permute(0, 3, 1, 2).float()

        valid_mask = valid_pixels.unsqueeze(1)
        probabilities = probabilities * valid_mask
        one_hot_targets = one_hot_targets * valid_mask

        # Exclude background from the Dice calculation.
        probabilities = probabilities[:, 1:]
        one_hot_targets = one_hot_targets[:, 1:]

        dimensions = (0, 2, 3)
        intersection = (probabilities * one_hot_targets).sum(dimensions)
        denominator = probabilities.sum(dimensions) + one_hot_targets.sum(dimensions)

        dice_scores = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        present_classes = one_hot_targets.sum(dimensions) > 0

        if present_classes.any():
            return 1.0 - dice_scores[present_classes].mean()

        # Empty patches contain no foreground class, so cross-entropy
        # handles them while Dice contributes a differentiable zero.
        return logits.sum() * 0.0


class CombinedSegmentationLoss(nn.Module):
    """Combine pixel classification and foreground-overlap losses."""

    def __init__(
        self,
        cross_entropy_weight: float = 0.5,
        dice_weight: float = 0.5,
        ignore_label: int = 255,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        if cross_entropy_weight < 0 or dice_weight < 0:
            raise ValueError("Loss weights cannot be negative.")

        if cross_entropy_weight + dice_weight == 0:
            raise ValueError("At least one loss weight must be positive.")

        self.cross_entropy_weight = cross_entropy_weight
        self.dice_weight = dice_weight
        self.ignore_label = ignore_label
        self.register_buffer("class_weights", class_weights)

        self.dice_loss = MulticlassDiceLoss(
            ignore_label=ignore_label,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the weighted combined segmentation loss."""
        valid_pixels = targets != self.ignore_label

        if valid_pixels.any():
            cross_entropy = F.cross_entropy(
                logits,
                targets,
                weight=self.class_weights,
                ignore_index=self.ignore_label,
            )
        else:
            cross_entropy = logits.sum() * 0.0

        dice = self.dice_loss(logits, targets)

        return self.cross_entropy_weight * cross_entropy + self.dice_weight * dice

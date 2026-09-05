"""Combined semantic and boundary-aware segmentation loss."""

from dataclasses import dataclass

import torch
from torch import nn

from pancancer_nuclei.models.losses import CombinedSegmentationLoss


@dataclass(frozen=True)
class BoundaryAwareLossResult:
    """Total loss and its two task components."""

    total_loss: torch.Tensor
    semantic_loss: torch.Tensor
    spatial_loss: torch.Tensor


class BoundaryAwareSegmentationLoss(nn.Module):
    """Combine semantic-class and spatial-boundary objectives."""

    def __init__(
        self,
        semantic_class_weights: torch.Tensor | None = None,
        spatial_class_weights: torch.Tensor | None = None,
        semantic_task_weight: float = 1.0,
        spatial_task_weight: float = 1.0,
        ignore_label: int = 255,
    ) -> None:
        super().__init__()

        if semantic_task_weight < 0 or spatial_task_weight < 0:
            raise ValueError("Task weights cannot be negative.")

        if semantic_task_weight + spatial_task_weight == 0:
            raise ValueError("At least one task weight must be positive.")

        self.semantic_task_weight = semantic_task_weight
        self.spatial_task_weight = spatial_task_weight

        self.semantic_criterion = CombinedSegmentationLoss(
            ignore_label=ignore_label,
            class_weights=semantic_class_weights,
        )
        self.spatial_criterion = CombinedSegmentationLoss(
            ignore_label=ignore_label,
            class_weights=spatial_class_weights,
        )

    def forward(
        self,
        semantic_logits: torch.Tensor,
        spatial_logits: torch.Tensor,
        semantic_targets: torch.Tensor,
        spatial_targets: torch.Tensor,
    ) -> BoundaryAwareLossResult:
        """Calculate the total and individual task losses."""
        semantic_loss = self.semantic_criterion(
            semantic_logits,
            semantic_targets,
        )
        spatial_loss = self.spatial_criterion(
            spatial_logits,
            spatial_targets,
        )

        total_loss = (
            self.semantic_task_weight * semantic_loss + self.spatial_task_weight * spatial_loss
        )

        return BoundaryAwareLossResult(
            total_loss=total_loss,
            semantic_loss=semantic_loss,
            spatial_loss=spatial_loss,
        )

"""Tests for the boundary-aware multitask loss."""

import pytest
import torch

from pancancer_nuclei.models.boundary_loss import (
    BoundaryAwareSegmentationLoss,
)


def create_example_tensors() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Create small semantic and spatial prediction examples."""
    torch.manual_seed(42)

    semantic_logits = torch.randn(
        2,
        6,
        8,
        8,
        requires_grad=True,
    )
    spatial_logits = torch.randn(
        2,
        3,
        8,
        8,
        requires_grad=True,
    )
    semantic_targets = torch.randint(
        0,
        6,
        (2, 8, 8),
    )
    spatial_targets = torch.randint(
        0,
        3,
        (2, 8, 8),
    )

    return (
        semantic_logits,
        spatial_logits,
        semantic_targets,
        spatial_targets,
    )


def test_total_loss_combines_both_tasks() -> None:
    """Total loss should equal the weighted component sum."""
    tensors = create_example_tensors()
    criterion = BoundaryAwareSegmentationLoss(
        semantic_task_weight=1.0,
        spatial_task_weight=0.5,
    )

    result = criterion(*tensors)

    expected = result.semantic_loss + 0.5 * result.spatial_loss
    torch.testing.assert_close(result.total_loss, expected)


def test_both_predictions_receive_gradients() -> None:
    """Both model heads should receive training gradients."""
    (
        semantic_logits,
        spatial_logits,
        semantic_targets,
        spatial_targets,
    ) = create_example_tensors()

    criterion = BoundaryAwareSegmentationLoss()
    result = criterion(
        semantic_logits,
        spatial_logits,
        semantic_targets,
        spatial_targets,
    )
    result.total_loss.backward()

    assert semantic_logits.grad is not None
    assert spatial_logits.grad is not None
    assert torch.isfinite(semantic_logits.grad).all()
    assert torch.isfinite(spatial_logits.grad).all()


def test_class_weights_are_registered() -> None:
    """Semantic and spatial weights should be stored as buffers."""
    semantic_weights = torch.arange(1, 7, dtype=torch.float32)
    spatial_weights = torch.arange(1, 4, dtype=torch.float32)

    criterion = BoundaryAwareSegmentationLoss(
        semantic_class_weights=semantic_weights,
        spatial_class_weights=spatial_weights,
    )

    assert torch.equal(
        criterion.semantic_criterion.class_weights,
        semantic_weights,
    )
    assert torch.equal(
        criterion.spatial_criterion.class_weights,
        spatial_weights,
    )


def test_fully_ignored_targets_produce_zero_loss() -> None:
    """A fully ignored batch should remain finite and differentiable."""
    semantic_logits = torch.randn(
        1,
        6,
        4,
        4,
        requires_grad=True,
    )
    spatial_logits = torch.randn(
        1,
        3,
        4,
        4,
        requires_grad=True,
    )
    semantic_targets = torch.full(
        (1, 4, 4),
        255,
        dtype=torch.long,
    )
    spatial_targets = torch.full(
        (1, 4, 4),
        255,
        dtype=torch.long,
    )

    criterion = BoundaryAwareSegmentationLoss()
    result = criterion(
        semantic_logits,
        spatial_logits,
        semantic_targets,
        spatial_targets,
    )

    assert result.total_loss.item() == pytest.approx(0.0)
    result.total_loss.backward()

    assert semantic_logits.grad is not None
    assert spatial_logits.grad is not None


@pytest.mark.parametrize(
    ("semantic_weight", "spatial_weight"),
    [
        (-1.0, 1.0),
        (1.0, -1.0),
        (0.0, 0.0),
    ],
)
def test_invalid_task_weights_are_rejected(
    semantic_weight: float,
    spatial_weight: float,
) -> None:
    """Task weights must be non-negative and not both zero."""
    with pytest.raises(ValueError):
        BoundaryAwareSegmentationLoss(
            semantic_task_weight=semantic_weight,
            spatial_task_weight=spatial_weight,
        )

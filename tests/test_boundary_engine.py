"""Tests for boundary-aware batching, training, and validation."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from pancancer_nuclei.models.boundary_loss import (
    BoundaryAwareSegmentationLoss,
)
from pancancer_nuclei.training.boundary_engine import (
    boundary_segmentation_collate,
    evaluate_boundary_model,
    train_boundary_epoch,
)


class TinyBoundaryModel(nn.Module):
    """Small two-head model used only by unit tests."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.semantic_head = nn.Conv2d(8, 6, kernel_size=1)
        self.spatial_head = nn.Conv2d(8, 3, kernel_size=1)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return semantic and spatial logits."""
        features = torch.relu(self.features(images))
        return self.semantic_head(features), self.spatial_head(features)


def create_sample() -> dict[str, torch.Tensor]:
    """Create a small sample containing two separate nuclei."""
    instance_masks = torch.zeros((2, 16, 16), dtype=torch.bool)
    instance_masks[0, 2:7, 2:7] = True
    instance_masks[1, 9:14, 9:14] = True

    semantic_mask = torch.zeros((16, 16), dtype=torch.long)
    semantic_mask[instance_masks[0]] = 1
    semantic_mask[instance_masks[1]] = 2

    return {
        "image": torch.rand(3, 16, 16),
        "semantic_mask": semantic_mask,
        "instance_masks": instance_masks,
    }


def test_boundary_collate_creates_expected_batch() -> None:
    """Collation should stack inputs and derive spatial labels."""
    batch = boundary_segmentation_collate([create_sample(), create_sample()])

    assert batch["image"].shape == (2, 3, 16, 16)
    assert batch["semantic_mask"].shape == (2, 16, 16)
    assert batch["spatial_mask"].shape == (2, 16, 16)
    assert set(torch.unique(batch["spatial_mask"]).tolist()) == {0, 1, 2}


def test_boundary_collate_rejects_empty_batch() -> None:
    """An empty collection cannot form a training batch."""
    with pytest.raises(ValueError, match="empty batch"):
        boundary_segmentation_collate([])


def test_boundary_training_and_evaluation() -> None:
    """One optimization step and validation should produce finite metrics."""
    loader = DataLoader(
        [create_sample(), create_sample()],
        batch_size=2,
        collate_fn=boundary_segmentation_collate,
    )
    model = TinyBoundaryModel()
    criterion = BoundaryAwareSegmentationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    training = train_boundary_epoch(
        model=model,
        data_loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )
    validation = evaluate_boundary_model(
        model=model,
        data_loader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert training["loss"] > 0
    assert training["semantic_loss"] > 0
    assert training["spatial_loss"] > 0
    assert validation["loss"] > 0
    assert len(validation["semantic"]["confusion_matrix"]) == 6
    assert len(validation["spatial"]["confusion_matrix"]) == 3
    assert 0.0 <= validation["selection_score"] <= 1.0


def test_boundary_engine_rejects_empty_loader() -> None:
    """Training and evaluation should reject a loader with no batches."""
    model = TinyBoundaryModel()
    criterion = BoundaryAwareSegmentationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    empty_loader: list[dict[str, torch.Tensor]] = []

    with pytest.raises(ValueError, match="training data loader was empty"):
        train_boundary_epoch(
            model=model,
            data_loader=empty_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=torch.device("cpu"),
        )

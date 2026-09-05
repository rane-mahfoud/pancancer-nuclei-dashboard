import pytest
import torch

from pancancer_nuclei.models.losses import (
    CombinedSegmentationLoss,
    MulticlassDiceLoss,
)


def test_combined_loss_is_finite_and_has_gradients() -> None:
    logits = torch.randn(
        2,
        6,
        8,
        8,
        requires_grad=True,
    )
    targets = torch.randint(0, 6, (2, 8, 8))
    targets[0, 0, 0] = 255

    criterion = CombinedSegmentationLoss()
    loss = criterion(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_dice_loss_is_small_for_correct_predictions() -> None:
    logits = torch.full((1, 3, 2, 2), -10.0)
    targets = torch.tensor([[[0, 1], [2, 255]]])

    logits[0, 0, 0, 0] = 10.0
    logits[0, 1, 0, 1] = 10.0
    logits[0, 2, 1, 0] = 10.0

    criterion = MulticlassDiceLoss()
    loss = criterion(logits, targets)

    assert loss.item() < 1.0e-4


def test_combined_loss_handles_empty_patch() -> None:
    logits = torch.randn(
        1,
        6,
        8,
        8,
        requires_grad=True,
    )
    targets = torch.zeros((1, 8, 8), dtype=torch.long)

    criterion = CombinedSegmentationLoss()
    loss = criterion(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)


def test_combined_loss_rejects_zero_weights() -> None:
    with pytest.raises(ValueError, match="positive"):
        CombinedSegmentationLoss(
            cross_entropy_weight=0,
            dice_weight=0,
        )

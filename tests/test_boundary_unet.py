"""Tests for the boundary-aware U-Net."""

import pytest
import torch

from pancancer_nuclei.models.boundary_unet import BoundaryAwareUNet


def test_boundary_unet_output_shapes() -> None:
    """Both model heads should preserve image dimensions."""
    model = BoundaryAwareUNet(base_channels=8)
    images = torch.randn(2, 3, 64, 64)

    semantic_logits, spatial_logits = model(images)

    assert semantic_logits.shape == (2, 6, 64, 64)
    assert spatial_logits.shape == (2, 3, 64, 64)


def test_both_heads_and_encoder_receive_gradients() -> None:
    """Both tasks should train the shared feature extractor."""
    model = BoundaryAwareUNet(base_channels=8)
    images = torch.randn(1, 3, 32, 32)

    semantic_logits, spatial_logits = model(images)
    loss = semantic_logits.mean() + spatial_logits.mean()
    loss.backward()

    assert model.encoder_1.layers[0].weight.grad is not None
    assert model.semantic_classifier.weight.grad is not None
    assert model.spatial_classifier.weight.grad is not None


def test_output_heads_are_independent() -> None:
    """Semantic and spatial classifiers require different outputs."""
    model = BoundaryAwareUNet(
        number_of_semantic_classes=6,
        number_of_spatial_classes=3,
        base_channels=8,
    )

    assert model.semantic_classifier.out_channels == 6
    assert model.spatial_classifier.out_channels == 3
    assert model.semantic_classifier.weight.data_ptr() != model.spatial_classifier.weight.data_ptr()


def test_invalid_base_channels_are_rejected() -> None:
    """Group normalization requires channels divisible by eight."""
    with pytest.raises(
        ValueError,
        match="divisible by 8",
    ):
        BoundaryAwareUNet(base_channels=10)

import pytest
import torch

from pancancer_nuclei.models.unet import UNet


def test_unet_output_shape_and_gradients() -> None:
    model = UNet(
        input_channels=3,
        number_of_classes=6,
        base_channels=8,
    )
    images = torch.randn(2, 3, 64, 64)

    predictions = model(images)

    assert predictions.shape == (2, 6, 64, 64)
    assert torch.isfinite(predictions).all()

    loss = predictions.square().mean()
    loss.backward()

    gradient = model.classifier.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()


def test_unet_rejects_invalid_base_channels() -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        UNet(base_channels=10)

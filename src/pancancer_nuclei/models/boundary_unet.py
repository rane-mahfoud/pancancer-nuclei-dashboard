"""Boundary-aware U-Net for nucleus segmentation."""

import torch
from torch import nn

from pancancer_nuclei.models.unet import (
    DecoderBlock,
    DoubleConvolution,
)


class BoundaryAwareUNet(nn.Module):
    """Shared U-Net with semantic and spatial output heads."""

    def __init__(
        self,
        input_channels: int = 3,
        number_of_semantic_classes: int = 6,
        number_of_spatial_classes: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        if base_channels % 8 != 0:
            raise ValueError("base_channels must be divisible by 8.")

        self.encoder_1 = DoubleConvolution(
            input_channels,
            base_channels,
        )
        self.encoder_2 = DoubleConvolution(
            base_channels,
            base_channels * 2,
        )
        self.encoder_3 = DoubleConvolution(
            base_channels * 2,
            base_channels * 4,
        )
        self.encoder_4 = DoubleConvolution(
            base_channels * 4,
            base_channels * 8,
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConvolution(
            base_channels * 8,
            base_channels * 16,
        )

        self.decoder_4 = DecoderBlock(
            base_channels * 16,
            base_channels * 8,
            base_channels * 8,
        )
        self.decoder_3 = DecoderBlock(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
        )
        self.decoder_2 = DecoderBlock(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
        )
        self.decoder_1 = DecoderBlock(
            base_channels * 2,
            base_channels,
            base_channels,
        )

        self.semantic_classifier = nn.Conv2d(
            base_channels,
            number_of_semantic_classes,
            kernel_size=1,
        )
        self.spatial_classifier = nn.Conv2d(
            base_channels,
            number_of_spatial_classes,
            kernel_size=1,
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return semantic-class and spatial-boundary scores."""
        features_1 = self.encoder_1(images)
        features_2 = self.encoder_2(self.pool(features_1))
        features_3 = self.encoder_3(self.pool(features_2))
        features_4 = self.encoder_4(self.pool(features_3))

        bottleneck = self.bottleneck(self.pool(features_4))

        decoded_4 = self.decoder_4(bottleneck, features_4)
        decoded_3 = self.decoder_3(decoded_4, features_3)
        decoded_2 = self.decoder_2(decoded_3, features_2)
        decoded_1 = self.decoder_1(decoded_2, features_1)

        semantic_logits = self.semantic_classifier(decoded_1)
        spatial_logits = self.spatial_classifier(decoded_1)

        return semantic_logits, spatial_logits

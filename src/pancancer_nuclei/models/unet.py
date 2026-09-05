"""A compact U-Net for PanNuke semantic segmentation."""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConvolution(nn.Module):
    """Apply two convolution, normalization, and activation steps."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Process an image-feature tensor."""
        return self.layers(inputs)


class DecoderBlock(nn.Module):
    """Enlarge features and combine them with encoder features."""

    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            input_channels,
            output_channels,
            kernel_size=2,
            stride=2,
        )
        self.convolutions = DoubleConvolution(
            output_channels + skip_channels,
            output_channels,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        skip_features: torch.Tensor,
    ) -> torch.Tensor:
        """Upsample and combine features from both sides of the U-Net."""
        inputs = self.upsample(inputs)

        if inputs.shape[-2:] != skip_features.shape[-2:]:
            inputs = F.interpolate(
                inputs,
                size=skip_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        combined = torch.cat((skip_features, inputs), dim=1)
        return self.convolutions(combined)


class UNet(nn.Module):
    """Compact U-Net producing one score map per semantic class."""

    def __init__(
        self,
        input_channels: int = 3,
        number_of_classes: int = 6,
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

        self.classifier = nn.Conv2d(
            base_channels,
            number_of_classes,
            kernel_size=1,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class scores for every image pixel."""
        features_1 = self.encoder_1(images)
        features_2 = self.encoder_2(self.pool(features_1))
        features_3 = self.encoder_3(self.pool(features_2))
        features_4 = self.encoder_4(self.pool(features_3))

        bottleneck = self.bottleneck(self.pool(features_4))

        decoded_4 = self.decoder_4(bottleneck, features_4)
        decoded_3 = self.decoder_3(decoded_4, features_3)
        decoded_2 = self.decoder_2(decoded_3, features_2)
        decoded_1 = self.decoder_1(decoded_2, features_1)

        return self.classifier(decoded_1)

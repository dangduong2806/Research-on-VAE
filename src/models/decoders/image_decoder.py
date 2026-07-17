"""Configurable convolutional image decoder based on PyTorch-VAE."""
"""Image Decoder cho Vanilla VAE."""

from dataclasses import dataclass
from math import prod

import torch
from torch import Tensor, nn


@dataclass
class ImageDecoderConfig:
    latent_dim: int
    feature_shape: tuple[int, int, int]

    hidden_dims: tuple[int, ...] = (
        512,
        256,
        128,
        64,
        32,
    )

    out_channels: int = 3
    negative_slope: float = 0.01

    def __post_init__(self):
        if self.latent_dim <= 0:
            raise ValueError(
                "latent_dim phải lớn hơn 0."
            )

        if len(self.feature_shape) != 3:
            raise ValueError(
                "feature_shape phải là (C, H, W)."
            )

        if self.feature_shape[0] != self.hidden_dims[0]:
            raise ValueError(
                "Channel của feature_shape phải bằng "
                "hidden_dims[0]."
            )


class ImageDecoder(nn.Module):
    """Biến latent vector z thành ảnh tái tạo."""

    def __init__(
        self,
        config: ImageDecoderConfig,
    ):
        super().__init__()

        self.config = config

        flattened_dim = prod(
            config.feature_shape
        )

        # z [B, latent_dim]
        # → [B, C*H*W]
        self.fc = nn.Linear(
            config.latent_dim,
            flattened_dim,
        )

        self.blocks = nn.ModuleList()

        for in_channels, out_channels in zip(
            config.hidden_dims[:-1],
            config.hidden_dims[1:],
        ):
            self.blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(
                        out_channels
                    ),
                    nn.LeakyReLU(
                        config.negative_slope,
                        inplace=True,
                    ),
                )
            )

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                config.hidden_dims[-1],
                config.out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Tanh(),
        )

    def forward(
        self,
        z: Tensor,
    ) -> Tensor:
        """Decode latent vector thành ảnh."""

        if z.ndim != 2:
            raise ValueError(
                f"Expected z [B, latent_dim], "
                f"got {tuple(z.shape)}."
            )

        if z.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"Expected latent_dim="
                f"{self.config.latent_dim}, "
                f"got {z.shape[1]}."
            )

        x = self.fc(z)

        x = x.view(
            z.shape[0],
            *self.config.feature_shape,
        )

        for block in self.blocks:
            x = block(x)

        reconstruction = self.final_layer(x)

        return reconstruction
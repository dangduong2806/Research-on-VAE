"""Configurable convolutional image encoder based on PyTorch-VAE."""
from dataclasses import dataclass

import torch
from torch import nn, Tensor

from src.models.outputs import ImageEncoderOutput, EncoderStageOutput


@dataclass
class ImageEncoderConfig:
    in_channels: int = 3
    hidden_dims: tuple[int, ...] = (
        32,
        64,
        128,
        256,
        512,
    )
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1
    negative_slope: float = 0.01


class ImageEncoder(nn.Module):
    def __init__(self, config: ImageEncoderConfig):
        super().__init__()

        self.config = config
        self.blocks = nn.ModuleList()

        in_channels = config.in_channels

        for out_channels in config.hidden_dims:
            block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=config.kernel_size,
                    stride=config.stride,
                    padding=config.padding,
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(
                    config.negative_slope,
                    inplace=True,
                ),
            )

            self.blocks.append(block)
            in_channels = out_channels

    def forward(
        self,
        images: Tensor,
        return_intermediates: bool = False,
    ) -> ImageEncoderOutput:

        if images.ndim != 4:
            raise ValueError(
                f"Expected [B,C,H,W], got {images.shape}"
            )

        stages = []
        x = images

        for index, block in enumerate(self.blocks):
            x = block(x)

            if return_intermediates:
                stages.append(
                    EncoderStageOutput(
                        name=f"block_{index}",
                        tensor=x,
                    )
                )

        flattened = torch.flatten(
            x,
            start_dim=1,
        )

        return ImageEncoderOutput(
            input_shape=tuple(images.shape),
            feature_map=x,
            flattened=flattened,
            stages=tuple(stages),
        )

    def infer_flattened_dim(
        self,
        input_size: tuple[int, int],
    ) -> int:

        dummy = torch.zeros(
            1,
            self.config.in_channels,
            input_size[0],
            input_size[1],
        )

        was_training = self.training
        self.eval()

        with torch.no_grad():
            output = self(dummy)

        self.train(was_training)

        return output.flattened.shape[1]
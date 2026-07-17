"""Inspect model architecture and trace tensor shapes through each block."""
"""Kiểm tra luồng input/output của ImageEncoder."""

import argparse

import torch

from src.data.dataloader import build_dataloaders
from src.models.encoders.image_encoder import (
    ImageEncoder,
    ImageEncoderConfig,
)
from src.utils.config import load_project_config


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/image_128.yaml",
    )

    return parser.parse_args()


def build_encoder(project_config):
    """Tạo ImageEncoder từ phần model.encoder trong YAML."""

    encoder_cfg = project_config.model["encoder"]

    config = ImageEncoderConfig(
        in_channels=project_config.data["in_channels"],
        hidden_dims=tuple(
            encoder_cfg.get(
                "hidden_dims",
                [32, 64, 128, 256, 512],
            )
        ),
        kernel_size=encoder_cfg.get(
            "kernel_size",
            3,
        ),
        stride=encoder_cfg.get(
            "stride",
            2,
        ),
        padding=encoder_cfg.get(
            "padding",
            1,
        ),
        negative_slope=encoder_cfg.get(
            "negative_slope",
            0.01,
        ),
    )

    return ImageEncoder(config)


def count_parameters(model):
    """Đếm số parameter có thể được huấn luyện."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def print_encoder_output(output):
    """In shape sau từng bước của encoder."""

    print("\nInput")
    print(" shape:", output.input_shape)

    for stage in output.stages:
        tensor = stage.tensor

        print(f"\n{stage.name}")
        print(" shape:", tuple(tensor.shape))
        print(" min:", tensor.min().item())
        print(" max:", tensor.max().item())
        print(" mean:", tensor.mean().item())
        print(" std:", tensor.std().item())

    print("\nFinal feature map")
    print(" shape:", tuple(output.feature_map.shape))

    print("\nFlattened feature")
    print(" shape:", tuple(output.flattened.shape))


def main():
    args = parse_args()

    project_config = load_project_config(
        args.config
    )

    loaders = build_dataloaders(
        project_config.data,
        seed=project_config.seed,
    )

    encoder = build_encoder(
        project_config
    )

    batch = next(
        iter(loaders.train)
    )

    images = batch.images

    encoder.eval()

    with torch.no_grad():
        output = encoder(
            images,
            return_intermediates=True,
        )

    print("=" * 60)
    print("IMAGE ENCODER INSPECTION")
    print("=" * 60)

    print("\nEncoder architecture:")
    print(encoder)

    print(
        "\nTrainable parameters:",
        count_parameters(encoder),
    )

    print_encoder_output(output)


if __name__ == "__main__":
    main()
"""Phân tích khả năng reconstruction của Vanilla VAE."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from analysis.common import load_trained_model
from src.data.dataloader import build_dataloaders


def denormalize(
    images: torch.Tensor,
    normalization: str,
) -> torch.Tensor:
    """Đưa ảnh về miền [0, 1] để hiển thị."""

    if normalization == "minus_one_to_one":
        images = (images + 1.0) / 2.0

    elif normalization == "zero_to_one":
        pass

    elif normalization == "imagenet":
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=images.device,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=images.device,
        ).view(1, 3, 1, 1)

        images = images * std + mean

    else:
        raise ValueError(
            f"Chưa hỗ trợ normalization: {normalization}"
        )

    return images.clamp(0.0, 1.0)


def save_comparison(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    output_path: str,
) -> None:
    """Lưu hai hàng: ảnh gốc và ảnh tái tạo."""

    num_images = originals.shape[0]

    figure, axes = plt.subplots(
        2,
        num_images,
        figsize=(2 * num_images, 4),
        squeeze=False,
    )

    for index in range(num_images):
        original = originals[index].cpu()
        reconstruction = reconstructions[index].cpu()

        if original.shape[0] == 1:
            axes[0, index].imshow(
                original[0],
                cmap="gray",
            )
            axes[1, index].imshow(
                reconstruction[0],
                cmap="gray",
            )
        else:
            axes[0, index].imshow(
                original.permute(1, 2, 0)
            )
            axes[1, index].imshow(
                reconstruction.permute(1, 2, 0)
            )

        axes[0, index].axis("off")
        axes[1, index].axis("off")

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstruction")

    figure.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def analyze_reconstruction(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    num_images: int,
) -> None:
    config, model, _, device = load_trained_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )

    loaders = build_dataloaders(
        config.data,
        seed=config.seed,
    )

    mse_sum = 0.0
    sample_count = 0

    displayed_originals = []
    displayed_reconstructions = []

    with torch.no_grad():
        for batch in loaders.test:
            images = batch.images.to(
                device,
                non_blocking=True,
            )

            reconstructions = model.reconstruct(
                images
            )

            mse_per_sample = F.mse_loss(
                reconstructions,
                images,
                reduction="none",
            )

            mse_per_sample = (
                mse_per_sample
                .flatten(start_dim=1)
                .mean(dim=1)
            )

            mse_sum += mse_per_sample.sum().item()
            sample_count += images.shape[0]

            remaining = (
                num_images
                - len(displayed_originals)
            )

            if remaining > 0:
                displayed_originals.extend(
                    images[:remaining].cpu()
                )

                displayed_reconstructions.extend(
                    reconstructions[:remaining].cpu()
                )

    if sample_count == 0:
        raise RuntimeError(
            "Test dataloader không có dữ liệu."
        )

    test_mse = mse_sum / sample_count

    originals = torch.stack(
        displayed_originals
    )

    reconstructions = torch.stack(
        displayed_reconstructions
    )

    normalization = config.data.get(
        "normalization",
        "minus_one_to_one",
    )

    originals = denormalize(
        originals,
        normalization,
    )

    reconstructions = denormalize(
        reconstructions,
        normalization,
    )

    save_comparison(
        originals=originals,
        reconstructions=reconstructions,
        output_path=output_path,
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test samples: {sample_count}")
    print(f"Test reconstruction MSE: {test_mse:.6f}")
    print(f"Saved comparison: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phân tích reconstruction của Vanilla VAE"
    )

    parser.add_argument(
        "--config",
        default="configs/image_128.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        default="best.pt",
    )

    parser.add_argument(
        "--output",
        default="outputs/analysis/reconstruction.png",
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    analyze_reconstruction(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_images=args.num_images,
    )


if __name__ == "__main__":
    main()
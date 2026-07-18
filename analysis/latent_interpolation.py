"""Nội suy giữa hai ảnh trong latent space."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from analysis.common import load_trained_model
from analysis.reconstruction import denormalize
from src.data.dataloader import build_dataloaders


def get_test_images(
    dataloader,
    index_a: int,
    index_b: int,
) -> torch.Tensor:
    """Lấy hai ảnh theo index trong test set."""

    target_index = max(index_a, index_b)
    collected = []

    for batch in dataloader:
        collected.append(batch.images)

        images = torch.cat(collected)

        if len(images) > target_index:
            return torch.stack(
                [
                    images[index_a],
                    images[index_b],
                ]
            )

    raise IndexError(
        f"Test set không có index {target_index}."
    )


def interpolate_latents(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Nội suy tuyến tính từ z_a đến z_b."""

    alphas = torch.linspace(
        0.0,
        1.0,
        steps,
        device=z_a.device,
    )

    latents = [
        (1.0 - alpha) * z_a
        + alpha * z_b
        for alpha in alphas
    ]

    return torch.stack(latents)


def save_interpolation(
    original_images: torch.Tensor,
    images: torch.Tensor,
    output_path: str,
) -> None:
    """Save original endpoints and the decoded interpolation sequence."""

    original_images = original_images.cpu()
    images = images.cpu()
    steps = len(images)
    panel_count = steps + 2

    figure, axes = plt.subplots(
        1,
        panel_count,
        figsize=(2 * panel_count, 2.8),
        squeeze=False,
    )

    displayed_images = [
        original_images[0],
        *images,
        original_images[1],
    ]
    titles = [
        "Original A",
        *[
            f"α={index / (steps - 1):.2f}"
            for index in range(steps)
        ],
        "Original B",
    ]

    for axis, image, title in zip(
        axes[0],
        displayed_images,
        titles,
    ):

        if image.shape[0] == 1:
            axis.imshow(
                image[0],
                cmap="gray",
            )
        else:
            axis.imshow(
                image.permute(1, 2, 0)
            )

        axis.set_title(title)
        axis.axis("off")

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


def analyze_interpolation(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    index_a: int,
    index_b: int,
    steps: int,
) -> None:
    if steps < 2:
        raise ValueError(
            "steps phải lớn hơn hoặc bằng 2."
        )

    config, model, _, device = (
        load_trained_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
        )
    )

    loaders = build_dataloaders(
        config.data,
        seed=config.seed,
    )

    images = get_test_images(
        dataloader=loaders.test,
        index_a=index_a,
        index_b=index_b,
    ).to(device)

    with torch.no_grad():
        gaussian = model.encode(images)

        z_a = gaussian.mu[0]
        z_b = gaussian.mu[1]

        latents = interpolate_latents(
            z_a=z_a,
            z_b=z_b,
            steps=steps,
        )

        interpolated_images = model.decode(
            latents
        )

    normalization = config.data.get(
        "normalization",
        "minus_one_to_one",
    )

    interpolated_images = denormalize(
        interpolated_images,
        normalization,
    )

    original_images = denormalize(
        images,
        normalization,
    )

    save_interpolation(
        original_images=original_images,
        images=interpolated_images,
        output_path=output_path,
    )

    latent_distance = torch.norm(
        z_a - z_b,
        p=2,
    ).item()

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Image A index: {index_a}")
    print(f"Image B index: {index_b}")
    print(f"Latent distance: {latent_distance:.4f}")
    print(f"Interpolation steps: {steps}")
    print(f"Saved interpolation: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Latent interpolation cho Vanilla VAE"
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
        default=(
            "outputs/analysis/"
            "latent_interpolation.png"
        ),
    )

    parser.add_argument(
        "--index-a",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--index-b",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=11,
    )

    args = parser.parse_args()

    analyze_interpolation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        index_a=args.index_a,
        index_b=args.index_b,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()

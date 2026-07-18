"""Sinh ảnh ngẫu nhiên từ latent prior N(0, I)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from analysis.common import load_trained_model
from analysis.reconstruction import denormalize


def save_samples(
    images: torch.Tensor,
    output_path: str,
    columns: int = 8,
) -> None:
    """Lưu các ảnh sinh ra thành một lưới."""

    images = images.cpu()

    num_images = images.shape[0]
    rows = (num_images + columns - 1) // columns

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2 * columns, 2 * rows),
        squeeze=False,
    )

    for index, axis in enumerate(
        axes.flatten()
    ):
        axis.axis("off")

        if index >= num_images:
            continue

        image = images[index]

        if image.shape[0] == 1:
            axis.imshow(
                image[0],
                cmap="gray",
            )
        else:
            axis.imshow(
                image.permute(1, 2, 0)
            )

        axis.set_title(
            f"Sample {index + 1}"
        )

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


def analyze_random_sampling(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    num_samples: int,
    seed: int,
) -> None:
    config, model, _, device = (
        load_trained_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
        )
    )

    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        z = torch.randn(
            num_samples,
            model.config.latent_dim,
            device=device,
        )

        generated_images = model.decode(z)

    normalization = config.data.get(
        "normalization",
        "minus_one_to_one",
    )

    generated_images = denormalize(
        generated_images,
        normalization,
    )

    save_samples(
        images=generated_images,
        output_path=output_path,
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Latent shape: {tuple(z.shape)}")
    print(
        f"Generated shape: "
        f"{tuple(generated_images.shape)}"
    )
    print(f"Saved samples: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Random sampling với Vanilla VAE"
        )
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
            "random_sampling.png"
        ),
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    analyze_random_sampling(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_samples=args.num_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
"""Phân tích phân bố latent của Vanilla VAE."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from analysis.common import load_trained_model
from src.data.dataloader import build_dataloaders


def collect_latents(
    model,
    dataloader,
    device: torch.device,
):
    """Thu thập mu, std và label trên test set."""

    mu_batches = []
    std_batches = []
    label_batches = []
    labels_available = True

    with torch.no_grad():
        for batch in dataloader:
            images = batch.images.to(
                device,
                non_blocking=True,
            )

            gaussian = model.encode(images)

            mu_batches.append(
                gaussian.mu.cpu()
            )

            std_batches.append(
                torch.exp(
                    0.5 * gaussian.log_var
                ).cpu()
            )

            labels = getattr(
                batch,
                "labels",
                None,
            )

            if labels is None:
                labels_available = False
            elif labels_available:
                label_batches.append(
                    torch.as_tensor(labels).cpu()
                )

    if not mu_batches:
        raise RuntimeError(
            "Test dataloader không có dữ liệu."
        )

    mu = torch.cat(mu_batches)
    std = torch.cat(std_batches)

    labels = None

    if labels_available and label_batches:
        labels = torch.cat(label_batches)

    return mu, std, labels


def project_pca_2d(
    latents: torch.Tensor,
) -> torch.Tensor:
    """Chiếu latent vector xuống hai chiều bằng PCA."""

    if latents.shape[0] < 2:
        raise ValueError(
            "Cần ít nhất hai sample để chạy PCA."
        )

    if latents.shape[1] < 2:
        raise ValueError(
            "latent_dim phải lớn hơn hoặc bằng 2."
        )

    centered = (
        latents
        - latents.mean(
            dim=0,
            keepdim=True,
        )
    )

    _, _, components = torch.pca_lowrank(
        centered,
        q=2,
    )

    return centered @ components[:, :2]


def save_histogram(
    values: torch.Tensor,
    title: str,
    x_label: str,
    output_path: Path,
) -> None:
    """Lưu histogram của một tensor."""

    figure = plt.figure(
        figsize=(7, 5)
    )

    plt.hist(
        values.flatten().numpy(),
        bins=60,
    )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_pca_plot(
    points: torch.Tensor,
    labels: torch.Tensor | None,
    output_path: Path,
) -> None:
    """Lưu biểu đồ PCA 2D."""

    figure = plt.figure(
        figsize=(7, 6)
    )

    if labels is None:
        plt.scatter(
            points[:, 0],
            points[:, 1],
            s=14,
            alpha=0.7,
        )
    else:
        scatter = plt.scatter(
            points[:, 0],
            points[:, 1],
            c=labels,
            s=14,
            alpha=0.7,
        )

        plt.colorbar(
            scatter,
            label="Class",
        )

    plt.title("PCA of latent means")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def calculate_statistics(
    mu: torch.Tensor,
    std: torch.Tensor,
) -> dict:
    """Tính thống kê tổng quát của latent space."""

    dimension_means = mu.mean(dim=0)
    dimension_stds = mu.std(
        dim=0,
        unbiased=False,
    )

    return {
        "samples": mu.shape[0],
        "latent_dim": mu.shape[1],
        "mu_global_mean": mu.mean().item(),
        "mu_global_std": mu.std(
            unbiased=False
        ).item(),
        "mean_absolute_dimension_mean": (
            dimension_means.abs().mean().item()
        ),
        "mean_dimension_std": (
            dimension_stds.mean().item()
        ),
        "posterior_std_mean": std.mean().item(),
        "posterior_std_min": std.min().item(),
        "posterior_std_max": std.max().item(),
    }


def analyze_latent_distribution(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
) -> None:
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

    mu, std, labels = collect_latents(
        model=model,
        dataloader=loaders.test,
        device=device,
    )

    pca_points = project_pca_2d(mu)

    output = Path(output_dir)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_histogram(
        values=mu,
        title="Distribution of latent means",
        x_label="mu",
        output_path=output / "mu_histogram.png",
    )

    save_histogram(
        values=std,
        title="Distribution of posterior standard deviations",
        x_label="std",
        output_path=output / "std_histogram.png",
    )

    save_pca_plot(
        points=pca_points,
        labels=labels,
        output_path=output / "latent_pca.png",
    )

    statistics = calculate_statistics(
        mu,
        std,
    )

    statistics_path = (
        output / "latent_statistics.json"
    )

    statistics_path.write_text(
        json.dumps(
            statistics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples: {statistics['samples']}")
    print(f"Latent dim: {statistics['latent_dim']}")

    print(
        "Global mu mean: "
        f"{statistics['mu_global_mean']:.4f}"
    )

    print(
        "Global mu std: "
        f"{statistics['mu_global_std']:.4f}"
    )

    print(
        "Posterior std mean: "
        f"{statistics['posterior_std_mean']:.4f}"
    )

    print(f"Saved results: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phân tích latent distribution "
            "của Vanilla VAE"
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
        "--output-dir",
        default=(
            "outputs/analysis/"
            "latent_distribution"
        ),
    )

    args = parser.parse_args()

    analyze_latent_distribution(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
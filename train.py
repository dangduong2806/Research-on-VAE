"""Train the modular Vanilla VAE using an explicit PyTorch loop."""
"""Chạy huấn luyện Vanilla VAE."""

import argparse

import torch

from src.data.dataloader import build_dataloaders
from src.engine.evaluator import evaluate
from src.engine.trainer import train_one_epoch
from src.losses.kl_divergence import KLDivergenceLoss
from src.losses.reconstruction_loss import ReconstructionLoss
from src.models.vanilla_vae import (
    VanillaVAE,
    VanillaVAEConfig,
)
from src.utils.config import load_project_config


def main(config_path: str) -> None:
    config = load_project_config(
        config_path
    )

    torch.manual_seed(config.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            config.seed
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    loaders = build_dataloaders(
        config.data,
        seed=config.seed,
    )

    model = VanillaVAE(
        VanillaVAEConfig(
            input_size=tuple(
                config.data["input_size"]
            ),
            in_channels=config.data[
                "in_channels"
            ],
            latent_dim=config.model[
                "latent_dim"
            ],
            hidden_dims=tuple(
                config.model[
                    "encoder"
                ]["hidden_dims"]
            ),
        )
    ).to(device)

    reconstruction_loss_fn = (
        ReconstructionLoss(
            loss_type=config.loss.get(
                "reconstruction",
                "mse",
            ),
            reduction="mean",
        )
    )

    kl_loss_fn = KLDivergenceLoss(
        reduction="mean"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.get(
            "learning_rate",
            1e-3,
        ),
        weight_decay=config.training.get(
            "weight_decay",
            0.0,
        ),
    )

    epochs = config.training.get(
        "epochs",
        10,
    )

    beta = config.loss.get(
        "beta",
        1.0,
    )

    print(
        f"Epochs: {epochs} | "
        f"Beta: {beta} | "
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=loaders.train,
            optimizer=optimizer,
            reconstruction_loss_fn=(
                reconstruction_loss_fn
            ),
            kl_loss_fn=kl_loss_fn,
            device=device,
            beta=beta,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=loaders.val,
            reconstruction_loss_fn=(
                reconstruction_loss_fn
            ),
            kl_loss_fn=kl_loss_fn,
            device=device,
            beta=beta,
        )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train={train_metrics['total']:.4f} "
            f"(recon={train_metrics['reconstruction']:.4f}, "
            f"kl={train_metrics['kl']:.4f}) | "
            f"val={val_metrics['total']:.4f} "
            f"(recon={val_metrics['reconstruction']:.4f}, "
            f"kl={val_metrics['kl']:.4f})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Vanilla VAE"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/image_128.yaml",
        help="Đường dẫn tới file YAML.",
    )

    args = parser.parse_args()

    main(args.config)
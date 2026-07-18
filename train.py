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
from src.utils.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from src.utils.config import load_project_config

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

def plot_losses(history, output_path):
    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.plot(
        epochs,
        history["train_loss"],
        label="Training loss",
    )

    axis.plot(
        epochs,
        history["val_loss"],
        label="Validation loss",
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title(
        "Training and validation loss"
    )
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=150,
    )
    plt.close(figure)

def main(
    config_path: str,
    resume_path: str | None = None,
) -> None:
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

    output_dir = Path(
        config.logging.get(
            "output_dir",
            "outputs",
        )
    )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_every = config.training.get(
        "save_every",
        5,
    )
    patience = config.training.get(
        "early_stopping_patience",
        10,
    )
    min_delta = config.training.get(
        "early_stopping_min_delta",
        0.0,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {
        "train_loss": [],
        "val_loss": [],
    }

    if resume_path is not None:
        checkpoint = load_checkpoint(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get(
            "best_val_loss",
            float("inf"),
        )
        epochs_without_improvement = (
            checkpoint.get(
                "epochs_without_improvement",
                0,
            )
        )
        history = checkpoint.get(
            "history",
            history,
        )
        print(
            f"Resumed from {resume_path} | "
            f"next epoch: {start_epoch}"
        )

    print(
        f"Epochs: {epochs} | "
        f"Beta: {beta} | "
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    for epoch in range(
        start_epoch,
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

        train_loss = float(
            train_metrics["total"]
        )
        val_loss = float(
            val_metrics["total"]
        )
        history["train_loss"].append(
            train_loss
        )
        history["val_loss"].append(
            val_loss
        )

        improved = (
            val_loss
            < best_val_loss - min_delta
        )
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint_state = {
            "model": model,
            "optimizer": optimizer,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": (
                epochs_without_improvement
            ),
            "history": history,
        }

        save_checkpoint(
            checkpoint_dir / "latest.pt",
            **checkpoint_state,
        )
        if improved:
            save_checkpoint(
                checkpoint_dir / "best.pt",
                **checkpoint_state,
            )
        if (
            save_every > 0
            and epoch % save_every == 0
        ):
            save_checkpoint(
                checkpoint_dir
                / f"epoch_{epoch:03d}.pt",
                **checkpoint_state,
            )

        plot_losses(
            history,
            output_dir / "losses.png",
        )

        if (
            patience > 0
            and epochs_without_improvement
            >= patience
        ):
            print(
                f"Early stopping at epoch "
                f"{epoch}: validation loss "
                f"did not improve for "
                f"{patience} epochs."
            )
            break


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

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint from which "
            "training resumes."
        ),
    )

    args = parser.parse_args()

    main(
        args.config,
        args.resume,
    )

"""Explicit PyTorch training loop for the Vanilla VAE."""
"""Training engine cho Vanilla VAE."""

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.losses.kl_divergence import KLDivergenceLoss
from src.losses.reconstruction_loss import ReconstructionLoss
from src.models.vanilla_vae import VanillaVAE


def train_one_epoch(
    model: VanillaVAE,
    dataloader: DataLoader,
    optimizer: Optimizer,
    reconstruction_loss_fn: ReconstructionLoss,
    kl_loss_fn: KLDivergenceLoss,
    device: torch.device,
    beta: float,
) -> dict[str, float]:
    """Huấn luyện model trong một epoch."""

    model.train()

    total_sum = 0.0
    reconstruction_sum = 0.0
    kl_sum = 0.0
    sample_count = 0

    for batch in dataloader:
        images = batch.images.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            images,
            sample=True,
        )

        reconstruction_loss = (
            reconstruction_loss_fn(
                output.reconstruction,
                images,
            )
        )

        kl_loss = kl_loss_fn(
            output.mu,
            output.log_var,
        )

        total_loss = (
            reconstruction_loss
            + beta * kl_loss
        )

        total_loss.backward()
        optimizer.step()

        batch_size = images.shape[0]

        total_sum += (
            total_loss.item()
            * batch_size
        )

        reconstruction_sum += (
            reconstruction_loss.item()
            * batch_size
        )

        kl_sum += (
            kl_loss.item()
            * batch_size
        )

        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError(
            "Train dataloader không có dữ liệu."
        )

    return {
        "total": total_sum / sample_count,
        "reconstruction": (
            reconstruction_sum / sample_count
        ),
        "kl": kl_sum / sample_count,
        "samples": sample_count,
    }
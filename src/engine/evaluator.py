"""Evaluation, reconstruction, and sampling utilities."""
"""Evaluation engine cho Vanilla VAE."""

import torch
from torch.utils.data import DataLoader

from src.losses.kl_divergence import KLDivergenceLoss
from src.losses.reconstruction_loss import ReconstructionLoss
from src.models.vanilla_vae import VanillaVAE


def evaluate(
    model: VanillaVAE,
    dataloader: DataLoader,
    reconstruction_loss_fn: ReconstructionLoss,
    kl_loss_fn: KLDivergenceLoss,
    device: torch.device,
    beta: float,
) -> dict[str, float | int]:
    """Đánh giá model trong một epoch."""

    model.eval()

    total_sum = 0.0
    reconstruction_sum = 0.0
    kl_sum = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch.images.to(
                device,
                non_blocking=True,
            )

            output = model(
                images,
                sample=False,
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
            "Evaluation dataloader không có dữ liệu."
        )

    return {
        "total": total_sum / sample_count,
        "reconstruction": (
            reconstruction_sum / sample_count
        ),
        "kl": kl_sum / sample_count,
        "samples": sample_count,
    }
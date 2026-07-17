"""KL divergence between approximate posterior and unit Gaussian prior."""
"""KL divergence loss cho Gaussian latent space."""

import torch
from torch import Tensor, nn


class KLDivergenceLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
    ):
        super().__init__()

        if reduction not in {
            "mean",
            "sum",
            "none",
        }:
            raise ValueError(
                "reduction phải là 'mean', 'sum' hoặc 'none'."
            )

        self.reduction = reduction

    def forward(
        self,
        mu: Tensor,
        log_var: Tensor,
    ) -> Tensor:
        """Tính KL giữa N(mu, variance) và N(0, I)."""

        if mu.shape != log_var.shape:
            raise ValueError(
                "mu và log_var phải cùng shape, "
                f"nhận được {tuple(mu.shape)} và "
                f"{tuple(log_var.shape)}."
            )

        if mu.ndim != 2:
            raise ValueError(
                "mu và log_var phải có shape "
                f"[B, latent_dim], nhận được {tuple(mu.shape)}."
            )

        # Cộng KL trên toàn bộ latent dimensions.
        kl_per_sample = -0.5 * torch.sum(
            1
            + log_var
            - mu.pow(2)
            - log_var.exp(),
            dim=1,
        )

        if self.reduction == "mean":
            return kl_per_sample.mean()

        if self.reduction == "sum":
            return kl_per_sample.sum()

        return kl_per_sample
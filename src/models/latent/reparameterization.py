"""Reparameterization trick cho VAE."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class ReparameterizationOutput:
    std: Tensor
    epsilon: Tensor
    z: Tensor


class Reparameterization(nn.Module):
    def forward(
        self,
        mu: Tensor,
        log_var: Tensor,
        sample: bool = True,
    ) -> ReparameterizationOutput:
        """Tạo latent vector z từ mu và log_var."""

        if mu.shape != log_var.shape:
            raise ValueError(
                f"mu và log_var phải cùng shape, "
                f"nhận được {mu.shape} và {log_var.shape}."
            )

        if mu.ndim != 2:
            raise ValueError(
                f"Expected [B, latent_dim], got {mu.shape}."
            )

        std = torch.exp(
            0.5 * log_var
        )

        if sample:
            epsilon = torch.randn_like(std)
            z = mu + std * epsilon
        else:
            epsilon = torch.zeros_like(std)
            z = mu

        return ReparameterizationOutput(
            std=std,
            epsilon=epsilon,
            z=z,
        )
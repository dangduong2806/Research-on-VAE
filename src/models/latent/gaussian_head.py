"""Projection from encoder features to Gaussian mu and log-variance."""
"""Gaussian latent head cho Vanilla VAE."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class GaussianOutput:
    """Kết quả phân phối Gaussian trong latent space."""

    mu: Tensor
    log_var: Tensor


class GaussianHead(nn.Module):
    """Chuyển encoder feature thành mu và log_var."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim phải lớn hơn 0."
            )

        if latent_dim <= 0:
            raise ValueError(
                "latent_dim phải lớn hơn 0."
            )

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.fc_mu = nn.Linear(
            input_dim,
            latent_dim,
        )

        self.fc_log_var = nn.Linear(
            input_dim,
            latent_dim,
        )

    def forward(
        self,
        features: Tensor,
    ) -> GaussianOutput:
        """Tạo tham số Gaussian từ encoder features.

        Input:
            features: [B, F]

        Output:
            mu:      [B, latent_dim]
            log_var: [B, latent_dim]
        """

        if features.ndim != 2:
            raise ValueError(
                "GaussianHead yêu cầu input [B,F], "
                f"nhận được {tuple(features.shape)}."
            )

        if features.shape[1] != self.input_dim:
            raise ValueError(
                "Feature dimension không khớp. "
                f"Expected {self.input_dim}, "
                f"got {features.shape[1]}."
            )

        return GaussianOutput(
            mu=self.fc_mu(features),
            log_var=self.fc_log_var(features),
        )
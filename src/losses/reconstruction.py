"""Image reconstruction losses such as MSE, L1, and BCE."""
"""Reconstruction loss cho VAE."""

from torch import Tensor, nn
from torch.nn import functional as F


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        loss_type: str = "mse",
        reduction: str = "mean",
    ):
        super().__init__()

        if loss_type not in {"mse", "l1"}:
            raise ValueError(
                "loss_type phải là 'mse' hoặc 'l1'."
            )

        if reduction not in {
            "mean",
            "sum",
            "none",
        }:
            raise ValueError(
                "reduction phải là 'mean', 'sum' hoặc 'none'."
            )

        self.loss_type = loss_type
        self.reduction = reduction

    def forward(
        self,
        reconstruction: Tensor,
        target: Tensor,
    ) -> Tensor:
        """So sánh ảnh tái tạo với ảnh gốc."""

        if reconstruction.shape != target.shape:
            raise ValueError(
                "Reconstruction và target phải cùng shape, "
                f"nhận được {tuple(reconstruction.shape)} "
                f"và {tuple(target.shape)}."
            )

        if self.loss_type == "mse":
            return F.mse_loss(
                reconstruction,
                target,
                reduction=self.reduction,
            )

        return F.l1_loss(
            reconstruction,
            target,
            reduction=self.reduction,
        )
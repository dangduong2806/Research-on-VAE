"""Loss functions for Vanilla VAE training."""
from src.losses.reconstruction_loss import (
    ReconstructionLoss,
)

from src.losses.kl_divergence import (
    KLDivergenceLoss,
)

__all__ = [
    "ReconstructionLoss",
    "KLDivergenceLoss",
]
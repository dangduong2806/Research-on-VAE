"""Latent-space modules."""

from src.models.latent.gaussian_head import (
    GaussianHead,
    GaussianOutput,
)

from src.models.latent.reparameterization import (
    Reparameterization,
    ReparameterizationOutput,
)

__all__ = [
    "GaussianHead",
    "GaussianOutput",
    "Reparameterization",
    "ReparameterizationOutput",
]
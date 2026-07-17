"""Model Vanilla VAE hoàn chỉnh."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.models.encoders.image_encoder import (
    ImageEncoder,
    ImageEncoderConfig,
)
from src.models.decoders.image_decoder import (
    ImageDecoder,
    ImageDecoderConfig,
)
from src.models.latent.gaussian_head import (
    GaussianHead,
    GaussianOutput,
)
from src.models.latent.reparameterization import (
    Reparameterization,
)


@dataclass
class VanillaVAEConfig:
    input_size: tuple[int, int]
    in_channels: int = 3
    latent_dim: int = 128

    hidden_dims: tuple[int, ...] = (
        32,
        64,
        128,
        256,
        512,
    )


@dataclass
class VanillaVAEOutput:
    reconstruction: Tensor
    mu: Tensor
    log_var: Tensor
    std: Tensor
    epsilon: Tensor
    z: Tensor


class VanillaVAE(nn.Module):
    def __init__(
        self,
        config: VanillaVAEConfig,
    ):
        super().__init__()

        self.config = config

        self.encoder = ImageEncoder(
            ImageEncoderConfig(
                in_channels=config.in_channels,
                hidden_dims=config.hidden_dims,
            )
        )

        feature_shape = self._infer_feature_shape()

        flattened_dim = (
            feature_shape[0]
            * feature_shape[1]
            * feature_shape[2]
        )

        self.gaussian_head = GaussianHead(
            input_dim=flattened_dim,
            latent_dim=config.latent_dim,
        )

        self.reparameterization = (
            Reparameterization()
        )

        self.decoder = ImageDecoder(
            ImageDecoderConfig(
                latent_dim=config.latent_dim,
                feature_shape=feature_shape,
                hidden_dims=tuple(
                    reversed(
                        config.hidden_dims
                    )
                ),
                out_channels=config.in_channels,
            )
        )

    def _infer_feature_shape(
        self,
    ) -> tuple[int, int, int]:
        """Tự xác định shape cuối encoder."""

        dummy = torch.zeros(
            2,
            self.config.in_channels,
            *self.config.input_size,
        )

        was_training = self.encoder.training
        self.encoder.eval()

        with torch.no_grad():
            output = self.encoder(dummy)

        self.encoder.train(was_training)

        return tuple(
            output.feature_map.shape[1:]
        )

    def encode(
        self,
        images: Tensor,
    ) -> GaussianOutput:
        """Ảnh → mu và log_var."""

        encoder_output = self.encoder(
            images
        )

        return self.gaussian_head(
            encoder_output.flattened
        )

    def decode(
        self,
        z: Tensor,
    ) -> Tensor:
        """Latent vector → ảnh."""

        return self.decoder(z)

    def forward(
        self,
        images: Tensor,
        sample: bool = True,
    ) -> VanillaVAEOutput:
        """Forward toàn bộ Vanilla VAE."""

        gaussian = self.encode(images)

        latent = self.reparameterization(
            gaussian.mu,
            gaussian.log_var,
            sample=sample,
        )

        reconstruction = self.decode(
            latent.z
        )

        if reconstruction.shape != images.shape:
            raise RuntimeError(
                "Reconstruction shape không khớp input: "
                f"{tuple(reconstruction.shape)} "
                f"vs {tuple(images.shape)}."
            )

        return VanillaVAEOutput(
            reconstruction=reconstruction,
            mu=gaussian.mu,
            log_var=gaussian.log_var,
            std=latent.std,
            epsilon=latent.epsilon,
            z=latent.z,
        )

    def reconstruct(
        self,
        images: Tensor,
    ) -> Tensor:
        """Tái tạo ảnh theo chế độ deterministic."""

        return self(
            images,
            sample=False,
        ).reconstruction

    def generate(
        self,
        num_samples: int,
        device: torch.device | str,
    ) -> Tensor:
        """Sinh ảnh mới từ prior N(0, I)."""

        z = torch.randn(
            num_samples,
            self.config.latent_dim,
            device=device,
        )

        return self.decode(z)
"""Tests for decoder output shapes and ranges."""
"""Test cho ImageDecoder."""

import pytest
import torch
from torch import nn

from src.models.decoders.image_decoder import (
    ImageDecoder,
    ImageDecoderConfig,
)


def build_test_decoder() -> ImageDecoder:
    """Decoder nhỏ, hỗ trợ ảnh đầu ra 32×48."""

    config = ImageDecoderConfig(
        latent_dim=16,
        feature_shape=(64, 4, 6),
        hidden_dims=(
            64,
            32,
            16,
        ),
        out_channels=3,
    )

    return ImageDecoder(config)


def test_decoder_architecture():
    """Kiểm tra các thành phần chính của decoder."""

    decoder = build_test_decoder()

    assert isinstance(
        decoder.fc,
        nn.Linear,
    )

    assert decoder.fc.in_features == 16
    assert decoder.fc.out_features == 64 * 4 * 6

    assert len(decoder.blocks) == 2

    first_block = decoder.blocks[0]

    assert isinstance(
        first_block[0],
        nn.ConvTranspose2d,
    )
    assert isinstance(
        first_block[1],
        nn.BatchNorm2d,
    )
    assert isinstance(
        first_block[2],
        nn.LeakyReLU,
    )

    assert isinstance(
        decoder.final_layer[0],
        nn.ConvTranspose2d,
    )
    assert isinstance(
        decoder.final_layer[1],
        nn.Tanh,
    )


def test_decoder_output_shape():
    """Kiểm tra shape ảnh tái tạo."""

    decoder = build_test_decoder()

    z = torch.randn(
        8,
        16,
    )

    reconstruction = decoder(z)

    assert reconstruction.shape == (
        8,
        3,
        32,
        48,
    )


def test_decoder_output_range():
    """Tanh phải giới hạn output trong [-1, 1]."""

    decoder = build_test_decoder()

    z = torch.randn(
        8,
        16,
    )

    reconstruction = decoder(z)

    assert reconstruction.min().item() >= -1.0
    assert reconstruction.max().item() <= 1.0

    assert torch.isfinite(
        reconstruction
    ).all()


def test_gradient_flows_through_decoder():
    """Gradient phải truyền từ ảnh tái tạo về z và decoder."""

    decoder = build_test_decoder()
    decoder.train()

    z = torch.randn(
        4,
        16,
        requires_grad=True,
    )

    reconstruction = decoder(z)

    loss = reconstruction.square().mean()
    loss.backward()

    assert z.grad is not None

    assert decoder.fc.weight.grad is not None

    first_deconv = decoder.blocks[0][0]

    assert first_deconv.weight.grad is not None

    assert torch.isfinite(
        z.grad
    ).all()

    assert torch.isfinite(
        decoder.fc.weight.grad
    ).all()


def test_decoder_rejects_wrong_tensor_dimension():
    """Decoder chỉ nhận z có shape [B, latent_dim]."""

    decoder = build_test_decoder()

    invalid_z = torch.randn(
        2,
        4,
        16,
    )

    with pytest.raises(
        ValueError,
        match="Expected z",
    ):
        decoder(invalid_z)


def test_decoder_rejects_wrong_latent_dimension():
    """Chiều cuối của z phải bằng latent_dim."""

    decoder = build_test_decoder()

    invalid_z = torch.randn(
        8,
        32,
    )

    with pytest.raises(
        ValueError,
        match="latent_dim",
    ):
        decoder(invalid_z)


def test_config_rejects_mismatched_channels():
    """feature_shape và hidden_dims phải cùng channel đầu."""

    with pytest.raises(
        ValueError,
        match="hidden_dims",
    ):
        ImageDecoderConfig(
            latent_dim=16,
            feature_shape=(
                128,
                4,
                4,
            ),
            hidden_dims=(
                64,
                32,
                16,
            ),
        )
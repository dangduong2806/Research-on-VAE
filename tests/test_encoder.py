"""Tests for encoder shapes and gradients."""
"""Test cho ImageEncoder."""

import pytest
import torch
from torch import nn

from src.models.encoders.image_encoder import (
    ImageEncoder,
    ImageEncoderConfig,
)


def build_test_encoder() -> ImageEncoder:
    """Encoder nhỏ để test nhanh."""

    config = ImageEncoderConfig(
        in_channels=3,
        hidden_dims=(
            16,
            32,
            64,
        ),
        kernel_size=3,
        stride=2,
        padding=1,
    )

    return ImageEncoder(config)


def test_encoder_architecture():
    """Mỗi block phải bám đúng core của PyTorch-VAE."""

    encoder = build_test_encoder()

    assert len(encoder.blocks) == 3

    first_block = encoder.blocks[0]

    assert isinstance(
        first_block[0],
        nn.Conv2d,
    )
    assert isinstance(
        first_block[1],
        nn.BatchNorm2d,
    )
    assert isinstance(
        first_block[2],
        nn.LeakyReLU,
    )

    assert first_block[0].in_channels == 3
    assert first_block[0].out_channels == 16


def test_encoder_output_shapes():
    """Kiểm tra shape qua từng encoder block."""

    encoder = build_test_encoder()

    images = torch.randn(
        4,
        3,
        64,
        64,
    )

    output = encoder(
        images,
        return_intermediates=True,
    )

    assert output.input_shape == (
        4,
        3,
        64,
        64,
    )

    assert output.stages[0].tensor.shape == (
        4,
        16,
        32,
        32,
    )

    assert output.stages[1].tensor.shape == (
        4,
        32,
        16,
        16,
    )

    assert output.stages[2].tensor.shape == (
        4,
        64,
        8,
        8,
    )

    assert output.feature_map.shape == (
        4,
        64,
        8,
        8,
    )

    assert output.flattened.shape == (
        4,
        64 * 8 * 8,
    )


def test_encoder_without_intermediates():
    """Không lưu feature trung gian khi không cần."""

    encoder = build_test_encoder()

    images = torch.randn(
        2,
        3,
        64,
        64,
    )

    output = encoder(
        images,
        return_intermediates=False,
    )

    assert output.stages == ()

    assert output.feature_map.shape == (
        2,
        64,
        8,
        8,
    )


def test_encoder_supports_non_square_images():
    """Encoder phải nhận được ảnh không vuông."""

    encoder = build_test_encoder()

    images = torch.randn(
        2,
        3,
        32,
        48,
    )

    output = encoder(images)

    # 32×48
    # → 16×24
    # → 8×12
    # → 4×6
    assert output.feature_map.shape == (
        2,
        64,
        4,
        6,
    )

    assert output.flattened.shape == (
        2,
        64 * 4 * 6,
    )


def test_infer_flattened_dim():
    """Encoder phải tự suy luận flattened dimension."""

    encoder = build_test_encoder()

    flattened_dim = encoder.infer_flattened_dim(
        input_size=(64, 64)
    )

    assert flattened_dim == (
        64 * 8 * 8
    )


def test_gradient_flows_through_encoder():
    """Gradient phải quay về các convolution weights."""

    encoder = build_test_encoder()
    encoder.train()

    images = torch.randn(
        4,
        3,
        64,
        64,
        requires_grad=True,
    )

    output = encoder(images)

    loss = output.flattened.mean()
    loss.backward()

    first_conv = encoder.blocks[0][0]

    assert first_conv.weight.grad is not None

    assert torch.isfinite(
        first_conv.weight.grad
    ).all()

    assert images.grad is not None


def test_encoder_rejects_wrong_input_dimension():
    """Encoder chỉ nhận batch ảnh [B,C,H,W]."""

    encoder = build_test_encoder()

    invalid_input = torch.randn(
        3,
        64,
        64,
    )

    with pytest.raises(
        ValueError,
        match=r"\[B,C,H,W\]",
    ):
        encoder(invalid_input)
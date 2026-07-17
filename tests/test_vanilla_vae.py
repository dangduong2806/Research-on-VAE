"""End-to-end tests for the modular Vanilla VAE."""
"""Test cho model VanillaVAE hoàn chỉnh."""

import pytest
import torch

from src.models.vanilla_vae import (
    VanillaVAE,
    VanillaVAEConfig,
    VanillaVAEOutput,
)


def build_test_model() -> VanillaVAE:
    """Tạo model nhỏ để test nhanh."""

    config = VanillaVAEConfig(
        input_size=(32, 32),
        in_channels=3,
        latent_dim=16,
        hidden_dims=(
            16,
            32,
            64,
        ),
    )

    return VanillaVAE(config)


def test_forward_output_shapes():
    """Kiểm tra shape của toàn bộ forward pass."""

    model = build_test_model()

    images = torch.randn(
        4,
        3,
        32,
        32,
    )

    output = model(images)

    assert isinstance(
        output,
        VanillaVAEOutput,
    )

    assert output.reconstruction.shape == (
        4,
        3,
        32,
        32,
    )

    assert output.mu.shape == (4, 16)
    assert output.log_var.shape == (4, 16)
    assert output.std.shape == (4, 16)
    assert output.epsilon.shape == (4, 16)
    assert output.z.shape == (4, 16)


def test_reconstruction_range():
    """Ảnh decoder sinh ra phải nằm trong [-1, 1]."""

    model = build_test_model()

    images = torch.randn(
        4,
        3,
        32,
        32,
    )

    output = model(images)

    assert output.reconstruction.min() >= -1.0
    assert output.reconstruction.max() <= 1.0

    assert torch.isfinite(
        output.reconstruction
    ).all()


def test_deterministic_reconstruction():
    """sample=False phải dùng z=mu và cho kết quả ổn định."""

    model = build_test_model()
    model.eval()

    images = torch.randn(
        4,
        3,
        32,
        32,
    )

    with torch.no_grad():
        first = model(
            images,
            sample=False,
        )

        second = model(
            images,
            sample=False,
        )

    assert torch.equal(
        first.z,
        first.mu,
    )

    assert torch.equal(
        first.epsilon,
        torch.zeros_like(first.epsilon),
    )

    assert torch.allclose(
        first.reconstruction,
        second.reconstruction,
    )


def test_generate_images():
    """Model phải sinh được ảnh từ prior N(0, I)."""

    model = build_test_model()
    model.eval()

    with torch.no_grad():
        samples = model.generate(
            num_samples=6,
            device="cpu",
        )

    assert samples.shape == (
        6,
        3,
        32,
        32,
    )

    assert samples.min() >= -1.0
    assert samples.max() <= 1.0


def test_encode_and_decode_separately():
    """Kiểm tra encode() và decode() khi dùng độc lập."""

    model = build_test_model()

    images = torch.randn(
        4,
        3,
        32,
        32,
    )

    gaussian = model.encode(images)

    assert gaussian.mu.shape == (
        4,
        16,
    )

    assert gaussian.log_var.shape == (
        4,
        16,
    )

    reconstruction = model.decode(
        gaussian.mu
    )

    assert reconstruction.shape == (
        4,
        3,
        32,
        32,
    )


def test_gradient_flows_end_to_end():
    """Gradient phải đi qua toàn bộ VAE."""

    model = build_test_model()
    model.train()

    images = torch.randn(
        4,
        3,
        32,
        32,
        requires_grad=True,
    )

    output = model(
        images,
        sample=True,
    )

    loss = output.reconstruction.square().mean()
    loss.backward()

    first_encoder_conv = (
        model.encoder.blocks[0][0]
    )

    first_decoder_deconv = (
        model.decoder.blocks[0][0]
    )

    assert images.grad is not None

    assert (
        first_encoder_conv.weight.grad
        is not None
    )

    assert (
        model.gaussian_head.fc_mu.weight.grad
        is not None
    )

    assert (
        model.gaussian_head.fc_log_var.weight.grad
        is not None
    )

    assert (
        model.decoder.fc.weight.grad
        is not None
    )

    assert (
        first_decoder_deconv.weight.grad
        is not None
    )

    assert torch.isfinite(
        first_encoder_conv.weight.grad
    ).all()

    assert torch.isfinite(
        model.decoder.fc.weight.grad
    ).all()


def test_rejects_wrong_image_size():
    """Kích thước reconstruction phải khớp input."""

    model = build_test_model()

    wrong_size_images = torch.randn(
        4,
        3,
        64,
        64,
    )

    with pytest.raises(
        RuntimeError,
        match="Reconstruction shape",
    ):
        model(wrong_size_images)
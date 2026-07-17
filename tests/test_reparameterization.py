"""Test cho reparameterization trick."""

import pytest
import torch

from src.models.latent.gaussian_head import (
    GaussianHead,
)
from src.models.latent.reparameterization import (
    Reparameterization,
)


def test_output_shapes():
    reparameterize = Reparameterization()

    mu = torch.zeros(8, 16)
    log_var = torch.zeros(8, 16)

    output = reparameterize(
        mu,
        log_var,
    )

    assert output.std.shape == (8, 16)
    assert output.epsilon.shape == (8, 16)
    assert output.z.shape == (8, 16)


def test_reparameterization_formula():
    """Kiểm tra đúng công thức tạo std và z."""

    torch.manual_seed(42)

    reparameterize = Reparameterization()

    mu = torch.randn(4, 8)
    log_var = torch.randn(4, 8)

    output = reparameterize(
        mu,
        log_var,
    )

    expected_std = torch.exp(
        0.5 * log_var
    )

    expected_z = (
        mu
        + expected_std * output.epsilon
    )

    assert torch.allclose(
        output.std,
        expected_std,
    )

    assert torch.allclose(
        output.z,
        expected_z,
    )

    assert torch.all(output.std > 0)


def test_deterministic_mode_returns_mu():
    """sample=False phải trả z bằng mu."""

    reparameterize = Reparameterization()

    mu = torch.randn(4, 8)
    log_var = torch.randn(4, 8)

    output = reparameterize(
        mu,
        log_var,
        sample=False,
    )

    assert torch.equal(
        output.z,
        mu,
    )

    assert torch.equal(
        output.epsilon,
        torch.zeros_like(mu),
    )


def test_gradient_reaches_mu_and_log_var():
    """Gradient phải truyền qua z về mu và log_var."""

    reparameterize = Reparameterization()

    mu = torch.randn(
        4,
        8,
        requires_grad=True,
    )

    log_var = torch.randn(
        4,
        8,
        requires_grad=True,
    )

    output = reparameterize(
        mu,
        log_var,
    )

    loss = output.z.square().mean()
    loss.backward()

    assert mu.grad is not None
    assert log_var.grad is not None

    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(log_var.grad).all()


def test_gradient_reaches_gaussian_head():
    """Gradient phải truyền từ z về Gaussian Head và input."""

    head = GaussianHead(
        input_dim=64,
        latent_dim=16,
    )

    reparameterize = Reparameterization()

    features = torch.randn(
        8,
        64,
        requires_grad=True,
    )

    gaussian = head(features)

    latent = reparameterize(
        gaussian.mu,
        gaussian.log_var,
    )

    loss = latent.z.square().mean()
    loss.backward()

    assert features.grad is not None

    assert head.fc_mu.weight.grad is not None
    assert head.fc_log_var.weight.grad is not None

    assert torch.isfinite(
        features.grad
    ).all()


def test_rejects_different_shapes():
    reparameterize = Reparameterization()

    mu = torch.randn(4, 8)
    log_var = torch.randn(4, 16)

    with pytest.raises(
        ValueError,
        match="cùng shape",
    ):
        reparameterize(
            mu,
            log_var,
        )
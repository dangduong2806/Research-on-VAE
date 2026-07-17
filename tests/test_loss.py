"""Tests for reconstruction, KL, and total VAE losses."""
"""Test cho Reconstruction Loss và KL Divergence."""

import pytest
import torch

from src.losses.reconstruction_loss import (
    ReconstructionLoss,
)
from src.losses.kl_divergence import (
    KLDivergenceLoss,
)


# =========================================================
# Reconstruction Loss
# =========================================================


def test_mse_reconstruction_loss():
    """Kiểm tra MSE với một ví dụ tính được bằng tay."""

    loss_fn = ReconstructionLoss(
        loss_type="mse",
    )

    reconstruction = torch.tensor(
        [1.0, 2.0, 3.0]
    )

    target = torch.tensor(
        [1.0, 1.0, 1.0]
    )

    loss = loss_fn(
        reconstruction,
        target,
    )

    # (0² + 1² + 2²) / 3 = 5/3
    expected = torch.tensor(
        5.0 / 3.0
    )

    assert torch.allclose(
        loss,
        expected,
    )


def test_l1_reconstruction_loss():
    """Kiểm tra L1 loss."""

    loss_fn = ReconstructionLoss(
        loss_type="l1",
    )

    reconstruction = torch.tensor(
        [1.0, 2.0, 3.0]
    )

    target = torch.tensor(
        [1.0, 1.0, 1.0]
    )

    loss = loss_fn(
        reconstruction,
        target,
    )

    # (0 + 1 + 2) / 3 = 1
    assert torch.allclose(
        loss,
        torch.tensor(1.0),
    )


def test_reconstruction_loss_is_zero_for_identical_images():
    """Hai ảnh giống nhau phải có loss bằng 0."""

    loss_fn = ReconstructionLoss(
        loss_type="mse",
    )

    images = torch.randn(
        4,
        3,
        32,
        32,
    )

    loss = loss_fn(
        images,
        images,
    )

    assert torch.allclose(
        loss,
        torch.tensor(0.0),
    )


@pytest.mark.parametrize(
    "reduction, expected",
    [
        ("mean", torch.tensor(5.0 / 3.0)),
        ("sum", torch.tensor(5.0)),
        ("none", torch.tensor([0.0, 1.0, 4.0])),
    ],
)
def test_reconstruction_reductions(
    reduction: str,
    expected: torch.Tensor,
):
    """Kiểm tra mean, sum và none."""

    loss_fn = ReconstructionLoss(
        loss_type="mse",
        reduction=reduction,
    )

    reconstruction = torch.tensor(
        [1.0, 2.0, 3.0]
    )

    target = torch.tensor(
        [1.0, 1.0, 1.0]
    )

    loss = loss_fn(
        reconstruction,
        target,
    )

    assert torch.allclose(
        loss,
        expected,
    )


def test_reconstruction_gradient():
    """Gradient phải truyền về reconstruction."""

    loss_fn = ReconstructionLoss()

    reconstruction = torch.randn(
        4,
        3,
        16,
        16,
        requires_grad=True,
    )

    target = torch.randn(
        4,
        3,
        16,
        16,
    )

    loss = loss_fn(
        reconstruction,
        target,
    )

    loss.backward()

    assert reconstruction.grad is not None

    assert torch.isfinite(
        reconstruction.grad
    ).all()


def test_reconstruction_rejects_different_shapes():
    """Reconstruction và target phải cùng shape."""

    loss_fn = ReconstructionLoss()

    reconstruction = torch.randn(
        4,
        3,
        32,
        32,
    )

    target = torch.randn(
        4,
        3,
        64,
        64,
    )

    with pytest.raises(
        ValueError,
        match="cùng shape",
    ):
        loss_fn(
            reconstruction,
            target,
        )


# =========================================================
# KL Divergence
# =========================================================


def test_kl_is_zero_for_standard_normal():
    """N(0, I) so với N(0, I) phải có KL bằng 0."""

    loss_fn = KLDivergenceLoss()

    mu = torch.zeros(
        4,
        16,
    )

    log_var = torch.zeros(
        4,
        16,
    )

    loss = loss_fn(
        mu,
        log_var,
    )

    assert torch.allclose(
        loss,
        torch.tensor(0.0),
    )


def test_kl_known_value():
    """Kiểm tra KL bằng một ví dụ tính được bằng tay."""

    loss_fn = KLDivergenceLoss(
        reduction="mean",
    )

    mu = torch.tensor(
        [
            [1.0, 0.0],
        ]
    )

    log_var = torch.zeros_like(mu)

    loss = loss_fn(
        mu,
        log_var,
    )

    # KL = 0.5 khi:
    # mu = [1, 0]
    # variance = [1, 1]
    assert torch.allclose(
        loss,
        torch.tensor(0.5),
    )


@pytest.mark.parametrize(
    "reduction, expected",
    [
        (
            "mean",
            torch.tensor(0.5),
        ),
        (
            "sum",
            torch.tensor(1.0),
        ),
        (
            "none",
            torch.tensor(
                [0.5, 0.5]
            ),
        ),
    ],
)
def test_kl_reductions(
    reduction: str,
    expected: torch.Tensor,
):
    """Kiểm tra mean, sum và none của KL."""

    loss_fn = KLDivergenceLoss(
        reduction=reduction,
    )

    mu = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )

    log_var = torch.zeros_like(mu)

    loss = loss_fn(
        mu,
        log_var,
    )

    assert torch.allclose(
        loss,
        expected,
    )


def test_kl_gradient():
    """Gradient phải truyền về mu và log_var."""

    loss_fn = KLDivergenceLoss()

    mu = torch.randn(
        4,
        16,
        requires_grad=True,
    )

    log_var = torch.randn(
        4,
        16,
        requires_grad=True,
    )

    loss = loss_fn(
        mu,
        log_var,
    )

    loss.backward()

    assert mu.grad is not None
    assert log_var.grad is not None

    assert torch.isfinite(
        mu.grad
    ).all()

    assert torch.isfinite(
        log_var.grad
    ).all()


def test_kl_rejects_different_shapes():
    """mu và log_var phải cùng shape."""

    loss_fn = KLDivergenceLoss()

    mu = torch.randn(
        4,
        16,
    )

    log_var = torch.randn(
        4,
        32,
    )

    with pytest.raises(
        ValueError,
        match="cùng shape",
    ):
        loss_fn(
            mu,
            log_var,
        )


def test_kl_rejects_wrong_dimension():
    """KL loss yêu cầu tensor [B, latent_dim]."""

    loss_fn = KLDivergenceLoss()

    mu = torch.randn(
        2,
        4,
        8,
    )

    log_var = torch.randn_like(mu)

    with pytest.raises(
        ValueError,
        match="latent_dim",
    ):
        loss_fn(
            mu,
            log_var,
        )


# =========================================================
# Invalid configuration
# =========================================================


@pytest.mark.parametrize(
    "loss_type",
    [
        "bce",
        "ssim",
        "invalid",
    ],
)
def test_reconstruction_rejects_invalid_loss_type(
    loss_type: str,
):
    with pytest.raises(
        ValueError,
        match="loss_type",
    ):
        ReconstructionLoss(
            loss_type=loss_type,
        )


@pytest.mark.parametrize(
    "reduction",
    [
        "average",
        "batch",
        "invalid",
    ],
)
def test_losses_reject_invalid_reduction(
    reduction: str,
):
    with pytest.raises(ValueError):
        ReconstructionLoss(
            reduction=reduction,
        )

    with pytest.raises(ValueError):
        KLDivergenceLoss(
            reduction=reduction,
        )
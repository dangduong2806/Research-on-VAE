"""Test cho Gaussian latent head."""

import pytest
import torch
from torch import nn

from src.models.latent.gaussian_head import (
    GaussianHead,
    GaussianOutput,
)


def build_test_head() -> GaussianHead:
    """Tạo Gaussian Head nhỏ để test nhanh."""

    return GaussianHead(
        input_dim=64,
        latent_dim=16,
    )


def test_gaussian_head_architecture():
    """Gaussian Head phải có hai Linear layer độc lập."""

    head = build_test_head()

    assert isinstance(
        head.fc_mu,
        nn.Linear,
    )

    assert isinstance(
        head.fc_log_var,
        nn.Linear,
    )

    assert head.fc_mu.in_features == 64
    assert head.fc_mu.out_features == 16

    assert head.fc_log_var.in_features == 64
    assert head.fc_log_var.out_features == 16


def test_gaussian_output_shapes():
    """Kiểm tra shape của mu và log_var."""

    head = build_test_head()

    features = torch.randn(
        8,
        64,
    )

    output = head(features)

    assert isinstance(
        output,
        GaussianOutput,
    )

    assert output.mu.shape == (
        8,
        16,
    )

    assert output.log_var.shape == (
        8,
        16,
    )


def test_standard_deviation():
    """std phải bằng exp(0.5 * log_var)."""

    log_var = torch.tensor(
        [
            [0.0, 2.0, -2.0],
        ]
    )

    output = GaussianOutput(
        mu=torch.zeros_like(log_var),
        log_var=log_var,
    )

    expected_std = torch.exp(
        0.5 * log_var
    )

    assert torch.allclose(
        output.std,
        expected_std,
    )

    # exp() luôn tạo giá trị dương.
    assert torch.all(
        output.std > 0
    )


def test_sample_shape():
    """Latent sample z phải cùng shape với mu."""

    output = GaussianOutput(
        mu=torch.zeros(
            4,
            12,
        ),
        log_var=torch.zeros(
            4,
            12,
        ),
    )

    z = output.sample()

    assert z.shape == (
        4,
        12,
    )

    assert torch.isfinite(z).all()


def test_sampling_contains_randomness():
    """Hai lần lấy mẫu thường phải tạo hai z khác nhau."""

    output = GaussianOutput(
        mu=torch.zeros(
            4,
            16,
        ),
        log_var=torch.zeros(
            4,
            16,
        ),
    )

    first_sample = output.sample()
    second_sample = output.sample()

    assert not torch.equal(
        first_sample,
        second_sample,
    )


def test_gradient_flows_through_reparameterization():
    """Gradient phải truyền qua z về Gaussian Head và input."""

    head = build_test_head()

    features = torch.randn(
        8,
        64,
        requires_grad=True,
    )

    output = head(features)

    z = output.sample()

    loss = z.square().mean()
    loss.backward()

    assert features.grad is not None

    assert head.fc_mu.weight.grad is not None
    assert head.fc_log_var.weight.grad is not None

    assert torch.isfinite(
        features.grad
    ).all()

    assert torch.isfinite(
        head.fc_mu.weight.grad
    ).all()

    assert torch.isfinite(
        head.fc_log_var.weight.grad
    ).all()


def test_rejects_wrong_tensor_dimension():
    """Gaussian Head chỉ nhận tensor [B, F]."""

    head = build_test_head()

    invalid_features = torch.randn(
        2,
        4,
        8,
    )

    with pytest.raises(
        ValueError,
        match=r"\[B,F\]",
    ):
        head(invalid_features)


def test_rejects_wrong_feature_dimension():
    """Chiều F phải khớp input_dim của Gaussian Head."""

    head = build_test_head()

    invalid_features = torch.randn(
        8,
        32,
    )

    with pytest.raises(
        ValueError,
        match="Feature dimension",
    ):
        head(invalid_features)


@pytest.mark.parametrize(
    "input_dim, latent_dim",
    [
        (0, 16),
        (64, 0),
        (-1, 16),
        (64, -1),
    ],
)
def test_rejects_invalid_dimensions(
    input_dim: int,
    latent_dim: int,
):
    """input_dim và latent_dim phải là số dương."""

    with pytest.raises(ValueError):
        GaussianHead(
            input_dim=input_dim,
            latent_dim=latent_dim,
        )
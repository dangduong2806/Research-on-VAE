"""
Các test trong file này chỉ kiểm tra preprocessing ảnh,
không sử dụng Dataset hoặc DataLoader.

Mục tiêu:

1. Mọi resize mode đều trả đúng kích thước.
2. Chuyển đổi channel hoạt động đúng.
3. Normalization tạo đúng miền giá trị.
4. Denormalization khôi phục ảnh về [0, 1].
5. Evaluation transform có tính deterministic.
6. Cấu hình không hợp lệ bị phát hiện sớm.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from src.data.transforms import (
    ImageTransformConfig,
    build_image_transform,
    denormalize_image_tensor,
)


def create_pattern_image(
    *,
    width: int = 80,
    height: int = 40,
) -> Image.Image:
    """Tạo ảnh RGB có pattern để phục vụ kiểm thử.

    Ảnh không chỉ có một màu vì ta muốn các phép crop và resize
    xử lý một ảnh có nội dung thay đổi theo vị trí.
    """

    image = Image.new(
        mode="RGB",
        size=(width, height),
    )

    pixels: list[tuple[int, int, int]] = []

    for y in range(height):
        for x in range(width):
            red = int(
                255 * x / max(width - 1, 1)
            )

            green = int(
                255 * y / max(height - 1, 1)
            )

            blue = (
                255
                if (x // 8 + y // 8) % 2 == 0
                else 0
            )

            pixels.append(
                (
                    red,
                    green,
                    blue,
                )
            )

    image.putdata(pixels)

    return image


@pytest.mark.parametrize(
    "resize_mode",
    [
        "resize",
        "resize_and_pad",
        "center_crop",
        "random_crop",
    ],
)
def test_all_resize_modes_return_target_shape(
    resize_mode: str,
) -> None:
    """Mọi resize mode phải trả đúng [C, H, W]."""

    config = ImageTransformConfig(
        input_size=(32, 48),
        in_channels=3,
        resize_mode=resize_mode,
        normalization="zero_to_one",
    )

    transform = build_image_transform(
        config,
        train=True,
    )

    image = create_pattern_image(
        width=90,
        height=50,
    )

    tensor = transform(image)

    assert tensor.shape == (
        3,
        32,
        48,
    )

    assert tensor.dtype == torch.float32

    assert torch.isfinite(tensor).all()

    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0


@pytest.mark.parametrize(
    (
        "in_channels",
        "expected_shape",
    ),
    [
        (
            1,
            (1, 24, 36),
        ),
        (
            3,
            (3, 24, 36),
        ),
        (
            4,
            (4, 24, 36),
        ),
    ],
)
def test_transform_converts_number_of_channels(
    in_channels: int,
    expected_shape: tuple[int, int, int],
) -> None:
    """Ảnh nguồn RGB phải được chuyển đúng thành L, RGB hoặc RGBA."""

    config = ImageTransformConfig(
        input_size=(24, 36),
        in_channels=in_channels,
        resize_mode="resize",
        normalization="zero_to_one",
    )

    transform = build_image_transform(
        config,
        train=False,
    )

    image = create_pattern_image()

    tensor = transform(image)

    assert tensor.shape == expected_shape

    assert tensor.dtype == torch.float32


def test_minus_one_to_one_normalization_can_be_reversed() -> None:
    """Normalization [-1,1] phải có thể đảo lại chính xác về [0,1]."""

    # Ảnh gồm một pixel đen và một pixel trắng.
    image = Image.new(
        mode="RGB",
        size=(2, 1),
    )

    image.putdata(
        [
            (0, 0, 0),
            (255, 255, 255),
        ]
    )

    config = ImageTransformConfig(
        input_size=(1, 2),
        in_channels=3,
        resize_mode="resize",
        normalization="minus_one_to_one",
        interpolation="nearest",
    )

    transform = build_image_transform(
        config,
        train=False,
    )

    normalized = transform(image)

    assert normalized.shape == (
        3,
        1,
        2,
    )

    # Pixel đen phải trở thành -1.
    assert torch.allclose(
        normalized[:, :, 0],
        torch.full(
            (3, 1),
            -1.0,
        ),
    )

    # Pixel trắng phải trở thành 1.
    assert torch.allclose(
        normalized[:, :, 1],
        torch.full(
            (3, 1),
            1.0,
        ),
    )

    restored = denormalize_image_tensor(
        normalized,
        config,
    )

    expected = torch.tensor(
        [
            [
                [0.0, 1.0],
            ],
            [
                [0.0, 1.0],
            ],
            [
                [0.0, 1.0],
            ],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(
        restored,
        expected,
        atol=1e-6,
    )


def test_denormalize_supports_batch_tensor() -> None:
    """Denormalization phải hỗ trợ cả [B,C,H,W]."""

    config = ImageTransformConfig(
        input_size=(16, 16),
        in_channels=3,
        resize_mode="resize",
        normalization="minus_one_to_one",
    )

    transform = build_image_transform(
        config,
        train=False,
    )

    image = create_pattern_image()

    sample = transform(image)

    batch = torch.stack(
        [
            sample,
            sample,
        ],
        dim=0,
    )

    restored = denormalize_image_tensor(
        batch,
        config,
    )

    assert restored.shape == (
        2,
        3,
        16,
        16,
    )

    assert restored.min().item() >= 0.0
    assert restored.max().item() <= 1.0


def test_evaluation_random_crop_is_deterministic() -> None:
    """random_crop ở validation/test phải được thay bằng center crop."""

    config = ImageTransformConfig(
        input_size=(32, 32),
        in_channels=3,
        resize_mode="random_crop",
        normalization="zero_to_one",
    )

    evaluation_transform = build_image_transform(
        config,
        train=False,
    )

    image = create_pattern_image(
        width=100,
        height=50,
    )

    first_result = evaluation_transform(image)
    second_result = evaluation_transform(image)

    assert torch.equal(
        first_result,
        second_result,
    )


def test_custom_normalization_requires_mean_and_std() -> None:
    """Custom normalization không được thiếu mean/std."""

    with pytest.raises(
        ValueError,
        match="normalization_mean",
    ):
        ImageTransformConfig(
            input_size=(32, 32),
            in_channels=3,
            normalization="custom",
        )


def test_imagenet_normalization_requires_rgb() -> None:
    """ImageNet normalization chỉ hợp lệ với ba channel."""

    with pytest.raises(
        ValueError,
        match="in_channels=3",
    ):
        ImageTransformConfig(
            input_size=(32, 32),
            in_channels=1,
            normalization="imagenet",
        )


def test_transform_config_from_mapping() -> None:
    """Config đọc từ YAML mapping phải được parse đúng."""

    mapping = {
        "input_size": [48, 80],
        "in_channels": 1,
        "resize_mode": "center_crop",
        "normalization": "zero_to_one",
        "interpolation": "bicubic",
        "fill_value": 128,
    }

    config = ImageTransformConfig.from_mapping(
        mapping
    )

    assert config.input_size == (
        48,
        80,
    )

    assert config.in_channels == 1
    assert config.resize_mode == "center_crop"
    assert config.normalization == "zero_to_one"
    assert config.interpolation == "bicubic"
    assert config.fill_value == 128
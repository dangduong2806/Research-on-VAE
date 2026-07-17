"""Các cấu trúc output có tên rõ ràng cho model Vanilla VAE.

Ở giai đoạn Image Encoder, file này định nghĩa:

1. EncoderStageOutput:
       Output của một convolution block.

2. ImageEncoderOutput:
       Output hoàn chỉnh của ImageEncoder.

Quy ước shape:

    Input image:
        [B, C_in, H_in, W_in]

    Intermediate feature:
        [B, C_i, H_i, W_i]

    Final feature map:
        [B, C_out, H_out, W_out]

    Flattened feature:
        [B, F]

Trong đó:

    F = C_out * H_out * W_out
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

from torch import Tensor


# Alias dùng để biểu diễn shape tensor.
TensorShape = tuple[int, ...]


def tensor_shape(
    tensor: Tensor,
) -> TensorShape:
    """Chuyển torch.Size thành tuple[int, ...].

    Ví dụ:

        tensor.shape
            → torch.Size([8, 64, 16, 16])

        tensor_shape(tensor)
            → (8, 64, 16, 16)
    """

    return tuple(
        int(dimension)
        for dimension in tensor.shape
    )


@dataclass(slots=True)
class EncoderStageOutput:
    """Output của một block trong Image Encoder.

    Ví dụ encoder có ba block:

        block_0:
            [B, 3, 64, 64]
                →
            [B, 32, 32, 32]

        block_1:
            [B, 32, 32, 32]
                →
            [B, 64, 16, 16]

        block_2:
            [B, 64, 16, 16]
                →
            [B, 128, 8, 8]

    Mỗi kết quả trên được biểu diễn bằng một
    EncoderStageOutput.

    Attributes:
        name:
            Tên của encoder block.

            Ví dụ:
                "block_0"
                "block_1"
                "block_2"

        tensor:
            Feature map sau block đó.

            Shape:
                [B, C, H, W]
    """

    name: str
    tensor: Tensor

    def __post_init__(self) -> None:
        """Tự động kiểm tra sau khi object được tạo."""

        self.validate()

    @property
    def shape(self) -> TensorShape:
        """Shape đầy đủ của feature map."""

        return tensor_shape(self.tensor)

    @property
    def batch_size(self) -> int:
        """Số sample trong batch."""

        return int(self.tensor.shape[0])

    @property
    def channels(self) -> int:
        """Số channel của feature map."""

        return int(self.tensor.shape[1])

    @property
    def height(self) -> int:
        """Chiều cao feature map."""

        return int(self.tensor.shape[2])

    @property
    def width(self) -> int:
        """Chiều rộng feature map."""

        return int(self.tensor.shape[3])

    def validate(self) -> None:
        """Kiểm tra output của encoder block."""

        if not self.name.strip():
            raise ValueError(
                "Encoder stage name không được để trống."
            )

        if self.tensor.ndim != 4:
            raise ValueError(
                f"Encoder stage '{self.name}' phải có tensor "
                "shape [B, C, H, W], nhưng nhận được "
                f"{tensor_shape(self.tensor)}."
            )

        if self.batch_size <= 0:
            raise ValueError(
                f"Encoder stage '{self.name}' có batch size "
                "không hợp lệ."
            )

        if self.channels <= 0:
            raise ValueError(
                f"Encoder stage '{self.name}' có số channel "
                "không hợp lệ."
            )

        if self.height <= 0 or self.width <= 0:
            raise ValueError(
                f"Encoder stage '{self.name}' có kích thước "
                "không gian không hợp lệ: "
                f"{self.height}x{self.width}."
            )

    def detached(self) -> "EncoderStageOutput":
        """Tạo output mới đã tách khỏi computation graph.

        Hàm này hữu ích khi:

        - logging;
        - visualization;
        - debug;
        - lưu feature map;
        - tránh giữ computation graph không cần thiết.
        """

        return EncoderStageOutput(
            name=self.name,
            tensor=self.tensor.detach(),
        )

    def summary(self) -> dict[str, Any]:
        """Thông tin ngắn gọn dùng để debug."""

        return {
            "name": self.name,
            "shape": list(self.shape),
            "batch_size": self.batch_size,
            "channels": self.channels,
            "height": self.height,
            "width": self.width,
            "dtype": str(self.tensor.dtype),
            "device": str(self.tensor.device),
        }


@dataclass(slots=True)
class ImageEncoderOutput:
    """Output hoàn chỉnh của ImageEncoder.

    Attributes:
        input_shape:
            Shape của ảnh trước khi đi vào encoder.

            Dạng:
                [B, C_in, H_in, W_in]

        feature_map:
            Feature map cuối cùng của encoder.

            Dạng:
                [B, C_out, H_out, W_out]

        flattened:
            Feature map cuối đã được flatten.

            Dạng:
                [B, F]

            Trong đó:
                F = C_out * H_out * W_out

        stages:
            Output của từng convolution block.

            Có thể là tuple rỗng nếu encoder được gọi với:

                return_intermediates=False
    """

    input_shape: TensorShape
    feature_map: Tensor
    flattened: Tensor
    stages: tuple[EncoderStageOutput, ...] = ()

    def __post_init__(self) -> None:
        """Kiểm tra output ngay sau khi tạo."""

        self.validate()

    @property
    def batch_size(self) -> int:
        """Số ảnh trong batch."""

        return int(self.feature_map.shape[0])

    @property
    def feature_shape(self) -> TensorShape:
        """Shape của final feature map."""

        return tensor_shape(
            self.feature_map
        )

    @property
    def flattened_shape(self) -> TensorShape:
        """Shape của flattened feature."""

        return tensor_shape(
            self.flattened
        )

    @property
    def output_channels(self) -> int:
        """Số channel của final feature map."""

        return int(
            self.feature_map.shape[1]
        )

    @property
    def output_height(self) -> int:
        """Chiều cao của final feature map."""

        return int(
            self.feature_map.shape[2]
        )

    @property
    def output_width(self) -> int:
        """Chiều rộng của final feature map."""

        return int(
            self.feature_map.shape[3]
        )

    @property
    def flattened_dim(self) -> int:
        """Số feature của mỗi sample sau flatten."""

        return int(
            self.flattened.shape[1]
        )

    @property
    def number_of_stages(self) -> int:
        """Số intermediate stage được lưu."""

        return len(self.stages)

    def validate(self) -> None:
        """Kiểm tra tính nhất quán của encoder output."""

        self._validate_input_shape()
        self._validate_feature_map()
        self._validate_flattened()
        self._validate_stages()

    def _validate_input_shape(self) -> None:
        """Kiểm tra input_shape."""

        if len(self.input_shape) != 4:
            raise ValueError(
                "ImageEncoderOutput.input_shape phải có dạng "
                "[B, C, H, W], nhưng nhận được "
                f"{self.input_shape}."
            )

        if any(
            dimension <= 0
            for dimension in self.input_shape
        ):
            raise ValueError(
                "Mọi dimension trong input_shape phải lớn hơn 0, "
                f"nhận được {self.input_shape}."
            )

    def _validate_feature_map(self) -> None:
        """Kiểm tra final feature map."""

        if self.feature_map.ndim != 4:
            raise ValueError(
                "feature_map phải có shape [B, C, H, W], "
                f"nhưng nhận được {self.feature_shape}."
            )

        if self.batch_size != self.input_shape[0]:
            raise ValueError(
                "Batch size của feature_map không khớp input. "
                f"Input batch={self.input_shape[0]}, "
                f"feature batch={self.batch_size}."
            )

        if self.output_channels <= 0:
            raise ValueError(
                "Số channel của feature_map phải lớn hơn 0."
            )

        if (
            self.output_height <= 0
            or self.output_width <= 0
        ):
            raise ValueError(
                "Kích thước không gian của feature_map phải "
                "lớn hơn 0."
            )

    def _validate_flattened(self) -> None:
        """Kiểm tra flattened feature."""

        if self.flattened.ndim != 2:
            raise ValueError(
                "flattened phải có shape [B, F], "
                f"nhưng nhận được {self.flattened_shape}."
            )

        if int(self.flattened.shape[0]) != self.batch_size:
            raise ValueError(
                "Batch size của flattened không khớp "
                "feature_map. "
                f"Feature batch={self.batch_size}, "
                f"flattened batch={self.flattened.shape[0]}."
            )

        expected_flattened_dim = prod(
            self.feature_shape[1:]
        )

        if self.flattened_dim != expected_flattened_dim:
            raise ValueError(
                "Flattened dimension không khớp feature_map. "
                f"Feature map={self.feature_shape}, "
                f"expected flattened dim="
                f"{expected_flattened_dim}, "
                f"actual flattened dim={self.flattened_dim}."
            )

    def _validate_stages(self) -> None:
        """Kiểm tra các intermediate encoder stages."""

        stage_names: set[str] = set()

        for stage in self.stages:
            if stage.name in stage_names:
                raise ValueError(
                    "Tên encoder stage không được trùng nhau: "
                    f"'{stage.name}'."
                )

            stage_names.add(
                stage.name
            )

            if stage.batch_size != self.batch_size:
                raise ValueError(
                    f"Batch size của stage '{stage.name}' "
                    "không khớp final feature map. "
                    f"Stage batch={stage.batch_size}, "
                    f"final batch={self.batch_size}."
                )

        # Nếu có lưu intermediate stages, stage cuối phải có
        # cùng shape với final feature map.
        if self.stages:
            last_stage = self.stages[-1]

            if last_stage.shape != self.feature_shape:
                raise ValueError(
                    "Encoder stage cuối không khớp final "
                    "feature map. "
                    f"Last stage={last_stage.shape}, "
                    f"feature map={self.feature_shape}."
                )

    def detached(self) -> "ImageEncoderOutput":
        """Tạo output mới đã detach khỏi computation graph.

        Output này phù hợp để:

        - in thống kê;
        - lưu feature;
        - visualization;
        - phân tích activation.
        """

        return ImageEncoderOutput(
            input_shape=self.input_shape,
            feature_map=self.feature_map.detach(),
            flattened=self.flattened.detach(),
            stages=tuple(
                stage.detached()
                for stage in self.stages
            ),
        )

    def shape_summary(self) -> dict[str, Any]:
        """Trả về báo cáo shape của toàn bộ encoder."""

        return {
            "input_shape": list(
                self.input_shape
            ),
            "feature_shape": list(
                self.feature_shape
            ),
            "flattened_shape": list(
                self.flattened_shape
            ),
            "flattened_dim": (
                self.flattened_dim
            ),
            "number_of_stages": (
                self.number_of_stages
            ),
            "stages": [
                stage.summary()
                for stage in self.stages
            ],
        }

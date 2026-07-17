"""Kiểm tra trực quan và thống kê Image Data Pipeline.

Script này thực hiện:

1. Đọc file cấu hình YAML.
2. Xây dựng train/validation/test DataLoader.
3. Lấy một số batch thực tế.
4. Kiểm tra shape, dtype và metadata.
5. Tính min, max, mean và standard deviation theo channel.
6. Đưa ảnh đã normalize trở lại miền [0, 1].
7. Lưu một figure trực quan.
8. Lưu báo cáo JSON.

Ví dụ chạy:

    python inspect_data.py \
        --config configs/image_128.yaml \
        --split train

Hoặc:

    python inspect_data.py \
        --config configs/image_128.yaml \
        --split val \
        --num-batches 3 \
        --num-images 8
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.batch import ImageBatch
from src.data.dataloader import (
    DataLoaderBundle,
    build_dataloaders,
)
from src.data.transforms import (
    ImageTransformConfig,
    denormalize_image_tensor,
)
from src.utils.config import (
    ProjectConfig,
    load_project_config,
)


def parse_arguments() -> argparse.Namespace:
    """Đọc tham số từ command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Kiểm tra input, output và thống kê của "
            "Image Data Pipeline."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/image_128.yaml",
        help=(
            "Đường dẫn đến file cấu hình YAML. "
            "Mặc định: configs/image_128.yaml"
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=[
            "train",
            "val",
            "test",
        ],
        default="train",
        help=(
            "Split cần kiểm tra. "
            "Các lựa chọn: train, val, test."
        ),
    )

    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help=(
            "Số batch dùng để tính thống kê. "
            "Mặc định: 1."
        ),
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=8,
        help=(
            "Số ảnh trong batch đầu tiên được trực quan hóa. "
            "Mặc định: 8."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Thư mục lưu figure và báo cáo JSON. "
            "Nếu không truyền, script dùng logging.output_dir."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Hiển thị figure bằng matplotlib ngoài việc lưu file."
        ),
    )

    arguments = parser.parse_args()

    if arguments.num_batches <= 0:
        parser.error(
            "--num-batches phải lớn hơn 0."
        )

    if arguments.num_images <= 0:
        parser.error(
            "--num-images phải lớn hơn 0."
        )

    return arguments


def inspect_dataloader(
    loader: DataLoader,
    *,
    num_batches: int,
) -> tuple[dict[str, Any], ImageBatch]:
    """Đọc một số batch và tính thống kê theo channel.

    Args:
        loader:
            DataLoader cần kiểm tra.

        num_batches:
            Số batch tối đa được đọc.

    Returns:
        Một tuple gồm:

        1. Dictionary thống kê.
        2. Batch đầu tiên để trực quan hóa.

    Thống kê được tính trực tiếp trên tensor sau normalization,
    tức là đúng tensor sẽ được đưa vào model.
    """

    channel_sum: Optional[Tensor] = None
    channel_square_sum: Optional[Tensor] = None
    channel_min: Optional[Tensor] = None
    channel_max: Optional[Tensor] = None

    total_pixels_per_channel = 0
    total_samples = 0
    inspected_batches = 0

    first_batch: Optional[ImageBatch] = None

    original_size_counter: Counter[
        tuple[int, int]
    ] = Counter()

    processed_size_counter: Counter[
        tuple[int, int]
    ] = Counter()

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= num_batches:
                break

            if not isinstance(batch, ImageBatch):
                raise TypeError(
                    "DataLoader phải trả về ImageBatch, "
                    f"nhưng nhận được {type(batch).__name__}."
                )

            if first_batch is None:
                first_batch = batch

            images = batch.images

            validate_batch_tensor(
                images,
                batch_index=batch_index,
            )

            # images có shape:
            #
            # [B, C, H, W]
            batch_size = int(images.shape[0])
            height = int(images.shape[2])
            width = int(images.shape[3])

            # Mỗi channel có B × H × W giá trị pixel.
            pixels_per_channel = (
                batch_size
                * height
                * width
            )

            current_sum = images.sum(
                dim=(0, 2, 3)
            )

            current_square_sum = (
                images.square().sum(
                    dim=(0, 2, 3)
                )
            )

            current_min = images.amin(
                dim=(0, 2, 3)
            )

            current_max = images.amax(
                dim=(0, 2, 3)
            )

            if channel_sum is None:
                channel_sum = current_sum
                channel_square_sum = (
                    current_square_sum
                )
                channel_min = current_min
                channel_max = current_max
            else:
                channel_sum += current_sum

                if channel_square_sum is None:
                    raise RuntimeError(
                        "channel_square_sum chưa được khởi tạo."
                    )

                if channel_min is None:
                    raise RuntimeError(
                        "channel_min chưa được khởi tạo."
                    )

                if channel_max is None:
                    raise RuntimeError(
                        "channel_max chưa được khởi tạo."
                    )

                channel_square_sum += (
                    current_square_sum
                )

                channel_min = torch.minimum(
                    channel_min,
                    current_min,
                )

                channel_max = torch.maximum(
                    channel_max,
                    current_max,
                )

            total_pixels_per_channel += (
                pixels_per_channel
            )
            total_samples += batch_size
            inspected_batches += 1

            update_size_counter(
                original_size_counter,
                batch.original_sizes,
            )

            update_size_counter(
                processed_size_counter,
                batch.processed_sizes,
            )

    if first_batch is None:
        raise ValueError(
            "DataLoader không trả về batch nào. "
            "Dataset có thể đang rỗng."
        )

    if (
        channel_sum is None
        or channel_square_sum is None
        or channel_min is None
        or channel_max is None
    ):
        raise RuntimeError(
            "Không thể tính thống kê channel."
        )

    channel_mean = (
        channel_sum
        / total_pixels_per_channel
    )

    # Var(X) = E[X²] - E[X]²
    channel_variance = (
        channel_square_sum
        / total_pixels_per_channel
        - channel_mean.square()
    )

    # Do sai số số thực, variance đôi khi có thể âm rất nhỏ.
    channel_variance = (
        channel_variance.clamp_min(0.0)
    )

    channel_std = torch.sqrt(
        channel_variance
    )

    statistics = {
        "inspected_batches": inspected_batches,
        "inspected_samples": total_samples,
        "pixels_per_channel": (
            total_pixels_per_channel
        ),
        "batch_shape": list(
            first_batch.shape
        ),
        "dtype": str(
            first_batch.images.dtype
        ),
        "channel_statistics": {
            "min": tensor_to_float_list(
                channel_min
            ),
            "max": tensor_to_float_list(
                channel_max
            ),
            "mean": tensor_to_float_list(
                channel_mean
            ),
            "std": tensor_to_float_list(
                channel_std
            ),
        },
        "original_size_distribution": (
            size_counter_to_dict(
                original_size_counter
            )
        ),
        "processed_size_distribution": (
            size_counter_to_dict(
                processed_size_counter
            )
        ),
        "first_batch": (
            describe_batch(first_batch)
        ),
    }

    return statistics, first_batch


def validate_batch_tensor(
    images: Tensor,
    *,
    batch_index: int,
) -> None:
    """Kiểm tra tensor ảnh trong một batch."""

    if images.ndim != 4:
        raise ValueError(
            "Batch tensor phải có shape [B, C, H, W], "
            f"nhưng batch {batch_index} có shape "
            f"{tuple(images.shape)}."
        )

    if images.shape[0] <= 0:
        raise ValueError(
            f"Batch {batch_index} không chứa ảnh."
        )

    if images.shape[1] <= 0:
        raise ValueError(
            f"Batch {batch_index} không có channel."
        )

    if images.shape[2] <= 0:
        raise ValueError(
            f"Batch {batch_index} có height không hợp lệ."
        )

    if images.shape[3] <= 0:
        raise ValueError(
            f"Batch {batch_index} có width không hợp lệ."
        )

    if not images.is_floating_point():
        raise TypeError(
            "Tensor ảnh sau transform phải là floating point, "
            f"nhưng nhận được dtype={images.dtype}."
        )

    if not torch.isfinite(images).all():
        raise ValueError(
            f"Batch {batch_index} chứa NaN hoặc Infinity."
        )


def update_size_counter(
    counter: Counter[tuple[int, int]],
    sizes: Tensor,
) -> None:
    """Đếm tần suất xuất hiện của các kích thước ảnh.

    Args:
        counter:
            Counter đang được cập nhật.

        sizes:
            Tensor có shape [B, 2], mỗi hàng là [H, W].
    """

    if sizes.ndim != 2 or sizes.shape[1] != 2:
        raise ValueError(
            "Tensor kích thước phải có shape [B, 2], "
            f"nhận được {tuple(sizes.shape)}."
        )

    for size in sizes.cpu().tolist():
        height = int(size[0])
        width = int(size[1])

        counter[
            (
                height,
                width,
            )
        ] += 1


def describe_batch(
    batch: ImageBatch,
) -> dict[str, Any]:
    """Chuyển metadata của batch đầu tiên thành dictionary."""

    samples: list[dict[str, Any]] = []

    for index in range(batch.batch_size):
        label = (
            int(batch.labels[index].item())
            if batch.labels is not None
            else None
        )

        samples.append(
            {
                "index": index,
                "path": batch.paths[index],
                "original_size": [
                    int(
                        batch.original_sizes[
                            index,
                            0,
                        ].item()
                    ),
                    int(
                        batch.original_sizes[
                            index,
                            1,
                        ].item()
                    ),
                ],
                "processed_size": [
                    int(
                        batch.processed_sizes[
                            index,
                            0,
                        ].item()
                    ),
                    int(
                        batch.processed_sizes[
                            index,
                            1,
                        ].item()
                    ),
                ],
                "label": label,
                "class_name": (
                    batch.class_names[index]
                ),
            }
        )

    return {
        "shape": list(batch.shape),
        "batch_size": batch.batch_size,
        "channels": batch.channels,
        "height": batch.height,
        "width": batch.width,
        "samples": samples,
    }


def visualize_batch(
    batch: ImageBatch,
    transform_config: ImageTransformConfig,
    *,
    num_images: int,
    output_path: Path,
    show: bool,
) -> None:
    """Trực quan hóa một số ảnh trong batch đầu tiên.

    Ảnh được denormalize về miền [0, 1] trước khi hiển thị.
    """

    display_images = denormalize_image_tensor(
        batch.images.detach().cpu(),
        transform_config,
        clamp=True,
    )

    number_to_show = min(
        num_images,
        batch.batch_size,
    )

    columns = min(
        4,
        number_to_show,
    )

    rows = int(
        np.ceil(
            number_to_show / columns
        )
    )

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            columns * 4.0,
            rows * 4.0,
        ),
        squeeze=False,
    )

    flattened_axes = axes.flatten()

    for index in range(number_to_show):
        axis = flattened_axes[index]

        image_tensor = display_images[index]

        plot_tensor_image(
            axis,
            image_tensor,
        )

        image_name = Path(
            batch.paths[index]
        ).name

        original_height = int(
            batch.original_sizes[
                index,
                0,
            ].item()
        )
        original_width = int(
            batch.original_sizes[
                index,
                1,
            ].item()
        )

        processed_height = int(
            batch.processed_sizes[
                index,
                0,
            ].item()
        )
        processed_width = int(
            batch.processed_sizes[
                index,
                1,
            ].item()
        )

        title_lines = [
            image_name,
            (
                f"original: "
                f"{original_height}×{original_width}"
            ),
            (
                f"processed: "
                f"{processed_height}×{processed_width}"
            ),
        ]

        class_name = batch.class_names[index]

        if class_name is not None:
            title_lines.append(
                f"class: {class_name}"
            )

        axis.set_title(
            "\n".join(title_lines),
            fontsize=9,
        )
        axis.axis("off")

    # Ẩn các subplot còn thừa.
    for index in range(
        number_to_show,
        len(flattened_axes),
    ):
        flattened_axes[index].axis("off")

    figure.suptitle(
        (
            "Images after preprocessing "
            "and inverse normalization"
        ),
        fontsize=14,
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(figure)


def plot_tensor_image(
    axis: plt.Axes,
    image: Tensor,
) -> None:
    """Hiển thị tensor ảnh [C, H, W] trên một matplotlib axis."""

    if image.ndim != 3:
        raise ValueError(
            "plot_tensor_image yêu cầu tensor [C, H, W], "
            f"nhận được {tuple(image.shape)}."
        )

    channels = int(image.shape[0])

    if channels == 1:
        image_array = (
            image.squeeze(0).numpy()
        )

        axis.imshow(
            image_array,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        return

    if channels in {3, 4}:
        image_array = (
            image.permute(
                1,
                2,
                0,
            ).numpy()
        )

        axis.imshow(image_array)
        return

    raise ValueError(
        "Matplotlib visualization hiện chỉ hỗ trợ "
        f"1, 3 hoặc 4 channel, nhận được {channels}."
    )


def check_expected_value_range(
    statistics: dict[str, Any],
    transform_config: ImageTransformConfig,
) -> dict[str, Any]:
    """Kiểm tra miền tensor có phù hợp normalization hay không."""

    channel_statistics = (
        statistics["channel_statistics"]
    )

    actual_min = min(
        channel_statistics["min"]
    )

    actual_max = max(
        channel_statistics["max"]
    )

    tolerance = 1e-4

    expected_min: Optional[float] = None
    expected_max: Optional[float] = None

    if (
        transform_config.normalization
        == "zero_to_one"
    ):
        expected_min = 0.0
        expected_max = 1.0

    elif (
        transform_config.normalization
        == "minus_one_to_one"
    ):
        expected_min = -1.0
        expected_max = 1.0

    if (
        expected_min is None
        or expected_max is None
    ):
        return {
            "status": "not_checked",
            "reason": (
                "Normalization hiện tại không có miền "
                "đầu ra cố định đơn giản."
            ),
            "actual_min": actual_min,
            "actual_max": actual_max,
        }

    passed = (
        actual_min
        >= expected_min - tolerance
        and actual_max
        <= expected_max + tolerance
    )

    return {
        "status": (
            "passed"
            if passed
            else "failed"
        ),
        "expected_range": [
            expected_min,
            expected_max,
        ],
        "actual_range": [
            actual_min,
            actual_max,
        ],
    }


def resolve_output_directory(
    project_config: ProjectConfig,
    command_line_output: Optional[str],
) -> Path:
    """Xác định thư mục lưu kết quả kiểm tra."""

    if command_line_output is not None:
        path = Path(
            command_line_output
        ).expanduser()

        if not path.is_absolute():
            path = (
                project_config.project_root
                / path
            )

        return path.resolve()

    configured_output = (
        project_config.logging.get(
            "output_dir"
        )
    )

    if configured_output is not None:
        return (
            Path(str(configured_output))
            / "data_inspection"
        ).resolve()

    return (
        project_config.project_root
        / "outputs"
        / project_config.experiment_name
        / "data_inspection"
    ).resolve()


def save_json_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Lưu báo cáo dưới dạng JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )


def print_pipeline_summary(
    project_config: ProjectConfig,
    dataloaders: DataLoaderBundle,
    *,
    split: str,
    statistics: dict[str, Any],
    range_check: dict[str, Any],
    figure_path: Path,
    report_path: Path,
) -> None:
    """In kết quả kiểm tra ra terminal."""

    pipeline_summary = (
        dataloaders.summary()
    )

    channel_statistics = (
        statistics["channel_statistics"]
    )

    print("=" * 72)
    print("VANILLA VAE — DATA PIPELINE INSPECTION")
    print("=" * 72)

    print("\n[Experiment]")
    print(
        "Name:",
        project_config.experiment_name,
    )
    print(
        "Seed:",
        project_config.seed,
    )
    print(
        "Config:",
        project_config.config_path,
    )

    print("\n[Dataset]")
    print(
        "Root:",
        pipeline_summary["root"],
    )
    print(
        "Source mode:",
        pipeline_summary["source_mode"],
    )
    print(
        "Split sizes:",
        pipeline_summary["split_sizes"],
    )
    print(
        "Selected split:",
        split,
    )

    print("\n[Transform]")
    print(
        "Input size:",
        pipeline_summary["input_size"],
    )
    print(
        "Channels:",
        pipeline_summary["in_channels"],
    )
    print(
        "Resize mode:",
        pipeline_summary["resize_mode"],
    )
    print(
        "Normalization:",
        pipeline_summary["normalization"],
    )

    print("\n[Inspected data]")
    print(
        "Batches:",
        statistics["inspected_batches"],
    )
    print(
        "Samples:",
        statistics["inspected_samples"],
    )
    print(
        "First batch shape:",
        statistics["batch_shape"],
    )
    print(
        "Tensor dtype:",
        statistics["dtype"],
    )

    print("\n[Channel statistics]")
    print(
        "Min:",
        format_float_list(
            channel_statistics["min"]
        ),
    )
    print(
        "Max:",
        format_float_list(
            channel_statistics["max"]
        ),
    )
    print(
        "Mean:",
        format_float_list(
            channel_statistics["mean"]
        ),
    )
    print(
        "Std:",
        format_float_list(
            channel_statistics["std"]
        ),
    )

    print("\n[Value-range check]")
    print(
        json.dumps(
            range_check,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n[Image sizes]")
    print(
        "Original:",
        statistics[
            "original_size_distribution"
        ],
    )
    print(
        "Processed:",
        statistics[
            "processed_size_distribution"
        ],
    )

    print("\n[Outputs]")
    print(
        "Visualization:",
        figure_path,
    )
    print(
        "JSON report:",
        report_path,
    )

    print("=" * 72)


def tensor_to_float_list(
    tensor: Tensor,
) -> list[float]:
    """Chuyển tensor một chiều thành list float."""

    return [
        float(value)
        for value in tensor.detach().cpu().tolist()
    ]


def size_counter_to_dict(
    counter: Counter[tuple[int, int]],
) -> dict[str, int]:
    """Chuyển Counter kích thước thành dictionary JSON-friendly."""

    return {
        f"{height}x{width}": count
        for (
            height,
            width,
        ), count in sorted(
            counter.items()
        )
    }


def format_float_list(
    values: list[float],
) -> list[str]:
    """Format list số thực để in dễ đọc."""

    return [
        f"{value:.6f}"
        for value in values
    ]


def main() -> None:
    """Entry point của script."""

    arguments = parse_arguments()

    project_config = load_project_config(
        arguments.config
    )

    dataloaders = build_dataloaders(
        project_config.data,
        seed=project_config.seed,
    )

    selected_loader = (
        dataloaders.get_loader(
            arguments.split
        )
    )

    statistics, first_batch = (
        inspect_dataloader(
            selected_loader,
            num_batches=arguments.num_batches,
        )
    )

    range_check = (
        check_expected_value_range(
            statistics,
            dataloaders.transform_config,
        )
    )

    output_directory = (
        resolve_output_directory(
            project_config,
            arguments.output_dir,
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_path = (
        output_directory
        / f"{arguments.split}_batch.png"
    )

    report_path = (
        output_directory
        / f"{arguments.split}_inspection.json"
    )

    visualize_batch(
        first_batch,
        dataloaders.transform_config,
        num_images=arguments.num_images,
        output_path=figure_path,
        show=arguments.show,
    )

    report = {
        "experiment": {
            "name": (
                project_config.experiment_name
            ),
            "seed": project_config.seed,
            "config_path": str(
                project_config.config_path
            ),
        },
        "pipeline": dataloaders.summary(),
        "selected_split": arguments.split,
        "statistics": statistics,
        "value_range_check": range_check,
        "outputs": {
            "visualization": str(
                figure_path
            ),
            "report": str(
                report_path
            ),
        },
    }

    save_json_report(
        report,
        report_path,
    )

    print_pipeline_summary(
        project_config,
        dataloaders,
        split=arguments.split,
        statistics=statistics,
        range_check=range_check,
        figure_path=figure_path,
        report_path=report_path,
    )


if __name__ == "__main__":
    main()

    
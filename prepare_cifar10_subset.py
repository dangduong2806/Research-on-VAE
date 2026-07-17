"""Tải CIFAR-10 và xuất một tập con thành các file PNG.

Cấu trúc đầu ra:

    data/images/
    ├── train/
    │   ├── airplane/
    │   ├── automobile/
    │   ├── bird/
    │   └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...

Cách chạy:

    python prepare_cifar10_subset.py

Hoặc:

    python prepare_cifar10_subset.py \
        --train-count 500 \
        --val-count 100 \
        --test-count 100
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Sequence

from torchvision.datasets import CIFAR10


def parse_arguments() -> argparse.Namespace:
    """Đọc tham số command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Tải CIFAR-10 và xuất một tập con "
            "thành các file ảnh PNG."
        )
    )

    parser.add_argument(
        "--download-root",
        type=str,
        default="data/raw",
        help=(
            "Thư mục lưu dữ liệu CIFAR-10 gốc. "
            "Mặc định: data/raw"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="data/images",
        help=(
            "Thư mục chứa các ảnh PNG đã xuất. "
            "Mặc định: data/images"
        ),
    )

    parser.add_argument(
        "--train-count",
        type=int,
        default=500,
        help="Số ảnh train cần xuất.",
    )

    parser.add_argument(
        "--val-count",
        type=int,
        default=100,
        help="Số ảnh validation cần xuất.",
    )

    parser.add_argument(
        "--test-count",
        type=int,
        default=100,
        help="Số ảnh test cần xuất.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed khi chọn ảnh.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Xóa thư mục output cũ trước khi xuất dữ liệu mới."
        ),
    )

    arguments = parser.parse_args()

    for argument_name in (
        "train_count",
        "val_count",
        "test_count",
    ):
        value = getattr(
            arguments,
            argument_name,
        )

        if value <= 0:
            parser.error(
                f"--{argument_name.replace('_', '-')} "
                "phải lớn hơn 0."
            )

    return arguments


def select_indices(
    dataset_size: int,
    *,
    count: int,
    seed: int,
) -> list[int]:
    """Chọn ngẫu nhiên các index không lặp lại."""

    if count > dataset_size:
        raise ValueError(
            f"Yêu cầu {count} ảnh nhưng dataset chỉ có "
            f"{dataset_size} ảnh."
        )

    indices = list(
        range(dataset_size)
    )

    random_generator = random.Random(seed)
    random_generator.shuffle(indices)

    return indices[:count]


def split_train_validation_indices(
    dataset_size: int,
    *,
    train_count: int,
    val_count: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Chọn train và validation không bị trùng nhau."""

    required_count = (
        train_count + val_count
    )

    if required_count > dataset_size:
        raise ValueError(
            "Tổng train_count và val_count vượt quá "
            f"kích thước train dataset: "
            f"{required_count} > {dataset_size}."
        )

    indices = list(
        range(dataset_size)
    )

    random_generator = random.Random(seed)
    random_generator.shuffle(indices)

    train_indices = indices[
        :train_count
    ]

    val_indices = indices[
        train_count:required_count
    ]

    return (
        train_indices,
        val_indices,
    )


def export_dataset_subset(
    dataset: CIFAR10,
    indices: Sequence[int],
    *,
    output_root: Path,
    split: str,
) -> dict[str, int]:
    """Xuất một số ảnh của CIFAR-10 thành PNG.

    Mỗi ảnh được lưu trong thư mục class tương ứng:

        output_root/split/class_name/image.png
    """

    class_counts = {
        class_name: 0
        for class_name in dataset.classes
    }

    for output_index, dataset_index in enumerate(
        indices
    ):
        image, label = dataset[
            dataset_index
        ]

        class_name = dataset.classes[
            int(label)
        ]

        class_directory = (
            output_root
            / split
            / class_name
        )

        class_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            class_directory
            / (
                f"{split}_"
                f"{output_index:05d}_"
                f"source_{dataset_index:05d}.png"
            )
        )

        image.save(
            output_path,
            format="PNG",
        )

        class_counts[class_name] += 1

    return class_counts


def prepare_output_directory(
    output_root: Path,
    *,
    overwrite: bool,
) -> None:
    """Chuẩn bị thư mục output."""

    if output_root.exists():
        existing_images = list(
            output_root.rglob("*.png")
        )

        if existing_images and not overwrite:
            raise FileExistsError(
                f"Thư mục {output_root} đã chứa "
                f"{len(existing_images)} ảnh PNG. "
                "Dùng --overwrite để tạo lại."
            )

        if overwrite:
            shutil.rmtree(
                output_root
            )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )


def print_split_summary(
    split: str,
    class_counts: dict[str, int],
) -> None:
    """In số ảnh đã xuất theo từng class."""

    total = sum(
        class_counts.values()
    )

    print(
        f"\n[{split}] Tổng số ảnh: {total}"
    )

    for class_name, count in sorted(
        class_counts.items()
    ):
        print(
            f"  {class_name:<12}: {count}"
        )


def main() -> None:
    """Tải và chuẩn bị tập con CIFAR-10."""

    arguments = parse_arguments()

    download_root = Path(
        arguments.download_root
    ).expanduser().resolve()

    output_root = Path(
        arguments.output_root
    ).expanduser().resolve()

    prepare_output_directory(
        output_root,
        overwrite=arguments.overwrite,
    )

    print(
        "Đang tải hoặc đọc CIFAR-10..."
    )

    train_dataset = CIFAR10(
        root=download_root,
        train=True,
        download=True,
    )

    test_dataset = CIFAR10(
        root=download_root,
        train=False,
        download=True,
    )

    (
        train_indices,
        val_indices,
    ) = split_train_validation_indices(
        len(train_dataset),
        train_count=arguments.train_count,
        val_count=arguments.val_count,
        seed=arguments.seed,
    )

    test_indices = select_indices(
        len(test_dataset),
        count=arguments.test_count,
        seed=arguments.seed + 1,
    )

    train_counts = export_dataset_subset(
        train_dataset,
        train_indices,
        output_root=output_root,
        split="train",
    )

    val_counts = export_dataset_subset(
        train_dataset,
        val_indices,
        output_root=output_root,
        split="val",
    )

    test_counts = export_dataset_subset(
        test_dataset,
        test_indices,
        output_root=output_root,
        split="test",
    )

    print_split_summary(
        "train",
        train_counts,
    )

    print_split_summary(
        "val",
        val_counts,
    )

    print_split_summary(
        "test",
        test_counts,
    )

    print(
        "\nHoàn thành."
    )

    print(
        "Ảnh đã được lưu tại:",
        output_root,
    )


if __name__ == "__main__":
    main()
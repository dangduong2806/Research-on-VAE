"""Unit test cho src/data/image_dataset.py.

Các test trong file này kiểm tra:

1. Khám phá file ảnh.
2. Bỏ qua file không phải ảnh.
3. Đọc ảnh theo cơ chế lazy loading.
4. Metadata kích thước gốc và kích thước xử lý.
5. Label theo cấu trúc thư mục.
6. Class mapping nhất quán.
7. Xử lý ảnh bị hỏng.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.image_dataset import (
    ImageReadError,
    ImageVAEDataset,
    discover_image_paths,
    discover_image_records,
)
from src.data.transforms import (
    ImageTransformConfig,
    build_image_transform,
)


def save_test_image(
    path: Path,
    *,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    """Tạo và lưu một ảnh RGB dùng trong test."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = Image.new(
        mode="RGB",
        size=(width, height),
        color=color,
    )

    image.save(path)


def build_test_transform(
    *,
    input_size: tuple[int, int] = (
        32,
        48,
    ),
):
    """Tạo transform đơn giản dùng chung trong file test."""

    config = ImageTransformConfig(
        input_size=input_size,
        in_channels=3,
        resize_mode="resize_and_pad",
        normalization="minus_one_to_one",
    )

    return build_image_transform(
        config,
        train=False,
    )


def test_discover_image_paths_filters_files(
    tmp_path: Path,
) -> None:
    """Chỉ file ảnh hợp lệ mới được tìm thấy."""

    direct_image = (
        tmp_path
        / "image_a.jpg"
    )

    nested_image = (
        tmp_path
        / "nested"
        / "image_b.png"
    )

    text_file = (
        tmp_path
        / "notes.txt"
    )

    save_test_image(
        direct_image,
        width=20,
        height=10,
        color=(255, 0, 0),
    )

    save_test_image(
        nested_image,
        width=30,
        height=15,
        color=(0, 255, 0),
    )

    text_file.write_text(
        "Đây không phải ảnh.",
        encoding="utf-8",
    )

    recursive_paths = discover_image_paths(
        tmp_path,
        recursive=True,
    )

    assert recursive_paths == sorted(
        [
            direct_image.resolve(),
            nested_image.resolve(),
        ]
    )

    non_recursive_paths = discover_image_paths(
        tmp_path,
        recursive=False,
    )

    assert non_recursive_paths == [
        direct_image.resolve()
    ]


def test_dataset_returns_tensor_and_metadata(
    tmp_path: Path,
) -> None:
    """Dataset phải trả ImageSample đúng shape và metadata."""

    image_path = (
        tmp_path
        / "sample.jpg"
    )

    save_test_image(
        image_path,
        width=60,
        height=40,
        color=(100, 150, 200),
    )

    records, class_to_idx = (
        discover_image_records(
            tmp_path,
            label_mode="none",
        )
    )

    dataset = ImageVAEDataset(
        records=records,
        transform=build_test_transform(
            input_size=(32, 48),
        ),
    )

    sample = dataset[0]

    assert len(dataset) == 1
    assert class_to_idx == {}

    assert sample.path == str(
        image_path.resolve()
    )

    # Project lưu kích thước theo thứ tự (H, W).
    assert sample.original_size == (
        40,
        60,
    )

    assert sample.processed_size == (
        32,
        48,
    )

    assert sample.image.shape == (
        3,
        32,
        48,
    )

    assert sample.image.dtype == torch.float32
    assert sample.label is None
    assert sample.class_name is None

    assert dataset.has_labels is False
    assert dataset.class_names == ()

    sample.validate()


def test_parent_folder_label_mapping_is_deterministic(
    tmp_path: Path,
) -> None:
    """Tên class phải được sắp xếp alphabet trước khi gán label."""

    save_test_image(
        tmp_path
        / "dog"
        / "dog_1.jpg",
        width=20,
        height=20,
        color=(0, 255, 0),
    )

    save_test_image(
        tmp_path
        / "cat"
        / "cat_1.jpg",
        width=20,
        height=20,
        color=(255, 0, 0),
    )

    records, class_to_idx = (
        discover_image_records(
            tmp_path,
            label_mode="parent_folder",
        )
    )

    assert class_to_idx == {
        "cat": 0,
        "dog": 1,
    }

    label_by_class = {
        record.class_name: record.label
        for record in records
    }

    assert label_by_class == {
        "cat": 0,
        "dog": 1,
    }

    dataset = ImageVAEDataset(
        records=records,
        transform=build_test_transform(),
    )

    assert dataset.has_labels is True

    assert dataset.class_names == (
        "cat",
        "dog",
    )


def test_validation_reuses_training_class_mapping(
    tmp_path: Path,
) -> None:
    """Validation phải dùng đúng mapping được tạo từ train."""

    train_root = tmp_path / "train"
    val_root = tmp_path / "val"

    save_test_image(
        train_root
        / "cat"
        / "cat.jpg",
        width=20,
        height=20,
        color=(255, 0, 0),
    )

    save_test_image(
        train_root
        / "dog"
        / "dog.jpg",
        width=20,
        height=20,
        color=(0, 255, 0),
    )

    # Validation chỉ có class dog.
    save_test_image(
        val_root
        / "dog"
        / "dog_val.jpg",
        width=20,
        height=20,
        color=(0, 0, 255),
    )

    _, train_mapping = (
        discover_image_records(
            train_root,
            label_mode="parent_folder",
        )
    )

    val_records, _ = (
        discover_image_records(
            val_root,
            label_mode="parent_folder",
            class_to_idx=train_mapping,
        )
    )

    assert train_mapping == {
        "cat": 0,
        "dog": 1,
    }

    assert len(val_records) == 1
    assert val_records[0].class_name == "dog"
    assert val_records[0].label == 1


def test_unknown_validation_class_raises_error(
    tmp_path: Path,
) -> None:
    """Validation không được xuất hiện class chưa có trong train."""

    train_root = tmp_path / "train"
    val_root = tmp_path / "val"

    save_test_image(
        train_root
        / "cat"
        / "cat.jpg",
        width=20,
        height=20,
        color=(255, 0, 0),
    )

    save_test_image(
        val_root
        / "bird"
        / "bird.jpg",
        width=20,
        height=20,
        color=(0, 0, 255),
    )

    _, train_mapping = (
        discover_image_records(
            train_root,
            label_mode="parent_folder",
        )
    )

    with pytest.raises(
        ValueError,
        match="class không tồn tại",
    ):
        discover_image_records(
            val_root,
            label_mode="parent_folder",
            class_to_idx=train_mapping,
        )


def test_corrupted_image_raises_image_read_error(
    tmp_path: Path,
) -> None:
    """File có extension .jpg nhưng nội dung sai phải báo lỗi rõ."""

    corrupted_path = (
        tmp_path
        / "corrupted.jpg"
    )

    corrupted_path.write_text(
        "Đây không phải dữ liệu ảnh.",
        encoding="utf-8",
    )

    records, _ = discover_image_records(
        tmp_path,
        label_mode="none",
    )

    dataset = ImageVAEDataset(
        records=records,
        transform=build_test_transform(),
    )

    with pytest.raises(
        ImageReadError,
        match="không phải ảnh hợp lệ",
    ):
        _ = dataset[0]

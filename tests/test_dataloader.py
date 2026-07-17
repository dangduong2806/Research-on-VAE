"""Integration test cho src/data/dataloader.py.

Các test trong file này kiểm tra:

1. Tính số lượng train/val/test.
2. Random split có tính tái lập.
3. DataLoader trả ImageBatch đúng định dạng.
4. Existing split hoạt động đúng.
5. Class mapping nhất quán giữa các split.
6. Auto mode chọn đúng kiểu dữ liệu.
7. num_workers=0 xử lý đúng các worker options.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.data.batch import ImageBatch
from src.data.dataloader import (
    build_dataloaders,
    calculate_split_counts,
)


def save_test_image(
    path: Path,
    *,
    width: int = 40,
    height: int = 30,
    color: tuple[int, int, int] = (
        100,
        150,
        200,
    ),
) -> None:
    """Tạo một ảnh RGB phục vụ DataLoader test."""

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


def create_flat_image_dataset(
    root: Path,
    *,
    count: int,
) -> None:
    """Tạo nhiều ảnh nằm trực tiếp trong một thư mục."""

    for index in range(count):
        save_test_image(
            root
            / f"image_{index:03d}.png",
            width=40 + index,
            height=30 + index,
            color=(
                index % 256,
                (index * 2) % 256,
                (index * 3) % 256,
            ),
        )


def build_base_data_mapping(
    root: Path,
) -> dict[str, Any]:
    """Cấu hình tối thiểu dùng chung trong các test."""

    return {
        "root": str(root),

        "input_size": [32, 48],
        "in_channels": 3,

        "resize_mode": "resize_and_pad",
        "normalization": "minus_one_to_one",

        "split_mode": "random",
        "split_ratios": [
            0.5,
            0.25,
            0.25,
        ],

        "train_batch_size": 4,
        "val_batch_size": 4,
        "test_batch_size": 4,

        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,

        "shuffle_train": True,
        "drop_last_train": False,

        "recursive": True,
        "label_mode": "none",
    }


def dataset_paths(
    dataset,
) -> tuple[str, ...]:
    """Chuyển danh sách Path của Dataset thành tuple string."""

    return tuple(
        str(path)
        for path in dataset.paths
    )


def test_calculate_split_counts_distributes_all_samples() -> None:
    """Số lượng split phải cộng lại đúng bằng tổng dataset."""

    counts = calculate_split_counts(
        total_size=11,
        split_ratios=[
            0.8,
            0.1,
            0.1,
        ],
    )

    assert counts == (
        9,
        1,
        1,
    )

    assert sum(counts) == 11


def test_random_dataloader_returns_image_batch(
    tmp_path: Path,
) -> None:
    """Pipeline random split phải trả đúng ImageBatch."""

    create_flat_image_dataset(
        tmp_path,
        count=12,
    )

    config = build_base_data_mapping(
        tmp_path
    )

    bundle = build_dataloaders(
        config,
        seed=42,
    )

    assert bundle.datasets.source_mode == "random"

    assert bundle.datasets.split_sizes == {
        "train": 6,
        "val": 3,
        "test": 3,
    }

    batch = next(
        iter(bundle.train)
    )

    assert isinstance(
        batch,
        ImageBatch,
    )

    assert batch.shape == (
        4,
        3,
        32,
        48,
    )

    assert batch.images.dtype == torch.float32

    assert batch.original_sizes.shape == (
        4,
        2,
    )

    assert batch.processed_sizes.shape == (
        4,
        2,
    )

    expected_processed_sizes = torch.tensor(
        [
            [32, 48],
            [32, 48],
            [32, 48],
            [32, 48],
        ],
        dtype=torch.long,
    )

    assert torch.equal(
        batch.processed_sizes,
        expected_processed_sizes,
    )

    assert batch.labels is None

    assert len(batch.paths) == 4
    assert len(batch.class_names) == 4

    assert batch.images.min().item() >= -1.0
    assert batch.images.max().item() <= 1.0


def test_random_split_is_reproducible(
    tmp_path: Path,
) -> None:
    """Cùng seed phải tạo cùng train/val/test split."""

    create_flat_image_dataset(
        tmp_path,
        count=15,
    )

    config = build_base_data_mapping(
        tmp_path
    )

    config["split_ratios"] = [
        0.6,
        0.2,
        0.2,
    ]

    first_bundle = build_dataloaders(
        config,
        seed=123,
    )

    second_bundle = build_dataloaders(
        config,
        seed=123,
    )

    assert dataset_paths(
        first_bundle.datasets.train
    ) == dataset_paths(
        second_bundle.datasets.train
    )

    assert first_bundle.datasets.val is not None
    assert second_bundle.datasets.val is not None

    assert dataset_paths(
        first_bundle.datasets.val
    ) == dataset_paths(
        second_bundle.datasets.val
    )

    assert first_bundle.datasets.test is not None
    assert second_bundle.datasets.test is not None

    assert dataset_paths(
        first_bundle.datasets.test
    ) == dataset_paths(
        second_bundle.datasets.test
    )


def test_existing_split_uses_shared_class_mapping(
    tmp_path: Path,
) -> None:
    """Train, val và test phải dùng cùng class_to_idx."""

    for split in [
        "train",
        "val",
        "test",
    ]:
        save_test_image(
            tmp_path
            / split
            / "cat"
            / f"{split}_cat.jpg",
            color=(255, 0, 0),
        )

        save_test_image(
            tmp_path
            / split
            / "dog"
            / f"{split}_dog.jpg",
            color=(0, 255, 0),
        )

    config = build_base_data_mapping(
        tmp_path
    )

    config.update(
        {
            "split_mode": "existing",
            "label_mode": "parent_folder",
            "train_batch_size": 2,
            "val_batch_size": 2,
            "test_batch_size": 2,
            "shuffle_train": False,
        }
    )

    bundle = build_dataloaders(
        config,
        seed=42,
    )

    assert bundle.datasets.source_mode == "existing"

    assert bundle.datasets.class_to_idx == {
        "cat": 0,
        "dog": 1,
    }

    assert bundle.datasets.split_sizes == {
        "train": 2,
        "val": 2,
        "test": 2,
    }

    assert bundle.val is not None
    assert bundle.test is not None

    val_batch = next(
        iter(bundle.val)
    )

    assert val_batch.labels is not None

    observed_mapping = {
        class_name: int(label)
        for class_name, label in zip(
            val_batch.class_names,
            val_batch.labels.tolist(),
        )
    }

    assert observed_mapping == {
        "cat": 0,
        "dog": 1,
    }


def test_auto_mode_prefers_existing_train_directory(
    tmp_path: Path,
) -> None:
    """Auto mode phải chọn existing nếu thư mục train tồn tại."""

    train_root = tmp_path / "train"

    save_test_image(
        train_root
        / "image_1.jpg",
    )

    save_test_image(
        train_root
        / "image_2.jpg",
    )

    config = build_base_data_mapping(
        tmp_path
    )

    config["split_mode"] = "auto"

    bundle = build_dataloaders(
        config,
        seed=42,
    )

    assert bundle.datasets.source_mode == "existing"

    assert bundle.datasets.split_sizes == {
        "train": 2,
        "val": 0,
        "test": 0,
    }

    assert bundle.val is None
    assert bundle.test is None


def test_worker_only_options_are_ignored_when_num_workers_is_zero(
    tmp_path: Path,
) -> None:
    """persistent_workers/prefetch không được gây lỗi khi workers=0."""

    create_flat_image_dataset(
        tmp_path,
        count=8,
    )

    config = build_base_data_mapping(
        tmp_path
    )

    config.update(
        {
            "num_workers": 0,
            "persistent_workers": True,
            "prefetch_factor": 4,
        }
    )

    bundle = build_dataloaders(
        config,
        seed=42,
    )

    assert bundle.train.num_workers == 0

    # Vì build_single_dataloader không truyền persistent_workers
    # khi num_workers=0 nên DataLoader giữ giá trị mặc định False.
    assert bundle.train.persistent_workers is False

    batch = next(
        iter(bundle.train)
    )

    assert isinstance(
        batch,
        ImageBatch,
    )
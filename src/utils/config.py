"""Đọc và kiểm tra cấu hình YAML của project Vanilla VAE.

File cấu hình dự kiến có dạng:

    experiment:
      name: vanilla_vae_image_128
      seed: 42
      device: auto

    data:
      root: data/images
      input_size: [128, 128]
      in_channels: 3
      ...

    model:
      ...

    loss:
      ...

    optimizer:
      ...

    training:
      ...

    logging:
      ...

Trong giai đoạn Data pipeline, hai section bắt buộc là:

    experiment
    data

Các section model, loss, optimizer và training sẽ được sử dụng
ở những giai đoạn tiếp theo.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


# Những section mà toàn project dự kiến sẽ sử dụng.
KNOWN_CONFIG_SECTIONS = {
    "experiment",
    "data",
    "model",
    "loss",
    "optimizer",
    "training",
    "logging",
}


class ConfigError(ValueError):
    """Lỗi liên quan đến nội dung hoặc cấu trúc file cấu hình."""


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Cấu hình chung của một thí nghiệm.

    Attributes:
        name:
            Tên thí nghiệm.

        seed:
            Random seed dùng cho:

            - chia train/val/test;
            - DataLoader shuffle;
            - random crop;
            - khởi tạo model;
            - quá trình training.

        device:
            Thiết bị chạy thí nghiệm.

            Các giá trị thường dùng:

            - auto
            - cpu
            - cuda
            - cuda:0
            - mps
    """

    name: str
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        """Kiểm tra cấu hình experiment."""

        if not self.name.strip():
            raise ConfigError(
                "experiment.name không được để trống."
            )

        if self.seed < 0:
            raise ConfigError(
                "experiment.seed phải lớn hơn hoặc bằng 0, "
                f"nhận được {self.seed}."
            )

        if not self.device.strip():
            raise ConfigError(
                "experiment.device không được để trống."
            )

    @classmethod
    def from_mapping(
        cls,
        experiment_mapping: Mapping[str, Any],
    ) -> "ExperimentConfig":
        """Tạo ExperimentConfig từ dictionary."""

        return cls(
            name=str(
                experiment_mapping.get(
                    "name",
                    "vanilla_vae_experiment",
                )
            ),
            seed=int(
                experiment_mapping.get(
                    "seed",
                    42,
                )
            ),
            device=str(
                experiment_mapping.get(
                    "device",
                    "auto",
                )
            ),
        )


@dataclass(slots=True)
class ProjectConfig:
    """Toàn bộ cấu hình đã được đọc và chuẩn hóa.

    Attributes:
        config_path:
            Đường dẫn tuyệt đối đến file YAML.

        project_root:
            Thư mục gốc của project.

        experiment:
            Cấu hình thí nghiệm đã được parse thành dataclass.

        data:
            Dictionary cấu hình Data pipeline.

        model:
            Dictionary cấu hình model.

        loss:
            Dictionary cấu hình loss.

        optimizer:
            Dictionary cấu hình optimizer.

        training:
            Dictionary cấu hình quá trình train.

        logging:
            Dictionary cấu hình output và logging.

        raw:
            Toàn bộ nội dung cấu hình sau khi chuẩn hóa.
    """

    config_path: Path
    project_root: Path

    experiment: ExperimentConfig

    data: dict[str, Any]
    model: dict[str, Any]
    loss: dict[str, Any]
    optimizer: dict[str, Any]
    training: dict[str, Any]
    logging: dict[str, Any]

    raw: dict[str, Any]

    @property
    def seed(self) -> int:
        """Lấy nhanh random seed."""

        return self.experiment.seed

    @property
    def experiment_name(self) -> str:
        """Lấy nhanh tên thí nghiệm."""

        return self.experiment.name

    def get_section(
        self,
        section_name: str,
    ) -> dict[str, Any]:
        """Lấy một section cấu hình theo tên.

        Args:
            section_name:
                Tên section, ví dụ:

                - data
                - model
                - loss
                - optimizer
                - training
                - logging

        Returns:
            Một bản copy của section.

        Việc trả về bản copy giúp code bên ngoài không vô tình
        thay đổi cấu hình đang được lưu trong ProjectConfig.
        """

        if section_name == "experiment":
            return {
                "name": self.experiment.name,
                "seed": self.experiment.seed,
                "device": self.experiment.device,
            }

        section_mapping = {
            "data": self.data,
            "model": self.model,
            "loss": self.loss,
            "optimizer": self.optimizer,
            "training": self.training,
            "logging": self.logging,
        }

        if section_name not in section_mapping:
            raise KeyError(
                f"Không tồn tại config section "
                f"'{section_name}'."
            )

        return deepcopy(
            section_mapping[section_name]
        )

    def as_dict(self) -> dict[str, Any]:
        """Trả về toàn bộ config dưới dạng dictionary."""

        return deepcopy(self.raw)

    def summary(self) -> dict[str, Any]:
        """Tạo thông tin tóm tắt để debug."""

        return {
            "config_path": str(self.config_path),
            "project_root": str(self.project_root),
            "experiment_name": self.experiment.name,
            "seed": self.experiment.seed,
            "device": self.experiment.device,
            "data_root": self.data.get("root"),
            "input_size": self.data.get("input_size"),
            "in_channels": self.data.get("in_channels"),
        }


def load_project_config(
    config_path: str | Path,
    *,
    project_root: Optional[str | Path] = None,
) -> ProjectConfig:
    """Đọc và chuẩn hóa toàn bộ cấu hình project.

    Args:
        config_path:
            Đường dẫn đến file YAML.

        project_root:
            Thư mục gốc của project.

            Nếu không truyền, hàm sẽ tự suy luận:

            - Nếu config nằm trong thư mục configs/:
                  project_root = thư mục cha của configs/

            - Nếu không:
                  project_root = thư mục chứa file config

    Returns:
        ProjectConfig hoàn chỉnh.

    Ví dụ:
        config = load_project_config(
            "configs/image_128.yaml"
        )

        print(config.seed)
        print(config.data)

        loaders = build_dataloaders(
            config.data,
            seed=config.seed,
        )
    """

    resolved_config_path = _resolve_config_path(
        config_path
    )

    resolved_project_root = _resolve_project_root(
        config_path=resolved_config_path,
        project_root=project_root,
    )

    raw_config = _load_yaml_mapping(
        resolved_config_path
    )

    _validate_top_level_config(raw_config)

    normalized_config = _normalize_config_paths(
        raw_config,
        project_root=resolved_project_root,
    )

    experiment_mapping = _get_required_mapping(
        normalized_config,
        "experiment",
    )

    data_mapping = _get_required_mapping(
        normalized_config,
        "data",
    )

    experiment_config = (
        ExperimentConfig.from_mapping(
            experiment_mapping
        )
    )

    return ProjectConfig(
        config_path=resolved_config_path,
        project_root=resolved_project_root,
        experiment=experiment_config,
        data=deepcopy(data_mapping),
        model=_get_optional_mapping(
            normalized_config,
            "model",
        ),
        loss=_get_optional_mapping(
            normalized_config,
            "loss",
        ),
        optimizer=_get_optional_mapping(
            normalized_config,
            "optimizer",
        ),
        training=_get_optional_mapping(
            normalized_config,
            "training",
        ),
        logging=_get_optional_mapping(
            normalized_config,
            "logging",
        ),
        raw=deepcopy(normalized_config),
    )


def _resolve_config_path(
    config_path: str | Path,
) -> Path:
    """Kiểm tra và chuẩn hóa đường dẫn file cấu hình."""

    path = Path(config_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file cấu hình: {path}"
        )

    if not path.is_file():
        raise ConfigError(
            f"Đường dẫn cấu hình không phải file: {path}"
        )

    if path.suffix.lower() not in {
        ".yaml",
        ".yml",
    }:
        raise ConfigError(
            "File cấu hình phải có extension .yaml hoặc .yml, "
            f"nhận được '{path.suffix}'."
        )

    return path


def _resolve_project_root(
    *,
    config_path: Path,
    project_root: Optional[str | Path],
) -> Path:
    """Xác định thư mục gốc của project.

    Ví dụ:

        project/
        ├── configs/
        │   └── image_128.yaml
        ├── data/
        └── src/

    Với config_path:

        project/configs/image_128.yaml

    project_root sẽ là:

        project/
    """

    if project_root is not None:
        resolved_root = (
            Path(project_root)
            .expanduser()
            .resolve()
        )
    elif config_path.parent.name == "configs":
        resolved_root = config_path.parent.parent
    else:
        resolved_root = config_path.parent

    if not resolved_root.exists():
        raise FileNotFoundError(
            f"Không tìm thấy project root: {resolved_root}"
        )

    if not resolved_root.is_dir():
        raise ConfigError(
            f"Project root không phải thư mục: "
            f"{resolved_root}"
        )

    return resolved_root


def _load_yaml_mapping(
    config_path: Path,
) -> dict[str, Any]:
    """Đọc YAML bằng yaml.safe_load.

    safe_load không cho phép YAML tự động tạo các Python object
    tùy ý, an toàn hơn yaml.load.
    """

    try:
        with config_path.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            loaded_data = yaml.safe_load(file)

    except yaml.YAMLError as error:
        raise ConfigError(
            f"File YAML không hợp lệ: {config_path}\n"
            f"Chi tiết: {error}"
        ) from error

    except OSError as error:
        raise ConfigError(
            f"Không thể đọc file cấu hình "
            f"{config_path}: {error}"
        ) from error

    if loaded_data is None:
        raise ConfigError(
            f"File cấu hình đang rỗng: {config_path}"
        )

    if not isinstance(loaded_data, dict):
        raise ConfigError(
            "Nội dung cấp cao nhất của file YAML phải là "
            f"dictionary, nhận được "
            f"{type(loaded_data).__name__}."
        )

    return loaded_data


def _validate_top_level_config(
    config: Mapping[str, Any],
) -> None:
    """Kiểm tra các section cấp cao nhất."""

    required_sections = {
        "experiment",
        "data",
    }

    missing_sections = (
        required_sections - config.keys()
    )

    if missing_sections:
        raise ConfigError(
            "File cấu hình thiếu các section bắt buộc: "
            f"{sorted(missing_sections)}"
        )

    unknown_sections = (
        config.keys() - KNOWN_CONFIG_SECTIONS
    )

    if unknown_sections:
        raise ConfigError(
            "Phát hiện các section cấu hình không được nhận diện: "
            f"{sorted(unknown_sections)}. "
            f"Các section hợp lệ: "
            f"{sorted(KNOWN_CONFIG_SECTIONS)}"
        )

    # Tất cả section top-level phải là dictionary.
    for section_name, section_value in config.items():
        if section_value is None:
            continue

        if not isinstance(section_value, Mapping):
            raise ConfigError(
                f"Section '{section_name}' phải là dictionary, "
                f"nhận được "
                f"{type(section_value).__name__}."
            )


def _get_required_mapping(
    config: Mapping[str, Any],
    section_name: str,
) -> dict[str, Any]:
    """Lấy một section bắt buộc."""

    if section_name not in config:
        raise ConfigError(
            f"Thiếu config section '{section_name}'."
        )

    value = config[section_name]

    if not isinstance(value, Mapping):
        raise ConfigError(
            f"Config section '{section_name}' phải là "
            f"dictionary."
        )

    return dict(value)


def _get_optional_mapping(
    config: Mapping[str, Any],
    section_name: str,
) -> dict[str, Any]:
    """Lấy một section không bắt buộc.

    Nếu section không tồn tại hoặc có giá trị null, trả về {}.
    """

    value = config.get(section_name)

    if value is None:
        return {}

    if not isinstance(value, Mapping):
        raise ConfigError(
            f"Config section '{section_name}' phải là "
            f"dictionary."
        )

    return deepcopy(dict(value))


def _normalize_config_paths(
    config: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Chuẩn hóa các đường dẫn trong cấu hình.

    Các đường dẫn tương đối được tính từ project_root, không phải
    từ thư mục configs/.

    Ví dụ YAML:

        data:
          root: data/images

    Project root:

        /home/user/vanilla_vae_research

    Kết quả:

        /home/user/vanilla_vae_research/data/images
    """

    normalized = deepcopy(dict(config))

    data_mapping = _get_required_mapping(
        normalized,
        "data",
    )

    raw_data_root = data_mapping.get(
        "root",
        "data/images",
    )

    data_mapping["root"] = str(
        _resolve_relative_path(
            raw_data_root,
            project_root=project_root,
        )
    )

    normalized["data"] = data_mapping

    logging_mapping = _get_optional_mapping(
        normalized,
        "logging",
    )

    if logging_mapping:
        raw_output_dir = logging_mapping.get(
            "output_dir",
            "outputs",
        )

        logging_mapping["output_dir"] = str(
            _resolve_relative_path(
                raw_output_dir,
                project_root=project_root,
            )
        )

        normalized["logging"] = logging_mapping

    return normalized


def _resolve_relative_path(
    raw_path: Any,
    *,
    project_root: Path,
) -> Path:
    """Chuyển một đường dẫn tương đối thành tuyệt đối."""

    if not isinstance(
        raw_path,
        (str, Path),
    ):
        raise ConfigError(
            "Giá trị đường dẫn phải là string hoặc Path, "
            f"nhận được {type(raw_path).__name__}."
        )

    path = Path(raw_path).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (
        project_root
        / path
    ).resolve()
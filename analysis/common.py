"""Các hàm dùng chung cho giai đoạn phân tích VAE."""

from pathlib import Path

import torch

from src.models.vanilla_vae import (
    VanillaVAE,
    VanillaVAEConfig,
)
from src.utils.config import load_project_config


def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def build_model(config) -> VanillaVAE:
    """Tạo model đúng theo config lúc training."""

    return VanillaVAE(
        VanillaVAEConfig(
            input_size=tuple(
                config.data["input_size"]
            ),
            in_channels=config.data[
                "in_channels"
            ],
            latent_dim=config.model[
                "latent_dim"
            ],
            hidden_dims=tuple(
                config.model[
                    "encoder"
                ]["hidden_dims"]
            ),
        )
    )


def load_checkpoint(
    model: VanillaVAE,
    checkpoint_path: str,
    device: torch.device,
) -> dict:
    """Load checkpoint vào model."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint[
            "model_state_dict"
        ]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint[
            "state_dict"
        ]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return checkpoint


def load_trained_model(
    config_path: str,
    checkpoint_path: str,
):
    """Load config and a checkpoint saved in its configured output directory."""

    config = load_project_config(
        config_path
    )

    resolved_checkpoint_path = Path(
        checkpoint_path
    )
    checkpoint_dir = (
        Path(
            config.logging.get(
                "output_dir",
                "outputs",
            )
        )
        / "checkpoints"
    )

    if not resolved_checkpoint_path.is_file():
        configured_checkpoint_path = (
            checkpoint_dir
            / resolved_checkpoint_path.name
        )

        if configured_checkpoint_path.is_file():
            resolved_checkpoint_path = (
                configured_checkpoint_path
            )

    if not resolved_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}. "
            f"Expected directory: "
            f"{checkpoint_dir}"
        )

    device = get_device()

    model = build_model(config)

    checkpoint = load_checkpoint(
        model=model,
        checkpoint_path=str(
            resolved_checkpoint_path
        ),
        device=device,
    )

    return config, model, checkpoint, device

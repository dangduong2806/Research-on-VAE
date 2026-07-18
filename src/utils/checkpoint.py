"""Checkpoint saving and loading utilities."""
from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_val_loss,
    epochs_without_improvement,
    history,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": (
                epochs_without_improvement
            ),
            "history": history,
        },
        path,
    )

def load_checkpoint(
    path,
    model,
    optimizer,
    device,
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    return checkpoint
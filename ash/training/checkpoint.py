from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from ash.config import ModelConfig, TrainingConfig


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_config: ModelConfig,
    train_config: TrainingConfig,
    metrics: dict,
    path: str | Path,
) -> None:
    """Save full training state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Load checkpoint. Returns the full checkpoint dict."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt

from __future__ import annotations

import torch

from ash.config import TrainingConfig
from ash.model.gpt import GPT


def configure_optimizer(model: GPT, config: TrainingConfig) -> torch.optim.AdamW:
    """Configure AdamW with weight-decay parameter groups."""
    return model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device=config.device,
    )

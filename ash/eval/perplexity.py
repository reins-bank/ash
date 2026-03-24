from __future__ import annotations

import math

import torch

from ash.model.gpt import GPT


@torch.no_grad()
def evaluate_perplexity(
    model: GPT, dataloader, max_batches: int = 200, device: str = "cpu"
) -> float:
    """Compute perplexity on a dataset."""
    model.eval()
    losses = []

    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        _, loss, _ = model(x, y)
        losses.append(loss.item())

    mean_loss = sum(losses) / len(losses) if losses else float("inf")
    return math.exp(mean_loss)

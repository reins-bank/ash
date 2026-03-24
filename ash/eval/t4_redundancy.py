"""
T4: Redundancy Tolerance

Repeat the same information N times in the context, then test comprehension.
CER should handle better — redundant tokens get closed after first integration.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ash.model.gpt import GPT
from ash.data.tokenizer import Tokenizer

STATEMENTS = [
    "The cat sat on the mat.",
    "Birds can fly south for the winter.",
    "The Earth orbits around the Sun.",
]


@torch.no_grad()
def test_redundancy(model: GPT, tokenizer: Tokenizer, device: str = "cpu") -> dict:
    """
    Measure how model loss changes with increasing redundancy.
    Stable loss = good redundancy tolerance.
    """
    model.eval()
    repetition_counts = [1, 2, 4, 8]
    results = {}

    for n_reps in repetition_counts:
        losses = []
        for stmt in STATEMENTS:
            repeated = (stmt + " ") * n_reps
            tokens = tokenizer.encode(repeated)
            if len(tokens) >= model.config.block_size:
                tokens = tokens[: model.config.block_size]

            if len(tokens) < 2:
                continue

            idx = torch.tensor([tokens[:-1]], device=device)
            targets = torch.tensor([tokens[1:]], device=device)
            logits, loss, _ = model(idx, targets)
            losses.append(loss.item())

        results[n_reps] = round(sum(losses) / len(losses), 4) if losses else float("inf")

    return {
        "loss_by_repetitions": results,
        "note": "stable loss across repetitions = good redundancy handling",
    }

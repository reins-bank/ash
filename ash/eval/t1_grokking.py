"""
T1: Grokking Speed Test

Train model on modular addition (a + b mod p) and measure steps to generalization.
CER should grok faster — it can release training-distribution artifacts.

This test requires training a small model from scratch, so it's more expensive
than the other tests. Run separately.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ash.model.gpt import GPT
from ash.data.tokenizer import Tokenizer


class ModularAdditionDataset(Dataset):
    """Dataset for a + b = c (mod p), encoded as token sequences."""

    def __init__(self, p: int = 97, split: str = "train", train_frac: float = 0.5):
        self.p = p
        pairs = [(a, b) for a in range(p) for b in range(p)]

        # Deterministic split
        n_train = int(len(pairs) * train_frac)
        if split == "train":
            self.pairs = pairs[:n_train]
        else:
            self.pairs = pairs[n_train:]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        c = (a + b) % self.p
        # Encode as: [a, +, b, =, c] using token IDs directly
        # Use first 97 tokens for numbers, 97 for +, 98 for =
        tokens = torch.tensor([a, self.p, b, self.p + 1, c], dtype=torch.long)
        return tokens[:-1], tokens[1:]


def test_grokking(model: GPT, tokenizer: Tokenizer, device: str = "cpu") -> dict:
    """
    Note: This test measures grokking on a simplified task.
    For the full test, train from scratch with the modular addition dataset.
    This stub returns a placeholder.
    """
    return {
        "note": "Grokking test requires separate training run. Use scripts/run_grokking.py.",
        "status": "not_run",
    }

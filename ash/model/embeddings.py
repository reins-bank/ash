from __future__ import annotations

import torch
import torch.nn as nn

from ash.config import ModelConfig


class Embeddings(nn.Module):
    """Token embedding + learned positional embedding."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.block_size = config.block_size

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        return self.drop(x)

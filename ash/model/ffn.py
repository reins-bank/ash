from __future__ import annotations

import torch.nn as nn

from ash.config import ModelConfig


class FeedForward(nn.Module):
    """GPT-2 style FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.ffn_hidden, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.ffn_hidden, config.d_model, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.drop(x)
        return x

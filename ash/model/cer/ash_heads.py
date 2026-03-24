from __future__ import annotations

import math

import torch
import torch.nn as nn


class ActiveSuppressionHeads(nn.Module):
    """
    Sigmoid-based attention heads that produce per-token suppression masks.

    Unlike standard attention (softmax, zero-sum), ASH uses sigmoid to produce
    independent suppression scores in [0, 1] for each query-key pair. This allows
    suppressing multiple tokens simultaneously.

    Input: hidden states [B, T, D], key states [B, H, T, D_h] (from standard attention)
    Output: suppression_mask [B, n_ash, T, T], ash_lambda [n_ash]
    """

    def __init__(self, d_model: int, n_ash_heads: int, head_dim: int, block_size: int,
                 lambda_init: float = 0.01):
        super().__init__()
        self.n_ash_heads = n_ash_heads
        self.head_dim = head_dim

        # Separate Q projection for ASH heads (K is shared with standard attention)
        self.w_q = nn.Linear(d_model, n_ash_heads * head_dim, bias=False)

        # Learnable per-head suppression strength
        self.ash_lambda = nn.Parameter(torch.full((n_ash_heads,), lambda_init))

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

        # Small init for Q projection
        nn.init.normal_(self.w_q.weight, mean=0.0, std=0.01)

    def forward(
        self, x: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: hidden states [B, T, D]
            k: key states from standard attention [B, H_total, T, D_h]
               We use the first n_ash_heads KV-groups (or average across all).

        Returns:
            suppress_mask: [B, n_ash, T, T] — suppression scores in [0, 1]
            ash_lambda: [n_ash] — learnable scaling factors
        """
        B, T, D = x.shape

        # Compute ASH queries
        q_ash = self.w_q(x).view(B, T, self.n_ash_heads, self.head_dim).transpose(1, 2)
        # [B, n_ash, T, D_h]

        # Use averaged K across all heads for ASH (shared KV)
        k_avg = k.mean(dim=1, keepdim=True).expand(-1, self.n_ash_heads, -1, -1)
        # [B, n_ash, T, D_h]

        # Sigmoid attention (not softmax — independent per-position scores)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q_ash @ k_avg.transpose(-2, -1)) * scale  # [B, n_ash, T, T]

        # Apply causal mask (can't suppress future tokens)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, -1e9)

        # Sigmoid for independent suppression scores
        suppress_mask = torch.sigmoid(scores)  # [B, n_ash, T, T]

        return suppress_mask, self.ash_lambda

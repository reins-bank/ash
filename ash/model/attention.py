from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ash.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with optional Active Suppression Heads.

    When CER is disabled: uses PyTorch's scaled_dot_product_attention (fused/flash path).
    When CER is enabled: uses manual attention to expose weights for ESC and ASH.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0

        self.n_head = config.n_head
        self.d_model = config.d_model
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.cer_enabled = config.cer.enabled

        # QKV projection for all heads
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # ASH components (initialized later in CER integration step)
        self.ash_heads = None

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            attn_out: [B, T, D]
            attn_weights: [B, n_head, T, T] if CER enabled, else None
        """
        B, T, D = x.shape

        # QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, H, T, D_h]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if not self.cer_enabled or self.ash_heads is None:
            # Fast path: fused attention (no weights returned)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            attn_weights = None
        else:
            # Manual attention: need weights for ESC stats and ASH modulation
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
            attn_weights = F.softmax(attn, dim=-1)
            attn_weights_drop = self.attn_drop(attn_weights)

            # ASH suppression
            suppress_mask, ash_lambda = self.ash_heads(x, k)
            # suppress_mask: [B, n_ash, T, T], ash_lambda: [n_ash]
            # Average across ASH heads and scale by lambda
            ash_lambda_clamped = ash_lambda.clamp(0.0, 1.0)
            # Weighted average: [B, 1, T, T]
            suppress_combined = (suppress_mask * ash_lambda_clamped[None, :, None, None]).mean(
                dim=1, keepdim=True
            )
            # Modulate: reduce attention to suppressed positions
            effective_weights = attn_weights_drop * (1.0 - suppress_combined)

            y = effective_weights @ v

        # Reassemble heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.resid_drop(self.c_proj(y))

        return y, attn_weights

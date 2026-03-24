from __future__ import annotations

import torch
import torch.nn as nn

from ash.config import ModelConfig
from ash.model.attention import CausalSelfAttention
from ash.model.ffn import FeedForward
from ash.model.cer.esc import EpistemicStateChannel
from ash.model.cer.ash_heads import ActiveSuppressionHeads
from ash.model.cer.closure_register import ClosureRegister


def _make_norm(config: ModelConfig) -> nn.Module:
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.d_model)
    return nn.LayerNorm(config.d_model, bias=config.bias)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block with optional CER augmentation.

    Standard path:  h -> LN -> Attn -> +residual -> LN -> FFN -> +residual
    CER path:       h -> LN -> Attn(+ASH) -> +residual -> ESC -> CR write -> +CR read
                      -> LN -> FFN -> +residual
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = _make_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = _make_norm(config)
        self.ffn = FeedForward(config)
        self.cer_enabled = config.cer.enabled

        # CER components
        self.esc = None
        self.cr = None
        if config.cer.enabled:
            self.esc = EpistemicStateChannel(
                d_model=config.d_model,
                n_head=config.n_head,
                mlp_hidden=config.cer.esc_mlp_hidden,
            )
            self.cr = ClosureRegister(
                d_model=config.d_model,
                d_cr=config.d_cr,
                injection_scale=config.cer.cr_injection_scale,
            )
            self.attn.ash_heads = ActiveSuppressionHeads(
                d_model=config.d_model,
                n_ash_heads=config.cer.n_ash_heads,
                head_dim=config.head_dim,
                block_size=config.block_size,
                lambda_init=config.cer.ash_lambda_init,
            )

    def forward(
        self, x: torch.Tensor, cr_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        """
        Returns:
            output: [B, T, D]
            cr_state: [B, d_cr] or None
            aux_info: dict with esc_scores, suppress_mask, attn_weights (empty if CER disabled)
        """
        aux_info: dict = {}

        # Attention
        attn_out, attn_weights = self.attn(self.ln_1(x))
        x = x + attn_out

        # CER: ESC + CR (when components are attached)
        if self.cer_enabled and self.esc is not None and attn_weights is not None:
            esc_scores, esc_target = self.esc(x, attn_weights)
            aux_info["esc_scores"] = esc_scores
            aux_info["esc_target"] = esc_target
            aux_info["attn_weights"] = attn_weights

            if self.cr is not None and cr_state is not None:
                cr_state = self.cr.write(x, esc_scores, cr_state)
                x = x + self.cr.read(cr_state).unsqueeze(1)  # [B, D] -> [B, 1, D] broadcast
                aux_info["cr_state"] = cr_state

        # FFN
        x = x + self.ffn(self.ln_2(x))

        return x, cr_state, aux_info

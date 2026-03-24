from __future__ import annotations

import math

import torch
import torch.nn as nn


class EpistemicStateChannel(nn.Module):
    """
    Computes per-token epistemic scores from hidden states and attention statistics.

    ESC tracks whether a token's information has been sufficiently integrated
    by the rest of the sequence. Score 0 = open/active, score 1 = closed/integrated.

    Input: hidden states [B, T, D], attention weights [B, H, T, T]
    Output: esc_scores [B, T, 1] in [0, 1], esc_target [B, T, 1]
    """

    def __init__(self, d_model: int, n_head: int, mlp_hidden: int = 64):
        super().__init__()
        self.n_head = n_head

        # 2-layer MLP: [h_t; attn_stats_t] -> score
        input_dim = d_model + n_head
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )

        # Critical: initialize final bias to -2.0 so tokens start "open"
        # sigmoid(-2) ≈ 0.12
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.constant_(self.mlp[2].bias, -2.0)

    def forward(
        self, h: torch.Tensor, attn_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: hidden states [B, T, D]
            attn_weights: attention weights [B, H, T, T]

        Returns:
            esc_scores: [B, T, 1] in [0, 1]
            esc_target: [B, T, 1] in [0, 1] — derived from attention entropy
        """
        B, T, D = h.shape

        # Compute attention statistics: how much each token is attended to by others
        # attn_weights[:, :, i, j] = how much query i attends to key j
        # Sum over query dim to get total incoming attention per key position
        incoming_attn = attn_weights.sum(dim=-2)  # [B, H, T] — total attn received per head
        # Transpose to [B, T, H] to concatenate with hidden states
        attn_stats = incoming_attn.transpose(1, 2)  # [B, T, H]

        # Concatenate and score
        mlp_input = torch.cat([h, attn_stats], dim=-1)  # [B, T, D + H]
        esc_scores = torch.sigmoid(self.mlp(mlp_input))  # [B, T, 1]

        # Compute ESC target from attention entropy
        esc_target = self._compute_target(attn_weights)  # [B, T, 1]

        return esc_scores, esc_target

    def _compute_target(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Derive ESC target from attention entropy.

        Tokens that many other tokens attend to diversely (high entropy) should
        stay open (low ESC target). Tokens with low incoming attention entropy
        can safely close (high ESC target).
        """
        B, H, T, _ = attn_weights.shape

        # Incoming attention distribution: for each key position, what is the
        # distribution of attention it receives from all query positions?
        # Normalize incoming attention to get a distribution over queries
        incoming = attn_weights.transpose(-2, -1)  # [B, H, T_key, T_query]
        incoming = incoming / (incoming.sum(dim=-1, keepdim=True) + 1e-8)

        # Entropy of incoming attention distribution
        entropy = -(incoming * torch.log(incoming + 1e-8)).sum(dim=-1)  # [B, H, T]
        entropy = entropy.mean(dim=1)  # [B, T] — average over heads

        # Normalize to [0, 1]
        max_entropy = math.log(T) if T > 1 else 1.0
        normalized_entropy = entropy / max_entropy

        # Target: high entropy = widely attended to = stay open (low target)
        esc_target = (1.0 - normalized_entropy).unsqueeze(-1)  # [B, T, 1]
        return esc_target.detach()  # Don't backprop through target

from __future__ import annotations

import torch
import torch.nn as nn


class ClosureRegister(nn.Module):
    """
    Compressed memory that accumulates closed epistemic state.

    When tokens' ESC scores indicate closure, their information is compressed
    and accumulated into a sequence-level register. This register is then
    injected back into the residual stream at a small scale, providing a
    faint signal of "forgotten" information.

    The CR state is [B, d_cr] — sequence-level, not per-token.
    """

    def __init__(self, d_model: int, d_cr: int, injection_scale: float = 0.1):
        super().__init__()
        self.d_cr = d_cr
        self.injection_scale = injection_scale

        # Compress: d_model -> d_cr
        self.compress = nn.Linear(d_model, d_cr, bias=False)
        # Expand: d_cr -> d_model
        self.expand = nn.Linear(d_cr, d_model, bias=False)
        # Gate: d_model -> d_cr
        self.gate_proj = nn.Linear(d_model, d_cr)

        # Initialize small so CR contribution is ~0.001 of residual norm at start
        nn.init.normal_(self.compress.weight, std=0.01)
        nn.init.normal_(self.expand.weight, std=0.01)
        nn.init.normal_(self.gate_proj.weight, std=0.01)
        nn.init.constant_(self.gate_proj.bias, -1.0)  # Gate starts mostly closed

    def write(
        self, h: torch.Tensor, esc_scores: torch.Tensor, cr_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Soft-write to closure register.

        Args:
            h: hidden states [B, T, D]
            esc_scores: epistemic scores [B, T, 1] in [0, 1]
            cr_state: current register state [B, d_cr]

        Returns:
            updated cr_state [B, d_cr]
        """
        gate = torch.sigmoid(self.gate_proj(h))  # [B, T, d_cr]
        compressed = self.compress(h)  # [B, T, d_cr]

        # Write amount proportional to ESC score (closed tokens contribute more)
        write_val = esc_scores * gate * compressed  # [B, T, d_cr]

        # Aggregate across token dimension
        cr_state = cr_state + write_val.sum(dim=1)  # [B, d_cr]

        return cr_state

    def read(self, cr_state: torch.Tensor) -> torch.Tensor:
        """
        Read from closure register and project back to model dimension.

        Args:
            cr_state: [B, d_cr]

        Returns:
            residual contribution [B, D] — broadcasts across token positions
        """
        return self.injection_scale * self.expand(cr_state)  # [B, D]

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ash.config import ModelConfig
from ash.model.embeddings import Embeddings
from ash.model.block import TransformerBlock, _make_norm


class GPT(nn.Module):
    """
    GPT-2 language model with optional CER augmentation.

    Composes: Embeddings -> N x TransformerBlock -> LayerNorm -> LM Head
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embeddings = Embeddings(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = _make_norm(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embeddings.tok_emb.weight

        # Gradient checkpointing (set by trainer before training)
        self.gradient_checkpointing = False

        # Init weights
        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 convention)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layer) ** 0.5)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        """
        Args:
            idx: [B, T] token indices
            targets: [B, T] target token indices (optional)

        Returns:
            logits: [B, T, vocab_size]
            loss: scalar or None
            cer_info: aggregated CER diagnostics
        """
        B, T = idx.shape

        x = self.embeddings(idx)

        # Initialize CR state
        cr_state = None
        if self.config.cer.enabled and self.config.cer.cr_persist_across_layers:
            cr_state = torch.zeros(B, self.config.d_cr, device=x.device, dtype=x.dtype)

        all_aux: list[dict] = []
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x, cr_state, aux = torch.utils.checkpoint.checkpoint(
                    block, x, cr_state, use_reentrant=False
                )
            else:
                x, cr_state, aux = block(x, cr_state)
            all_aux.append(aux)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        cer_info = self._aggregate_cer_info(all_aux)
        return logits, loss, cer_info

    def _aggregate_cer_info(self, all_aux: list[dict]) -> dict:
        """Collect CER diagnostics from all layers."""
        info: dict = {}
        if not self.config.cer.enabled:
            return info

        esc_scores = [a["esc_scores"] for a in all_aux if "esc_scores" in a]
        esc_targets = [a["esc_target"] for a in all_aux if "esc_target" in a]
        attn_weights = [a["attn_weights"] for a in all_aux if "attn_weights" in a]
        cr_states = [a["cr_state"] for a in all_aux if "cr_state" in a]

        if esc_scores:
            info["esc_scores_per_layer"] = esc_scores
            info["esc_targets_per_layer"] = esc_targets
        if attn_weights:
            info["attn_weights_per_layer"] = attn_weights
        if cr_states:
            info["cr_states_per_layer"] = cr_states

        return info

    def param_count(self) -> dict[str, int]:
        """Report parameter counts: total, base, and CER overhead."""
        total = sum(p.numel() for p in self.parameters())
        cer_params = 0
        for block in self.blocks:
            if block.esc is not None:
                cer_params += sum(p.numel() for p in block.esc.parameters())
            if block.cr is not None:
                cer_params += sum(p.numel() for p in block.cr.parameters())
            if block.attn.ash_heads is not None:
                cer_params += sum(p.numel() for p in block.attn.ash_heads.parameters())
        return {
            "total": total,
            "base": total - cer_params,
            "cer": cer_params,
        }

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, betas: tuple[float, float], device: str
    ) -> torch.optim.AdamW:
        """nanoGPT-style optimizer with weight-decay groups."""
        decay = set()
        no_decay = set()
        whitelist = (nn.Linear,)
        blacklist = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)

        # ASH lambda scalars and RMSNorm weights go to no_decay
        for fpn, _ in self.named_parameters():
            if fpn not in decay and fpn not in no_decay:
                no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # Filter to only params that exist (weight-tying removes lm_head.weight)
        decay = decay & param_dict.keys()
        no_decay = no_decay & param_dict.keys()
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]

        use_fused = device.startswith("cuda") and hasattr(torch.optim, "_multi_tensor")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, fused=False
        )
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

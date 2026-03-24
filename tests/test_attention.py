import torch
import pytest
from ash.config import ModelConfig
from ash.model.attention import CausalSelfAttention


def test_vanilla_attention_shape():
    cfg = ModelConfig()
    cfg.cer.enabled = False
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 64, 768)
    out, weights = attn(x)
    assert out.shape == (2, 64, 768)
    assert weights is None  # No weights returned in fused path


def test_cer_attention_returns_weights():
    cfg = ModelConfig()
    cfg.cer.enabled = True
    attn = CausalSelfAttention(cfg)
    # Manually attach ASH heads
    from ash.model.cer.ash_heads import ActiveSuppressionHeads
    attn.ash_heads = ActiveSuppressionHeads(
        d_model=768, n_ash_heads=2, head_dim=64, block_size=1024
    )
    x = torch.randn(2, 64, 768)
    out, weights = attn(x)
    assert out.shape == (2, 64, 768)
    assert weights is not None
    assert weights.shape == (2, 12, 64, 64)


def test_attention_causal():
    """Output at position t should not depend on positions > t."""
    cfg = ModelConfig(n_layer=1, n_head=4, d_model=128, block_size=32)
    cfg.cer.enabled = False
    attn = CausalSelfAttention(cfg)

    x = torch.randn(1, 8, 128)
    out1, _ = attn(x)

    # Change a future token
    x2 = x.clone()
    x2[0, 5, :] = torch.randn(128)
    out2, _ = attn(x2)

    # Positions 0-4 should be identical
    assert torch.allclose(out1[0, :5], out2[0, :5], atol=1e-5)

import torch
import pytest
from ash.config import ModelConfig
from ash.model.block import TransformerBlock


def test_block_no_cer():
    cfg = ModelConfig()
    cfg.cer.enabled = False
    block = TransformerBlock(cfg)
    x = torch.randn(2, 64, 768)
    out, cr_state, aux = block(x)
    assert out.shape == (2, 64, 768)
    assert cr_state is None
    assert aux == {}


def test_block_with_cer():
    cfg = ModelConfig()
    cfg.cer.enabled = True
    block = TransformerBlock(cfg)
    x = torch.randn(2, 64, 768)
    cr_state = torch.zeros(2, cfg.d_cr)
    out, new_cr, aux = block(x, cr_state)
    assert out.shape == (2, 64, 768)
    assert new_cr is not None
    assert new_cr.shape == (2, cfg.d_cr)
    assert "esc_scores" in aux
    assert "cr_state" in aux


def test_block_cr_state_propagates():
    """CR state should change between input and output."""
    cfg = ModelConfig()
    cfg.cer.enabled = True
    block = TransformerBlock(cfg)
    x = torch.randn(1, 32, 768)
    cr_state = torch.zeros(1, cfg.d_cr)
    _, new_cr, _ = block(x, cr_state)
    # CR should have accumulated something
    assert new_cr.norm() > 0


def test_block_gradient_flows_with_cer():
    cfg = ModelConfig(n_layer=1, n_head=4, d_model=128, block_size=64)
    cfg.cer.enabled = True
    cfg.cer.n_ash_heads = 1
    block = TransformerBlock(cfg)
    x = torch.randn(1, 16, 128, requires_grad=True)
    cr_state = torch.zeros(1, cfg.d_cr)
    out, _, _ = block(x, cr_state)
    out.sum().backward()
    assert x.grad is not None

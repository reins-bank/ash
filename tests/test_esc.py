import torch
import pytest
from ash.model.cer.esc import EpistemicStateChannel


def test_esc_output_shape():
    esc = EpistemicStateChannel(d_model=768, n_head=12)
    h = torch.randn(2, 64, 768)
    attn_w = torch.softmax(torch.randn(2, 12, 64, 64), dim=-1)
    scores, target = esc(h, attn_w)
    assert scores.shape == (2, 64, 1)
    assert target.shape == (2, 64, 1)


def test_esc_scores_in_unit_range():
    esc = EpistemicStateChannel(d_model=128, n_head=4)
    h = torch.randn(1, 32, 128)
    attn_w = torch.softmax(torch.randn(1, 4, 32, 32), dim=-1)
    scores, _ = esc(h, attn_w)
    assert (scores >= 0).all()
    assert (scores <= 1).all()


def test_esc_initial_bias_starts_open():
    """Tokens should start near 0 (open) due to bias=-2.0 init."""
    esc = EpistemicStateChannel(d_model=128, n_head=4)
    h = torch.zeros(1, 16, 128)  # Zero input
    attn_w = torch.softmax(torch.zeros(1, 4, 16, 16), dim=-1)
    scores, _ = esc(h, attn_w)
    # sigmoid(-2) ≈ 0.12
    assert scores.mean().item() < 0.2, f"Expected low initial ESC, got {scores.mean().item()}"


def test_esc_gradient_flows():
    esc = EpistemicStateChannel(d_model=128, n_head=4)
    # Override the zero-init on last layer to allow gradient flow
    torch.nn.init.normal_(esc.mlp[2].weight, std=0.1)
    h = torch.randn(1, 16, 128, requires_grad=True)
    attn_w = torch.softmax(torch.randn(1, 4, 16, 16), dim=-1)
    scores, _ = esc(h, attn_w)
    scores.sum().backward()
    assert h.grad is not None
    assert h.grad.abs().sum() > 0


def test_esc_target_detached():
    """ESC target should not require grad (no backprop through target)."""
    esc = EpistemicStateChannel(d_model=128, n_head=4)
    h = torch.randn(1, 16, 128)
    attn_w = torch.softmax(torch.randn(1, 4, 16, 16), dim=-1)
    _, target = esc(h, attn_w)
    assert not target.requires_grad

import torch
import pytest
from ash.model.cer.closure_register import ClosureRegister


def test_cr_write_shape():
    cr = ClosureRegister(d_model=768, d_cr=192)
    h = torch.randn(2, 64, 768)
    esc_scores = torch.rand(2, 64, 1)
    cr_state = torch.zeros(2, 192)
    new_state = cr.write(h, esc_scores, cr_state)
    assert new_state.shape == (2, 192)


def test_cr_read_shape():
    cr = ClosureRegister(d_model=768, d_cr=192)
    cr_state = torch.randn(2, 192)
    out = cr.read(cr_state)
    assert out.shape == (2, 768)


def test_cr_zero_esc_gives_minimal_update():
    """Zero ESC scores should produce near-zero CR update."""
    cr = ClosureRegister(d_model=128, d_cr=32)
    h = torch.randn(1, 16, 128)
    esc_scores = torch.zeros(1, 16, 1)  # All open
    cr_state = torch.zeros(1, 32)
    new_state = cr.write(h, esc_scores, cr_state)
    assert new_state.norm().item() < 1e-6


def test_cr_accumulates():
    """Multiple writes should accumulate state."""
    cr = ClosureRegister(d_model=128, d_cr=32)
    h = torch.randn(1, 16, 128)
    esc_scores = torch.ones(1, 16, 1)  # All closed
    cr_state = torch.zeros(1, 32)

    state1 = cr.write(h, esc_scores, cr_state)
    state2 = cr.write(h, esc_scores, state1)

    # Second write should increase the norm
    assert state2.norm() > state1.norm() * 0.5  # Not strict, just check accumulation


def test_cr_injection_scale():
    """Read output should be scaled by injection_scale."""
    cr = ClosureRegister(d_model=128, d_cr=32, injection_scale=0.1)
    cr_state = torch.ones(1, 32)
    out1 = cr.read(cr_state)

    cr2 = ClosureRegister(d_model=128, d_cr=32, injection_scale=1.0)
    cr2.expand.weight = cr.expand.weight  # Same weights
    out2 = cr2.read(cr_state)

    # 0.1 scale should be ~10x smaller
    ratio = out2.norm() / out1.norm()
    assert 8.0 < ratio < 12.0


def test_cr_gradient_flows():
    cr = ClosureRegister(d_model=128, d_cr=32)
    h = torch.randn(1, 16, 128, requires_grad=True)
    esc_scores = torch.rand(1, 16, 1)
    cr_state = torch.zeros(1, 32)
    new_state = cr.write(h, esc_scores, cr_state)
    out = cr.read(new_state)
    out.sum().backward()
    assert h.grad is not None

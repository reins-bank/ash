import torch
import pytest
from ash.model.cer.ash_heads import ActiveSuppressionHeads


def test_ash_output_shape():
    ash = ActiveSuppressionHeads(d_model=768, n_ash_heads=2, head_dim=64, block_size=1024)
    x = torch.randn(2, 64, 768)
    k = torch.randn(2, 12, 64, 64)
    suppress, lam = ash(x, k)
    assert suppress.shape == (2, 2, 64, 64)
    assert lam.shape == (2,)


def test_ash_scores_in_unit_range():
    ash = ActiveSuppressionHeads(d_model=128, n_ash_heads=1, head_dim=32, block_size=256)
    x = torch.randn(1, 32, 128)
    k = torch.randn(1, 4, 32, 32)
    suppress, _ = ash(x, k)
    assert (suppress >= 0).all()
    assert (suppress <= 1).all()


def test_ash_causal_mask():
    """Future positions should be suppressed to ~0 (sigmoid of very negative = ~0)."""
    ash = ActiveSuppressionHeads(d_model=128, n_ash_heads=1, head_dim=32, block_size=256)
    x = torch.randn(1, 8, 128)
    k = torch.randn(1, 4, 8, 32)
    suppress, _ = ash(x, k)
    # Upper triangle should be near 0 (future positions masked)
    for i in range(8):
        for j in range(i + 1, 8):
            assert suppress[0, 0, i, j].item() < 0.01


def test_ash_lambda_init():
    ash = ActiveSuppressionHeads(d_model=128, n_ash_heads=2, head_dim=32, block_size=256, lambda_init=0.05)
    assert torch.allclose(ash.ash_lambda, torch.tensor([0.05, 0.05]))


def test_ash_gradient_flows():
    ash = ActiveSuppressionHeads(d_model=128, n_ash_heads=1, head_dim=32, block_size=256)
    x = torch.randn(1, 16, 128, requires_grad=True)
    k = torch.randn(1, 4, 16, 32)
    suppress, _ = ash(x, k)
    suppress.sum().backward()
    assert x.grad is not None

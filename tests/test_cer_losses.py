import torch
import pytest
from ash.training.losses import esc_loss, ash_regularization_loss, combined_loss


def _make_cer_info(n_layers=4, B=2, T=16):
    """Helper to create mock CER info dict."""
    return {
        "esc_scores_per_layer": [torch.rand(B, T, 1) for _ in range(n_layers)],
        "esc_targets_per_layer": [torch.rand(B, T, 1) for _ in range(n_layers)],
        "attn_weights_per_layer": [
            torch.softmax(torch.randn(B, 4, T, T), dim=-1) for _ in range(n_layers)
        ],
    }


def test_esc_loss_zero_when_scores_below_target():
    """If scores are always below target, loss should be 0."""
    cer_info = {
        "esc_scores_per_layer": [torch.zeros(1, 8, 1)],
        "esc_targets_per_layer": [torch.ones(1, 8, 1)],
    }
    loss = esc_loss(cer_info)
    assert loss.item() == 0.0


def test_esc_loss_positive_when_scores_exceed_target():
    """If scores exceed target, loss should be positive."""
    cer_info = {
        "esc_scores_per_layer": [torch.ones(1, 8, 1)],  # All 1.0
        "esc_targets_per_layer": [torch.zeros(1, 8, 1)],  # All 0.0
    }
    loss = esc_loss(cer_info)
    assert loss.item() > 0.0


def test_ash_reg_loss_increases_with_esc():
    """Higher mean ESC should give higher regularization loss."""
    cer_info_low = {"esc_scores_per_layer": [torch.full((1, 8, 1), 0.1)]}
    cer_info_high = {"esc_scores_per_layer": [torch.full((1, 8, 1), 0.9)]}
    loss_low = ash_regularization_loss(cer_info_low)
    loss_high = ash_regularization_loss(cer_info_high)
    assert loss_high > loss_low


def test_combined_loss_weights():
    cer_info = _make_cer_info()
    lm_loss = torch.tensor(5.0)
    total, breakdown = combined_loss(lm_loss, cer_info, lambda_esc=0.0, lambda_ash=0.0)
    # With zero lambdas, total should equal LM loss
    assert abs(total.item() - 5.0) < 0.01


def test_combined_loss_with_cer():
    cer_info = _make_cer_info()
    lm_loss = torch.tensor(5.0)
    total, breakdown = combined_loss(lm_loss, cer_info, lambda_esc=0.1, lambda_ash=0.05)
    # Total should be >= LM loss (CER losses add to it)
    assert total.item() >= lm_loss.item() - 0.01
    assert "loss/lm" in breakdown
    assert "loss/esc" in breakdown
    assert "loss/ash_reg" in breakdown


def test_empty_cer_info():
    """Should handle empty CER info gracefully."""
    lm_loss = torch.tensor(5.0)
    total, breakdown = combined_loss(lm_loss, {}, lambda_esc=0.1, lambda_ash=0.05)
    assert abs(total.item() - 5.0) < 0.01

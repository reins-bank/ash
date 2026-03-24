from __future__ import annotations

import math

import torch


def esc_loss(cer_info: dict) -> torch.Tensor:
    """
    ESC auxiliary loss: penalizes ESC scores that exceed their entropy-derived targets.

    L_esc = mean over layers and tokens of max(0, esc_score - esc_target)^2

    This encourages ESC scores to be justified — tokens shouldn't be marked
    closed prematurely (before their information has been absorbed).
    """
    esc_scores = cer_info.get("esc_scores_per_layer", [])
    esc_targets = cer_info.get("esc_targets_per_layer", [])

    if not esc_scores:
        return torch.tensor(0.0)

    total = torch.tensor(0.0, device=esc_scores[0].device)
    for scores, targets in zip(esc_scores, esc_targets):
        # Penalize only when score exceeds target (premature closure)
        excess = torch.clamp(scores - targets, min=0.0)
        total = total + (excess ** 2).mean()

    return total / len(esc_scores)


def ash_regularization_loss(cer_info: dict) -> torch.Tensor:
    """
    ASH regularization: penalizes high mean ESC scores to prevent
    over-suppression. High mean ESC = model closing too aggressively.

    L_ash = mean ESC score across all layers
    """
    esc_scores = cer_info.get("esc_scores_per_layer", [])

    if not esc_scores:
        return torch.tensor(0.0)

    # Penalize high mean ESC (proxy for over-suppression)
    total = torch.tensor(0.0, device=esc_scores[0].device)
    for scores in esc_scores:
        total = total + scores.mean()

    return total / len(esc_scores)


def combined_loss(
    lm_loss: torch.Tensor,
    cer_info: dict,
    lambda_esc: float,
    lambda_ash: float,
) -> tuple[torch.Tensor, dict]:
    """
    Combined training loss.

    L = L_lm + lambda_esc * L_esc + lambda_ash * L_ash_reg

    Returns:
        total_loss: scalar
        breakdown: dict with per-component losses for logging
    """
    l_esc = esc_loss(cer_info)
    l_ash = ash_regularization_loss(cer_info)

    total = lm_loss + lambda_esc * l_esc + lambda_ash * l_ash

    breakdown = {
        "loss/total": total.item(),
        "loss/lm": lm_loss.item(),
        "loss/esc": l_esc.item(),
        "loss/ash_reg": l_ash.item(),
        "loss/lambda_esc": lambda_esc,
        "loss/lambda_ash": lambda_ash,
    }

    return total, breakdown

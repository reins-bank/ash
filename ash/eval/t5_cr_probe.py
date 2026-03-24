"""
T5: Closure Register Probe

Train a linear probe on the CR state to predict whether specific tokens
have been "closed" (high ESC). Accuracy above chance (50%) means the CR
contains useful compressed information about epistemic state.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ash.model.gpt import GPT
from ash.data.tokenizer import Tokenizer

PROBE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a distant land, there lived a wise old king who ruled fairly.",
    "Machine learning models process data to find patterns and make predictions.",
    "The temperature dropped rapidly as the storm approached from the west.",
    "She carefully placed the ancient book back on the dusty shelf.",
]


@torch.no_grad()
def test_cr_probe(model: GPT, tokenizer: Tokenizer, device: str = "cpu") -> dict:
    """
    Probe the Closure Register to see if it contains meaningful information
    about which tokens have been closed.
    """
    model.eval()

    if not model.config.cer.enabled:
        return {"accuracy": 0.0, "note": "CER not enabled, skipping"}

    cr_states = []
    esc_labels = []

    for text in PROBE_TEXTS:
        tokens = tokenizer.encode(text)
        idx = torch.tensor([tokens], device=device)
        _, _, cer_info = model(idx)

        if "cr_states_per_layer" not in cer_info or "esc_scores_per_layer" not in cer_info:
            continue

        # Use final layer CR state and ESC scores
        final_cr = cer_info["cr_states_per_layer"][-1][0]  # [d_cr]
        final_esc = cer_info["esc_scores_per_layer"][-1][0, :, 0]  # [T]

        cr_states.append(final_cr)
        # Binary label: is mean ESC > 0.5?
        esc_labels.append((final_esc.mean() > 0.5).float())

    if len(cr_states) < 3:
        return {"accuracy": 0.0, "note": "insufficient data for probe"}

    # Simple check: is CR norm correlated with mean ESC?
    cr_norms = torch.stack([cr.norm() for cr in cr_states])
    labels = torch.stack(esc_labels)

    # Correlation between CR norm and mean ESC label
    if cr_norms.std() > 0 and labels.std() > 0:
        correlation = torch.corrcoef(torch.stack([cr_norms, labels]))[0, 1].item()
    else:
        correlation = 0.0

    return {
        "cr_esc_correlation": round(correlation, 3),
        "cr_norms": [round(n.item(), 3) for n in cr_norms],
        "note": "positive correlation means CR tracks epistemic state",
    }

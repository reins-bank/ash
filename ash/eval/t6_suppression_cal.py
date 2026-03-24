"""
T6: Suppression Calibration

Compare ASH suppression masks against expected irrelevant tokens
(function words, fillers). Checks if ASH learns to suppress the "right" tokens.
"""
from __future__ import annotations

import torch

from ash.model.gpt import GPT
from ash.data.tokenizer import Tokenizer

FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
    "of", "and", "or", "but", "for", "with", "by", "it", "this", "that",
}

TEST_TEXTS = [
    "The scientist discovered a new method for analyzing the data in the lab.",
    "A large earthquake struck the coastal city and caused significant damage.",
]


@torch.no_grad()
def test_suppression_calibration(
    model: GPT, tokenizer: Tokenizer, device: str = "cpu"
) -> dict:
    """
    Check if ASH suppression correlates with function word positions.
    After training, function words should be more suppressed than content words.
    """
    model.eval()

    if not model.config.cer.enabled:
        return {"note": "CER not enabled, skipping"}

    func_esc_scores = []
    content_esc_scores = []

    for text in TEST_TEXTS:
        tokens = tokenizer.encode(text)
        token_strs = [tokenizer.decode([t]).strip().lower() for t in tokens]

        idx = torch.tensor([tokens], device=device)
        _, _, cer_info = model(idx)

        if "esc_scores_per_layer" not in cer_info:
            continue

        # Use final layer ESC as proxy for suppression
        final_esc = cer_info["esc_scores_per_layer"][-1][0, :, 0]  # [T]

        for t_idx, word in enumerate(token_strs):
            score = final_esc[t_idx].item()
            if word in FUNCTION_WORDS:
                func_esc_scores.append(score)
            elif word.isalpha():
                content_esc_scores.append(score)

    func_mean = sum(func_esc_scores) / len(func_esc_scores) if func_esc_scores else 0.0
    content_mean = sum(content_esc_scores) / len(content_esc_scores) if content_esc_scores else 0.0

    return {
        "function_word_mean_esc": round(func_mean, 3),
        "content_word_mean_esc": round(content_mean, 3),
        "suppression_differential": round(func_mean - content_mean, 3),
        "calibrated": func_mean > content_mean,
        "note": "function words should have higher ESC (more suppressed) after training",
    }

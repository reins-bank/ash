"""
T8: Cross-Layer ESC Flow Analysis

Extract ESC scores for every token at every layer and verify expected dynamics:
- Function words (articles, prepositions) should close early (high ESC in middle layers)
- Content words (nouns, verbs) should stay open longer
"""
from __future__ import annotations

import torch

from ash.model.gpt import GPT
from ash.data.tokenizer import Tokenizer

# Simple POS-like classification by common words
FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
    "of", "and", "or", "but", "for", "with", "by", "from", "it", "its",
    "this", "that", "these", "those", "has", "have", "had", "be", "been",
}

TEST_TEXTS = [
    "The quick brown fox jumped over the lazy dog in the park.",
    "A large ship sailed across the vast ocean towards the distant horizon.",
    "The scientist carefully examined the ancient artifacts from the excavation site.",
]


@torch.no_grad()
def test_esc_flow(model: GPT, tokenizer: Tokenizer, device: str = "cpu") -> dict:
    """
    Analyze ESC score dynamics across layers.
    Checks if function words close earlier than content words.
    """
    model.eval()

    if not model.config.cer.enabled:
        return {"note": "CER not enabled, skipping"}

    function_word_esc_trajectories = []
    content_word_esc_trajectories = []

    for text in TEST_TEXTS:
        tokens = tokenizer.encode(text)
        token_strs = [tokenizer.decode([t]).strip().lower() for t in tokens]

        idx = torch.tensor([tokens], device=device)
        _, _, cer_info = model(idx)

        if "esc_scores_per_layer" not in cer_info:
            continue

        # Stack ESC scores: [n_layers, T]
        esc_stack = torch.stack(
            [s[0, :, 0] for s in cer_info["esc_scores_per_layer"]]
        )  # [L, T]

        for t_idx, word in enumerate(token_strs):
            trajectory = esc_stack[:, t_idx].cpu().tolist()
            if word in FUNCTION_WORDS:
                function_word_esc_trajectories.append(trajectory)
            else:
                content_word_esc_trajectories.append(trajectory)

    if not function_word_esc_trajectories or not content_word_esc_trajectories:
        return {"note": "insufficient data for analysis"}

    # Average trajectories
    n_layers = len(function_word_esc_trajectories[0])
    func_mean = [
        sum(t[l] for t in function_word_esc_trajectories) / len(function_word_esc_trajectories)
        for l in range(n_layers)
    ]
    content_mean = [
        sum(t[l] for t in content_word_esc_trajectories) / len(content_word_esc_trajectories)
        for l in range(n_layers)
    ]

    # Check: do function words have higher ESC in later layers?
    func_late = sum(func_mean[n_layers // 2 :]) / (n_layers // 2)
    content_late = sum(content_mean[n_layers // 2 :]) / (n_layers // 2)
    func_closes_faster = func_late > content_late

    return {
        "function_word_mean_esc": [round(v, 3) for v in func_mean],
        "content_word_mean_esc": [round(v, 3) for v in content_mean],
        "function_late_mean": round(func_late, 3),
        "content_late_mean": round(content_late, 3),
        "function_closes_faster": func_closes_faster,
        "note": "function words should close faster than content words (after training)",
    }

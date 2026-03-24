"""
T3: Long Context Noise Accumulation

Insert a fact early in context, pad with irrelevant text, then measure
whether the model can still retrieve the fact. CER should degrade less
as noise increases because it actively suppresses noise tokens.
"""
from __future__ import annotations

import torch

from ash.model.gpt import GPT
from ash.data.tokenizer import Tokenizer

FACTS = [
    ("The capital of France is Paris.", "Paris"),
    ("Water boils at 100 degrees Celsius.", "100"),
    ("The speed of light is approximately 300000 kilometers per second.", "300000"),
    ("Shakespeare wrote Hamlet.", "Hamlet"),
    ("The chemical symbol for gold is Au.", "Au"),
]

FILLER = "The weather was particularly unusual that day, with clouds forming patterns that nobody had seen before. "


@torch.no_grad()
def test_noise_robustness(model: GPT, tokenizer: Tokenizer, device: str = "cpu") -> dict:
    """
    Measure fact retrieval accuracy as noise padding increases.
    Returns accuracy at different noise levels.
    """
    model.eval()
    noise_levels = [0, 50, 100, 200, 400]
    results_by_noise = {}

    for noise_tokens in noise_levels:
        correct = 0
        total = 0

        for fact, answer in FACTS:
            # Build context: fact + noise + prompt
            filler_text = FILLER * (noise_tokens // 10 + 1)
            filler_tokens = tokenizer.encode(filler_text)[:noise_tokens]
            fact_tokens = tokenizer.encode(fact)

            prompt = fact_tokens + filler_tokens
            if len(prompt) >= model.config.block_size - 10:
                continue

            idx = torch.tensor([prompt], device=device)
            logits, _, _ = model(idx)

            # Check if the answer token has high probability in the last position
            last_logits = logits[0, -1]
            top_tokens = torch.topk(last_logits, k=50).indices.tolist()
            answer_tokens = tokenizer.encode(" " + answer)

            if answer_tokens and answer_tokens[0] in top_tokens:
                correct += 1
            total += 1

        results_by_noise[noise_tokens] = correct / total if total > 0 else 0.0

    return {
        "accuracy_by_noise": results_by_noise,
        "note": "higher accuracy at higher noise = better noise robustness",
    }

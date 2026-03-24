"""
T7: Chekhov's Gun (Adversarial)

Present information that seems irrelevant but becomes critical later.
This is the most important CER failure mode test — the model must NOT
suppress information too aggressively.

Example: "John put the key under the mat. [300 tokens of other narrative].
Where is the key?" — the model must retain "key" / "mat" information.
"""
from __future__ import annotations

import torch

from ash.model.gpt import GPT
from ash.data.tokenizer import Tokenizer

FILLER = "Meanwhile, the weather outside was changing rapidly. Dark clouds gathered on the horizon and the wind picked up considerably. People on the street hurried to find shelter. "

CHEKHOV_TESTS = [
    {
        "setup": "John placed his old key under the doormat before leaving.",
        "filler_reps": 5,
        "query": " The key was under the",
        "expected": "doormat",
    },
    {
        "setup": "The red book on the top shelf contained the secret code.",
        "filler_reps": 5,
        "query": " The secret code was in the",
        "expected": "red",
    },
    {
        "setup": "Maria left her umbrella in the cafe on Tuesday morning.",
        "filler_reps": 5,
        "query": " Maria's umbrella was at the",
        "expected": "cafe",
    },
]


@torch.no_grad()
def test_chekhov(model: GPT, tokenizer: Tokenizer, device: str = "cpu") -> dict:
    """
    Test whether critical early information survives CER suppression.
    Measures ESC scores of critical tokens and answer accuracy.
    """
    model.eval()
    results = []

    for test in CHEKHOV_TESTS:
        setup_tokens = tokenizer.encode(test["setup"])
        filler_tokens = tokenizer.encode(FILLER * test["filler_reps"])
        query_tokens = tokenizer.encode(test["query"])

        full_tokens = setup_tokens + filler_tokens + query_tokens
        if len(full_tokens) >= model.config.block_size:
            full_tokens = full_tokens[: model.config.block_size]

        idx = torch.tensor([full_tokens], device=device)
        logits, _, cer_info = model(idx)

        # Check if expected answer is in top predictions
        last_logits = logits[0, -1]
        top_k = 20
        top_tokens = torch.topk(last_logits, k=top_k).indices.tolist()
        expected_tokens = tokenizer.encode(" " + test["expected"])

        found = any(t in top_tokens for t in expected_tokens) if expected_tokens else False

        # Check ESC scores of setup tokens (should remain open if CER is working well)
        setup_esc = None
        if "esc_scores_per_layer" in cer_info:
            final_esc = cer_info["esc_scores_per_layer"][-1][0, :, 0]
            setup_esc = final_esc[: len(setup_tokens)].mean().item()

        results.append({
            "found_answer": found,
            "setup_esc_mean": round(setup_esc, 3) if setup_esc is not None else None,
        })

    n_correct = sum(1 for r in results if r["found_answer"])
    return {
        "accuracy": n_correct / len(results) if results else 0.0,
        "details": results,
        "note": "critical: CER must NOT suppress information that becomes relevant later",
    }

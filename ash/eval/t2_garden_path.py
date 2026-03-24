"""
T2: Garden Path Recovery

Feed syntactically misleading sentences and measure surprisal at the
disambiguation point. CER should recover faster (lower surprisal) because
it can release the initial misparse.
"""
from __future__ import annotations

import math

import torch

from ash.model.gpt import GPT
from ash.data.tokenizer import Tokenizer

# Garden path sentences with marked disambiguation points.
# Format: (sentence, disambiguation_word)
GARDEN_PATHS = [
    ("The horse raced past the barn fell.", "fell"),
    ("The old man the boats.", "the"),
    ("The complex houses married and single soldiers and their families.", "married"),
    ("The cotton clothing is made of grows in Mississippi.", "grows"),
    ("The girl told the story cried.", "cried"),
    ("Fat people eat accumulates.", "accumulates"),
    ("The raft floated down the river sank.", "sank"),
    ("The man who hunts ducks out on weekends.", "out"),
    ("While the man hunted the deer ran into the woods.", "ran"),
    ("The dog walked to the park barked.", "barked"),
]


@torch.no_grad()
def test_garden_path(model: GPT, tokenizer: Tokenizer, device: str = "cpu") -> dict:
    """
    Measure mean surprisal at disambiguation points in garden path sentences.
    Lower surprisal = better recovery from misparse.
    """
    model.eval()
    surprisals = []

    for sentence, disambig_word in GARDEN_PATHS:
        tokens = tokenizer.encode(sentence)
        token_strs = [tokenizer.decode([t]) for t in tokens]

        # Find the disambiguation token
        disambig_idx = None
        for i, ts in enumerate(token_strs):
            if disambig_word in ts.strip().lower():
                disambig_idx = i
                break

        if disambig_idx is None or disambig_idx == 0:
            continue

        idx = torch.tensor([tokens], device=device)
        logits, _, _ = model(idx)

        # Surprisal at disambiguation point
        log_probs = torch.log_softmax(logits[0, disambig_idx - 1], dim=-1)
        surprisal = -log_probs[tokens[disambig_idx]].item()
        surprisals.append(surprisal)

    mean_surprisal = sum(surprisals) / len(surprisals) if surprisals else float("inf")
    return {
        "mean_surprisal": round(mean_surprisal, 3),
        "n_sentences": len(surprisals),
        "note": "lower is better (easier recovery from misparse)",
    }

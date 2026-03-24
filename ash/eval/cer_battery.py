from __future__ import annotations

import torch

from ash.model.gpt import GPT
from ash.data.tokenizer import Tokenizer


def run_cer_battery(model: GPT, tokenizer: Tokenizer, device: str = "cpu") -> dict:
    """
    Run all CER evaluation tests.

    Returns a dict mapping test name to result summary.
    """
    results = {}

    from ash.eval.t2_garden_path import test_garden_path
    from ash.eval.t3_noise_robustness import test_noise_robustness
    from ash.eval.t4_redundancy import test_redundancy
    from ash.eval.t5_cr_probe import test_cr_probe
    from ash.eval.t8_esc_flow import test_esc_flow

    results["T2_garden_path"] = test_garden_path(model, tokenizer, device)
    results["T3_noise_robustness"] = test_noise_robustness(model, tokenizer, device)
    results["T4_redundancy"] = test_redundancy(model, tokenizer, device)
    results["T5_cr_probe"] = test_cr_probe(model, tokenizer, device)
    results["T8_esc_flow"] = test_esc_flow(model, tokenizer, device)

    return results

"""
Evaluate an Ashy-Small checkpoint.

Usage:
    python scripts/eval.py --checkpoint checkpoints/ckpt_best.pt --battery cer
    python scripts/eval.py --checkpoint checkpoints/ckpt_best.pt --eval perplexity
"""
from __future__ import annotations

import argparse

import torch

from ash.config import ModelConfig, CERConfig
from ash.model.gpt import GPT
from ash.training.checkpoint import load_checkpoint
from ash.data.tokenizer import Tokenizer


def rebuild_config(config_dict: dict) -> ModelConfig:
    """Rebuild ModelConfig from saved dict."""
    cer_dict = config_dict.pop("cer", {})
    cer_config = CERConfig(**cer_dict)
    return ModelConfig(**config_dict, cer=cer_config)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ashy-Small")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--battery", type=str, default=None, choices=["cer"])
    parser.add_argument("--eval", type=str, default="perplexity", choices=["perplexity"])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = rebuild_config(ckpt["model_config"])
    model = GPT(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(args.device)
    model.eval()

    step = ckpt.get("step", "?")
    print(f"Model loaded (step {step}, {model.param_count()['total']:,} params)")

    tokenizer = Tokenizer()

    if args.battery == "cer":
        from ash.eval.cer_battery import run_cer_battery
        results = run_cer_battery(model, tokenizer, args.device)
        print("\nCER Battery Results:")
        for test_name, result in results.items():
            print(f"  {test_name}: {result}")

    if args.eval == "perplexity":
        from ash.eval.perplexity import evaluate_perplexity
        from ash.data.dataset import get_dataloader
        val_loader = get_dataloader(
            data_dir="data/",
            split="val",
            block_size=model_cfg.block_size,
            batch_size=8,
            vocab_size=model_cfg.vocab_size,
        )
        ppl = evaluate_perplexity(model, val_loader, device=args.device)
        print(f"Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()

"""
Visualize CER dynamics for a given text input.

Usage:
    python scripts/visualize_cer.py --checkpoint ckpt.pt --text "The horse raced past the barn fell."
"""
from __future__ import annotations

import argparse

import torch

from ash.config import ModelConfig, CERConfig
from ash.model.gpt import GPT
from ash.data.tokenizer import Tokenizer
from ash.utils.viz import plot_esc_heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--out", type=str, default="cer_viz.png")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cer_dict = ckpt["model_config"].pop("cer", {})
    model_cfg = ModelConfig(**ckpt["model_config"], cer=CERConfig(**cer_dict))
    model = GPT(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(args.device)
    model.eval()

    tokenizer = Tokenizer()
    tokens = tokenizer.encode(args.text)
    token_strs = [tokenizer.decode([t]) for t in tokens]

    idx = torch.tensor([tokens], device=args.device)
    with torch.no_grad():
        _, _, cer_info = model(idx)

    if "esc_scores_per_layer" not in cer_info:
        print("CER not enabled in this model.")
        return

    fig = plot_esc_heatmap(cer_info["esc_scores_per_layer"], token_strs)
    fig.savefig(args.out, dpi=150)
    print(f"Saved ESC heatmap to {args.out}")


if __name__ == "__main__":
    main()

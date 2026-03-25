"""
Train Ashy-Small.

Usage:
    python scripts/train.py --config configs/ashy_small.yaml
    python scripts/train.py --config configs/ashy_small_debug.yaml
    python scripts/train.py --config configs/ashy_small_modal.yaml   # runs on Modal GPU
"""
from __future__ import annotations

import argparse

from ash.config import load_config
from ash.training.runner import run_training


def main():
    parser = argparse.ArgumentParser(description="Train Ashy-Small")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    model_cfg, train_cfg = load_config(args.config)

    # Modal handoff
    if train_cfg.modal.enabled:
        from ash.infra.modal_runner import run_on_modal
        run_on_modal(model_cfg, train_cfg, resume_path=args.resume)
        return

    run_training(model_cfg, train_cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()

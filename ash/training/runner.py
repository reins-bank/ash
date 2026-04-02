"""Core training orchestration, shared by local and Modal code paths."""
from __future__ import annotations

from dataclasses import asdict

import torch

from ash.config import ModelConfig, TrainingConfig
from ash.model.gpt import GPT
from ash.data.dataset import get_dataloader
from ash.data.storage import resolve_data_dir
from ash.training.trainer import Trainer
from ash.utils.logging import Logger


def run_training(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    resume_path: str | None = None,
) -> None:
    """Run training end-to-end. Called locally or from Modal."""

    # Device setup
    device = train_cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        train_cfg.device = "cpu"
        train_cfg.compile = False
        if train_cfg.dtype == "bfloat16":
            train_cfg.dtype = "float32"

    # Model
    model = GPT(model_cfg)
    model = model.to(device)
    counts = model.param_count()
    print(f"Model: {counts['total']:,} params ({counts['total']/1e6:.1f}M)")
    print(f"  Base: {counts['base']:,} ({counts['base']/1e6:.1f}M)")
    print(f"  CER:  {counts['cer']:,} ({counts['cer']/1e6:.1f}M)")
    print(f"  CER enabled: {model_cfg.cer.enabled}")

    # Gradient checkpointing
    if train_cfg.gradient_checkpointing:
        model.gradient_checkpointing = True
        print("Gradient checkpointing enabled")

    # Compile
    if train_cfg.compile and device != "cpu":
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Data — resolve storage backend (S3 will fail locally with a clear message)
    data_dir = resolve_data_dir(train_cfg.storage, train_cfg.data_dir)
    train_loader = get_dataloader(
        data_dir=data_dir,
        split="train",
        block_size=train_cfg.block_size,
        batch_size=train_cfg.batch_size,
        vocab_size=model_cfg.vocab_size,
    )
    val_loader = get_dataloader(
        data_dir=data_dir,
        split="val",
        block_size=train_cfg.block_size,
        batch_size=train_cfg.batch_size,
        vocab_size=model_cfg.vocab_size,
    )

    # Logger
    logger = None
    if train_cfg.wandb_enabled:
        logger = Logger(
            wandb_enabled=True,
            project=train_cfg.wandb_project,
            config={**asdict(model_cfg), **asdict(train_cfg)},
        )

    # Trainer
    trainer = Trainer(
        model=model,
        model_config=model_cfg,
        train_config=train_cfg,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        logger=logger,
    )

    # Resume
    if resume_path:
        from ash.training.checkpoint import load_checkpoint
        ckpt = load_checkpoint(resume_path, model)
        trainer.step = ckpt.get("step", 0)
        print(f"Resumed from step {trainer.step}")

    # Train
    trainer.train()

    if logger:
        logger.finish()

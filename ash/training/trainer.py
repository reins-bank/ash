from __future__ import annotations

import time
from pathlib import Path

import torch

from ash.config import ModelConfig, TrainingConfig
from ash.model.gpt import GPT
from ash.training.losses import combined_loss
from ash.training.optimizer import configure_optimizer
from ash.training.scheduler import CosineWarmupScheduler, CERCurriculumScheduler
from ash.training.checkpoint import save_checkpoint


class Trainer:
    """Main training loop with CER curriculum scheduling."""

    def __init__(
        self,
        model: GPT,
        model_config: ModelConfig,
        train_config: TrainingConfig,
        train_dataloader,
        val_dataloader=None,
        logger=None,
    ):
        self.model = model
        self.model_config = model_config
        self.config = train_config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger

        self.device = train_config.device
        self.dtype = getattr(torch, train_config.dtype, torch.float32)

        # Optimizer
        self.optimizer = configure_optimizer(model, train_config)

        # LR scheduler
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=train_config.warmup_steps,
            max_steps=train_config.max_steps,
            lr_max=train_config.learning_rate,
            lr_min=train_config.lr_decay_to,
        )

        # CER curriculum
        self.cer_curriculum = CERCurriculumScheduler(
            lambda_esc_max=model_config.cer.lambda_esc_max,
            lambda_ash_max=model_config.cer.lambda_ash_reg_max,
            phase1_end=model_config.cer.curriculum_phase1_end,
            phase2_end=model_config.cer.curriculum_phase2_end,
        )

        self.step = 0
        self.best_val_loss = float("inf")

    def train(self):
        """Run full training loop."""
        model = self.model
        config = self.config
        model.train()

        data_iter = iter(self.train_dataloader)
        t0 = time.time()

        while self.step < config.max_steps:
            # Gradient accumulation
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            accum_breakdown = {}

            for micro_step in range(config.gradient_accumulation_steps):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_dataloader)
                    x, y = next(data_iter)

                x = x.to(self.device)
                y = y.to(self.device)

                with torch.autocast(device_type=self.device.split(":")[0], dtype=self.dtype):
                    logits, lm_loss, cer_info = model(x, y)

                    # CER curriculum
                    lambda_esc, lambda_ash = self.cer_curriculum.get_lambdas(
                        self.step, config.max_steps
                    )
                    total_loss, breakdown = combined_loss(
                        lm_loss, cer_info, lambda_esc, lambda_ash
                    )
                    scaled_loss = total_loss / config.gradient_accumulation_steps

                scaled_loss.backward()
                accum_loss += total_loss.item() / config.gradient_accumulation_steps

                # Accumulate breakdown
                for k, v in breakdown.items():
                    accum_breakdown[k] = accum_breakdown.get(k, 0.0) + v / config.gradient_accumulation_steps

            # Gradient clipping
            if config.grad_clip > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            else:
                grad_norm = torch.tensor(0.0)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.step += 1

            # Logging
            if self.step % config.log_interval == 0:
                dt = time.time() - t0
                t0 = time.time()
                tokens_per_sec = (
                    config.batch_size
                    * config.gradient_accumulation_steps
                    * config.block_size
                    * config.log_interval
                    / dt
                )
                lr = self.optimizer.param_groups[0]["lr"]
                log_msg = (
                    f"step {self.step:>6d} | loss {accum_loss:.4f} | "
                    f"lr {lr:.2e} | grad_norm {grad_norm:.2f} | "
                    f"tok/s {tokens_per_sec:.0f}"
                )
                if lambda_esc > 0:
                    log_msg += f" | λ_esc {lambda_esc:.4f} λ_ash {lambda_ash:.4f}"
                print(log_msg)

                if self.logger:
                    self.logger.log_step(self.step, {
                        **accum_breakdown,
                        "lr": lr,
                        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "tokens_per_sec": tokens_per_sec,
                    })

            # Evaluation
            if self.val_dataloader and self.step % config.eval_interval == 0:
                self._evaluate()

            # Checkpointing
            if self.step % config.save_interval == 0:
                self._save_checkpoint()

        print(f"Training complete at step {self.step}")

    @torch.no_grad()
    def _evaluate(self):
        """Evaluate on validation set."""
        model = self.model
        model.eval()

        losses = []
        esc_means = []

        data_iter = iter(self.val_dataloader)
        for i in range(self.config.eval_iters):
            try:
                x, y = next(data_iter)
            except StopIteration:
                break

            x = x.to(self.device)
            y = y.to(self.device)

            with torch.autocast(device_type=self.device.split(":")[0], dtype=self.dtype):
                logits, loss, cer_info = model(x, y)

            losses.append(loss.item())

            if "esc_scores_per_layer" in cer_info:
                esc = cer_info["esc_scores_per_layer"]
                esc_means.append([s.mean().item() for s in esc])

        val_loss = sum(losses) / len(losses) if losses else float("inf")
        val_ppl = 2.71828 ** val_loss

        log_msg = f"  eval | step {self.step} | loss {val_loss:.4f} | ppl {val_ppl:.2f}"
        if esc_means:
            # Mean ESC per layer across all eval batches
            n_layers = len(esc_means[0])
            layer_means = [
                sum(batch[l] for batch in esc_means) / len(esc_means)
                for l in range(n_layers)
            ]
            log_msg += f" | ESC mean: [{', '.join(f'{m:.3f}' for m in layer_means)}]"
        print(log_msg)

        if self.logger:
            metrics = {"eval/loss": val_loss, "eval/ppl": val_ppl}
            if esc_means:
                n_layers = len(esc_means[0])
                for l in range(n_layers):
                    metrics[f"eval/esc_mean_L{l}"] = sum(
                        batch[l] for batch in esc_means
                    ) / len(esc_means)
            self.logger.log_eval(self.step, metrics)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_checkpoint(suffix="best")

        model.train()

    def _save_checkpoint(self, suffix: str | None = None):
        """Save training checkpoint."""
        name = f"ckpt_{self.step}.pt" if suffix is None else f"ckpt_{suffix}.pt"
        path = Path(self.config.checkpoint_dir) / name
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            step=self.step,
            model_config=self.model_config,
            train_config=self.config,
            metrics={"best_val_loss": self.best_val_loss},
            path=path,
        )
        print(f"  saved checkpoint: {path}")

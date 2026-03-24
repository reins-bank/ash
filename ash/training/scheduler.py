from __future__ import annotations

import math


class CosineWarmupScheduler:
    """Cosine learning rate schedule with linear warmup."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        lr_max: float,
        lr_min: float,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self._step = 0

    def step(self):
        self._step += 1
        lr = self.get_lr(self._step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lr_max * step / max(1, self.warmup_steps)
        if step >= self.max_steps:
            return self.lr_min
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + math.cos(math.pi * progress))


class CERCurriculumScheduler:
    """
    Three-phase CER curriculum scheduler.

    Phase 1 (0 - phase1_end):     lambda_esc=0, lambda_ash=0 (no CER loss)
    Phase 2 (phase1_end - phase2_end): linear ramp to max values
    Phase 3 (phase2_end - 1.0):   full CER loss
    """

    def __init__(
        self,
        lambda_esc_max: float,
        lambda_ash_max: float,
        phase1_end: float = 0.1,
        phase2_end: float = 0.3,
    ):
        self.lambda_esc_max = lambda_esc_max
        self.lambda_ash_max = lambda_ash_max
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end

    def get_lambdas(self, step: int, max_steps: int) -> tuple[float, float]:
        """Returns (lambda_esc, lambda_ash) for the current step."""
        progress = step / max(1, max_steps)

        if progress < self.phase1_end:
            # Phase 1: no CER loss
            return 0.0, 0.0
        elif progress < self.phase2_end:
            # Phase 2: linear ramp
            ramp = (progress - self.phase1_end) / (self.phase2_end - self.phase1_end)
            return self.lambda_esc_max * ramp, self.lambda_ash_max * ramp
        else:
            # Phase 3: full CER
            return self.lambda_esc_max, self.lambda_ash_max

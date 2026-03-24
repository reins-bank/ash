from __future__ import annotations


class Logger:
    """Unified logging to W&B and console."""

    def __init__(self, wandb_enabled: bool = True, project: str = "ashy-small", config: dict | None = None):
        self.wandb_enabled = wandb_enabled
        self.run = None

        if wandb_enabled:
            try:
                import wandb
                self.run = wandb.init(project=project, config=config)
            except Exception as e:
                print(f"W&B init failed: {e}. Continuing without W&B.")
                self.wandb_enabled = False

    def log_step(self, step: int, metrics: dict):
        if self.wandb_enabled and self.run:
            import wandb
            wandb.log(metrics, step=step)

    def log_eval(self, step: int, metrics: dict):
        if self.wandb_enabled and self.run:
            import wandb
            wandb.log(metrics, step=step)

    def finish(self):
        if self.wandb_enabled and self.run:
            import wandb
            wandb.finish()

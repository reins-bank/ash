from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CERConfig:
    """Configuration for CER framework components."""

    enabled: bool = True

    # ESC
    esc_mlp_hidden: int = 64
    esc_input_attn_stats: bool = True
    lambda_esc_max: float = 0.1

    # ASH
    n_ash_heads: int = 2
    ash_lambda_init: float = 0.01
    lambda_ash_reg_max: float = 0.05

    # CR
    cr_dim_divisor: int = 4
    cr_injection_scale: float = 0.1
    cr_persist_across_layers: bool = True

    # Curriculum (fractions of total training)
    curriculum_phase1_end: float = 0.1  # No CER loss
    curriculum_phase2_end: float = 0.3  # Ramp CER loss to max


@dataclass
class ModelConfig:
    """GPT-2 scale model configuration."""

    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    vocab_size: int = 50257
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = True
    activation: str = "gelu"
    norm_type: str = "layernorm"
    cer: CERConfig = field(default_factory=CERConfig)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def ffn_hidden(self) -> int:
        return 4 * self.d_model

    @property
    def d_cr(self) -> int:
        return self.d_model // self.cer.cr_dim_divisor


@dataclass
class ModalConfig:
    """Modal cloud GPU configuration."""

    enabled: bool = False
    gpu: str = "A10G"
    timeout: int = 86400  # 24h cost safety net; override in YAML for longer runs
    data_volume: str = "ash-data"
    checkpoint_volume: str = "ash-checkpoints"
    wandb_secret: str = "wandb-secret"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Optimizer
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 2000
    max_steps: int = 100_000
    lr_decay_to: float = 6e-5

    # Batch
    batch_size: int = 12
    gradient_accumulation_steps: int = 40
    block_size: int = 1024

    # Data
    dataset: str = "openwebtext"
    data_dir: str = "data/"

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    eval_iters: int = 200
    wandb_project: str = "ashy-small"
    wandb_enabled: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints/"
    save_interval: int = 2000

    # Device
    device: str = "cuda"
    compile: bool = True
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = False

    # Modal
    modal: ModalConfig = field(default_factory=ModalConfig)


def _apply_overrides(dc: object, overrides: dict) -> None:
    """Recursively apply dict overrides to a dataclass instance."""
    for k, v in overrides.items():
        if not hasattr(dc, k):
            raise ValueError(f"Unknown config key: {k}")
        current = getattr(dc, k)
        if isinstance(v, dict) and hasattr(current, "__dataclass_fields__"):
            _apply_overrides(current, v)
        else:
            # Coerce type to match the default
            if current is not None and not isinstance(v, type(current)):
                try:
                    v = type(current)(v)
                except (TypeError, ValueError):
                    pass
            setattr(dc, k, v)


def load_config(yaml_path: str | Path) -> tuple[ModelConfig, TrainingConfig]:
    """Load config from YAML, applying overrides on top of defaults."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()

    if "model" in raw:
        _apply_overrides(model_cfg, raw["model"])
    if "training" in raw:
        _apply_overrides(train_cfg, raw["training"])

    # Sync block_size
    train_cfg.block_size = model_cfg.block_size

    return model_cfg, train_cfg

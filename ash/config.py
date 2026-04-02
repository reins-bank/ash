from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Multimodal encoder configs
# ---------------------------------------------------------------------------

@dataclass
class VisionConfig:
    """Vision encoder configuration (SigLIP 2 default)."""

    enabled: bool = True
    encoder_name: str = "google/siglip2-so400m-patch14-384"
    encoder_hidden_dim: int = 1152  # SigLIP 2 SO400M output dim
    image_size: int = 384
    patch_size: int = 14
    max_tiles: int = 6  # dynamic resolution tiling
    projection_dim: int = 6144  # synced to d_model at load time
    projection_layers: int = 2  # 2-layer MLP projector
    freeze_encoder: bool = True
    # Video (frames through vision encoder)
    video_max_frames: int = 32
    video_fps: float = 1.0


@dataclass
class AudioConfig:
    """Audio encoder configuration (Whisper v3 default)."""

    enabled: bool = True
    encoder_name: str = "openai/whisper-large-v3"
    encoder_hidden_dim: int = 1280  # Whisper large-v3 output dim
    sample_rate: int = 16000
    max_duration_sec: float = 30.0
    projection_dim: int = 6144  # synced to d_model at load time
    projection_layers: int = 2
    freeze_encoder: bool = True


@dataclass
class MultimodalConfig:
    """Top-level multimodal configuration. Disabled by default for backwards compat."""

    enabled: bool = False
    vision: VisionConfig = field(default_factory=VisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    # Special tokens (added to vocab)
    image_token: str = "<image>"
    video_token: str = "<video>"
    audio_token: str = "<audio>"
    # CER cross-modal: ESC receives modality-aware attention stats
    cer_cross_modal: bool = True


@dataclass
class MultimodalTrainingStage:
    """Per-stage config for multimodal training pipeline."""

    name: str = "alignment"
    max_steps: int = 5000
    learning_rate: float = 1e-3
    freeze_llm: bool = True
    freeze_vision_encoder: bool = True
    freeze_audio_encoder: bool = True
    freeze_projectors: bool = False
    data_sources: list[str] = field(default_factory=list)
    batch_size: int = 256
    gradient_accumulation_steps: int = 1


# ---------------------------------------------------------------------------
# MoE config
# ---------------------------------------------------------------------------

@dataclass
class MoEConfig:
    """Mixture of Experts configuration."""

    enabled: bool = False
    n_experts: int = 128
    n_active: int = 2  # top-k routing
    shared_expert: bool = True


# ---------------------------------------------------------------------------
# CER config
# ---------------------------------------------------------------------------

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
    """Model configuration. Supports GPT-2 scale (Ashy-Small) through full 17Bx128E."""

    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int | None = None  # GQA KV heads; None = MHA (n_kv_head == n_head)
    d_model: int = 768
    vocab_size: int = 50257
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = True
    activation: str = "gelu"  # gelu | swiglu
    norm_type: str = "layernorm"  # layernorm | rmsnorm
    cer: CERConfig = field(default_factory=CERConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)

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
class StorageConfig:
    """Data storage backend. Set backend to 's3' for remote bucket storage.

    When backend is 's3', bucket and endpoint are resolved from environment
    variables (ASH_S3_BUCKET, ASH_S3_ENDPOINT_URL) with YAML fields as
    optional overrides.  Credentials use AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
    """

    backend: str = "local"       # "local" | "s3"
    bucket: str = ""             # override ASH_S3_BUCKET env var
    endpoint_url: str = ""       # override ASH_S3_ENDPOINT_URL env var


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
    data_sources: list[str] = field(default_factory=list)

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

    # Storage
    storage: StorageConfig = field(default_factory=StorageConfig)

    # Multimodal training stages (empty = text-only training)
    multimodal_stages: list[MultimodalTrainingStage] = field(default_factory=list)


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


def _parse_multimodal_stages(raw_stages: list[dict]) -> list[MultimodalTrainingStage]:
    """Parse a list of raw dicts into MultimodalTrainingStage instances."""
    stages = []
    for raw_stage in raw_stages:
        stage = MultimodalTrainingStage()
        for k, v in raw_stage.items():
            if hasattr(stage, k):
                setattr(stage, k, v)
        stages.append(stage)
    return stages


def load_config(yaml_path: str | Path) -> tuple[ModelConfig, TrainingConfig]:
    """Load config from YAML, applying overrides on top of defaults."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()

    if "model" in raw:
        _apply_overrides(model_cfg, raw["model"])
    if "training" in raw:
        # Handle multimodal_stages list separately (not a nested dataclass)
        training_raw = raw["training"]
        stages_raw = training_raw.pop("multimodal_stages", None)
        _apply_overrides(train_cfg, training_raw)
        if stages_raw:
            train_cfg.multimodal_stages = _parse_multimodal_stages(stages_raw)

    # Sync block_size
    train_cfg.block_size = model_cfg.block_size

    # Sync multimodal projection dims with d_model
    if model_cfg.multimodal.enabled:
        model_cfg.multimodal.vision.projection_dim = model_cfg.d_model
        model_cfg.multimodal.audio.projection_dim = model_cfg.d_model

    return model_cfg, train_cfg

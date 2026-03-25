"""Launch Ash training on Modal serverless GPUs."""
from __future__ import annotations

from dataclasses import asdict

from ash.config import ModelConfig, TrainingConfig


def run_on_modal(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    resume_path: str | None = None,
) -> None:
    """Dispatch training to a Modal GPU container.

    Streams logs back to the local terminal in real time.
    """
    import modal

    app = modal.App("ash-training")

    image = (
        modal.Image.debian_slim(python_version="3.14")
        .pip_install(
            "torch>=2.1",
            "tiktoken>=0.5",
            "wandb",
            "datasets",
            "numpy",
            "pyyaml",
            "tqdm",
        )
        .add_local_python_source("ash")
    )

    data_vol = modal.Volume.from_name(
        train_cfg.modal.data_volume, create_if_missing=True
    )
    ckpt_vol = modal.Volume.from_name(
        train_cfg.modal.checkpoint_volume, create_if_missing=True
    )

    secrets = []
    if train_cfg.wandb_enabled:
        secrets.append(modal.Secret.from_name(train_cfg.modal.wandb_secret))

    # Serialize configs as plain dicts for the remote boundary
    config_dict = {
        "model": asdict(model_cfg),
        "training": asdict(train_cfg),
    }

    def _train_remote(config_dict: dict, resume_path: str | None = None):
        from ash.config import ModelConfig, TrainingConfig, _apply_overrides
        from ash.training.runner import run_training

        # Reconstruct config dataclasses on the remote side
        model_cfg = ModelConfig()
        train_cfg = TrainingConfig()
        _apply_overrides(model_cfg, config_dict["model"])
        _apply_overrides(train_cfg, config_dict["training"])
        train_cfg.block_size = model_cfg.block_size

        # Override paths for Modal volumes
        train_cfg.data_dir = "/data"
        train_cfg.checkpoint_dir = "/checkpoints"
        train_cfg.device = "cuda"

        print(f"[Modal] GPU: {config_dict['training']['modal']['gpu']}")
        run_training(model_cfg, train_cfg, resume_path=resume_path)

    # Programmatic decoration so GPU tier comes from config
    train_fn = app.function(
        image=image,
        gpu=train_cfg.modal.gpu,
        timeout=train_cfg.modal.timeout,
        volumes={"/data": data_vol, "/checkpoints": ckpt_vol},
        secrets=secrets,
        serialized=True,
    )(_train_remote)

    gpu = train_cfg.modal.gpu
    timeout = train_cfg.modal.timeout
    print(f"Launching training on Modal (GPU: {gpu}, timeout: {timeout}s)...")

    with app.run():
        train_fn.remote(config_dict, resume_path)

    # Commit checkpoint volume so results persist
    ckpt_vol.commit()

    print("Modal training finished.")

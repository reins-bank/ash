"""Launch Ash training on Modal serverless GPUs."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ash.config import ModelConfig, TrainingConfig

RUN_INFO_PATH = Path(".modal_run.json")


def _save_run_info(call_id: str, train_cfg: TrainingConfig) -> None:
    """Persist run metadata so modal_status can reconnect."""
    info = {
        "call_id": call_id,
        "gpu": train_cfg.modal.gpu,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_volume": train_cfg.modal.checkpoint_volume,
        "storage_backend": train_cfg.storage.backend,
        "data_volume": train_cfg.modal.data_volume if train_cfg.storage.backend == "local" else None,
        "bucket": (train_cfg.storage.bucket or os.environ.get("ASH_S3_BUCKET"))
        if train_cfg.storage.backend == "s3"
        else None,
    }
    RUN_INFO_PATH.write_text(json.dumps(info, indent=2) + "\n")


def run_on_modal(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    resume_path: str | None = None,
) -> None:
    """Dispatch training to a Modal GPU container in the background.

    The function returns immediately. Use ``make modal-status`` to check
    progress, ``make modal-checkpoints`` to list saved checkpoints.
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

    use_s3 = train_cfg.storage.backend == "s3"

    # Data mount: S3 CloudBucketMount or Modal Volume
    if use_s3:
        from ash.data.storage import resolve_s3_config

        bucket, endpoint_url, access_key, secret_key = resolve_s3_config(
            train_cfg.storage
        )
        s3_secret = modal.Secret.from_dict({
            "AWS_ACCESS_KEY_ID": access_key,
            "AWS_SECRET_ACCESS_KEY": secret_key,
        })
        data_mount = modal.CloudBucketMount(
            bucket_name=bucket,
            bucket_endpoint_url=endpoint_url,
            secret=s3_secret,
            read_only=True,
        )
        volumes_dict = {"/data": data_mount}
    else:
        data_vol = modal.Volume.from_name(
            train_cfg.modal.data_volume, create_if_missing=True
        )
        volumes_dict = {"/data": data_vol}

    ckpt_vol = modal.Volume.from_name(
        train_cfg.modal.checkpoint_volume, create_if_missing=True
    )
    volumes_dict["/checkpoints"] = ckpt_vol

    secrets = []
    if train_cfg.wandb_enabled:
        secrets.append(modal.Secret.from_name(train_cfg.modal.wandb_secret))

    # Serialize configs as plain dicts for the remote boundary
    config_dict = {
        "model": asdict(model_cfg),
        "training": asdict(train_cfg),
    }

    ckpt_volume_name = train_cfg.modal.checkpoint_volume

    def _train_remote(config_dict: dict, resume_path: str | None = None):
        import modal as _modal
        from ash.config import ModelConfig, TrainingConfig, _apply_overrides
        from ash.training.runner import run_training

        # Reconstruct config dataclasses on the remote side
        model_cfg = ModelConfig()
        train_cfg = TrainingConfig()
        _apply_overrides(model_cfg, config_dict["model"])
        _apply_overrides(train_cfg, config_dict["training"])
        train_cfg.block_size = model_cfg.block_size

        # Map data_dir onto the /data mount point.
        # Both CloudBucketMount (S3) and Volume mount at /data.
        from ash.data.storage import dataset_prefix
        prefix = dataset_prefix(train_cfg.data_dir)
        train_cfg.data_dir = f"/data/{prefix}" if prefix else "/data"

        # Override storage backend to local — data is already mounted as a filesystem
        train_cfg.storage.backend = "local"

        # Verify data exists — fail loudly rather than silently training on random tokens
        from pathlib import Path
        for split in ("train", "val"):
            p = Path(train_cfg.data_dir) / f"{split}.bin"
            if not p.exists():
                avail = list(Path("/data").rglob("*.bin"))
                msg = f"Data not found: {p}"
                if avail:
                    msg += f"\nAvailable .bin files: {[str(f) for f in avail]}"
                else:
                    backend = config_dict["training"]["storage"]["backend"]
                    if backend == "s3":
                        msg += "\nNo .bin files found. Check your S3 bucket contents."
                    else:
                        msg += "\nNo .bin files found on volume. Run: make modal-upload-data"
                raise FileNotFoundError(msg)

        train_cfg.checkpoint_dir = "/checkpoints"
        train_cfg.device = "cuda"

        print(f"[Modal] GPU: {config_dict['training']['modal']['gpu']}")
        run_training(model_cfg, train_cfg, resume_path=resume_path)

        # Commit checkpoint volume so results persist after container exits
        _ckpt_vol = _modal.Volume.from_name(ckpt_volume_name)
        _ckpt_vol.commit()
        print("[Modal] Checkpoints committed to volume.")

    # Programmatic decoration so GPU tier comes from config
    train_fn = app.function(
        image=image,
        gpu=train_cfg.modal.gpu,
        timeout=train_cfg.modal.timeout,
        volumes=volumes_dict,
        secrets=secrets,
        serialized=True,
    )(_train_remote)

    gpu = train_cfg.modal.gpu
    timeout_h = train_cfg.modal.timeout / 3600
    if use_s3:
        bucket = train_cfg.storage.bucket or os.environ.get("ASH_S3_BUCKET", "")
        print(f"Dispatching training on Modal (GPU: {gpu}, data: s3://{bucket})...")
    else:
        print(f"Dispatching training on Modal (GPU: {gpu}, timeout: {timeout_h:.0f}h)...")

    with app.run(detach=True):
        fc = train_fn.spawn(config_dict, resume_path)
        call_id = fc.object_id

    _save_run_info(call_id, train_cfg)

    print(f"Training dispatched! Call ID: {call_id}")
    print()
    print("Monitor with:")
    print("  make modal-status        # check if still running")
    print("  make modal-checkpoints   # list saved checkpoints")
    print("  make modal-download CHECKPOINT=ckpt_best.pt")
    print("  make modal-cancel        # stop the run")

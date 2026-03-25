"""Check on Modal training runs, list/download checkpoints, cancel jobs.

Usage:
    python -m ash.infra.modal_status status
    python -m ash.infra.modal_status checkpoints
    python -m ash.infra.modal_status download <checkpoint> [--out-dir DIR]
    python -m ash.infra.modal_status cancel
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

RUN_INFO_PATH = Path(".modal_run.json")


def _load_run_info() -> dict:
    if not RUN_INFO_PATH.exists():
        print("No active Modal run found (.modal_run.json missing).")
        print("Start a run with: make modal-train")
        sys.exit(1)
    return json.loads(RUN_INFO_PATH.read_text())


def cmd_status(args: argparse.Namespace) -> None:
    """Check whether the most recent Modal training run is still active."""
    import modal

    info = _load_run_info()
    call_id = info["call_id"]
    gpu = info.get("gpu", "?")
    started = info.get("started_at", "?")

    print(f"Run {call_id}")
    print(f"  GPU: {gpu}")
    print(f"  Started: {started}")

    fc = modal.FunctionCall.from_id(call_id)
    try:
        fc.get(timeout=0)
        print("  Status: completed")
    except TimeoutError:
        print("  Status: running")
    except modal.exception.FunctionTimeoutError:
        print("  Status: timed out")
    except Exception as exc:
        print(f"  Status: failed ({exc})")


def cmd_checkpoints(args: argparse.Namespace) -> None:
    """List checkpoint files on the Modal volume."""
    import modal

    info = _load_run_info()
    volume_name = info.get("checkpoint_volume", "ash-checkpoints")

    vol = modal.Volume.from_name(volume_name)
    entries = sorted(vol.listdir("/", recursive=True), key=lambda e: e.path)

    if not entries:
        print(f"No files found on volume '{volume_name}'")
        return

    print(f"{volume_name}:")
    for entry in entries:
        size_mb = entry.size / 1e6
        mtime = datetime.fromtimestamp(entry.mtime).strftime("%Y-%m-%d %H:%M") if entry.mtime else "?"
        print(f"  {entry.path}  ({size_mb:.1f} MB, {mtime})")


def cmd_download(args: argparse.Namespace) -> None:
    """Download a checkpoint from the Modal volume."""
    import modal

    info = _load_run_info()
    volume_name = info.get("checkpoint_volume", "ash-checkpoints")
    checkpoint = args.checkpoint
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vol = modal.Volume.from_name(volume_name)

    remote_path = checkpoint if checkpoint.startswith("/") else f"/{checkpoint}"
    local_path = out_dir / Path(checkpoint).name

    print(f"Downloading {remote_path} -> {local_path}")
    with open(local_path, "wb") as f:
        for chunk in vol.read_file(remote_path):
            f.write(chunk)

    size_mb = local_path.stat().st_size / 1e6
    print(f"Done ({size_mb:.1f} MB)")


def cmd_cancel(args: argparse.Namespace) -> None:
    """Cancel the most recent Modal training run."""
    import modal

    info = _load_run_info()
    call_id = info["call_id"]

    fc = modal.FunctionCall.from_id(call_id)
    fc.cancel()
    print(f"Cancelled run {call_id}")


def main():
    parser = argparse.ArgumentParser(
        prog="modal_status",
        description="Monitor and manage Modal training runs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Check if training is still running")
    sub.add_parser("checkpoints", help="List checkpoint files on the volume")

    dl = sub.add_parser("download", help="Download a checkpoint from the volume")
    dl.add_argument("checkpoint", help="Checkpoint filename (e.g. ckpt_best.pt)")
    dl.add_argument("--out-dir", default="checkpoints/", help="Local download directory")

    sub.add_parser("cancel", help="Cancel the running training job")

    args = parser.parse_args()
    commands = {
        "status": cmd_status,
        "checkpoints": cmd_checkpoints,
        "download": cmd_download,
        "cancel": cmd_cancel,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()

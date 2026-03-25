"""Upload local training data to a Modal Volume.

Handles both layouts:
  - Legacy flat:    data/train.bin, data/val.bin  (from prepare_data.py)
  - Pipeline nested: data/clean/<source>/tokens.bin  (from pipeline tokenize step)

Directory structure is preserved on the volume so training can point
``data_dir`` at the right path.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def upload_data(local_data_dir: str, volume_name: str = "ash-data") -> None:
    """Upload all .bin files under *local_data_dir* to a Modal Volume.

    Preserves the directory tree relative to *local_data_dir*, so
    ``data/clean/wikipedia/tokens.bin`` becomes ``/clean/wikipedia/tokens.bin``
    on the volume when ``--data-dir data/`` is used.
    """
    import modal

    data_dir = Path(local_data_dir)
    vol = modal.Volume.from_name(volume_name, create_if_missing=True)

    files = sorted(data_dir.rglob("*.bin"))
    if not files:
        print(f"No .bin files found under {data_dir}")
        return

    with vol.batch_upload() as batch:
        for f in files:
            remote_path = "/" + str(f.relative_to(data_dir))
            size_mb = f.stat().st_size / 1e6
            print(f"  {f} -> {remote_path} ({size_mb:.1f} MB)")
            batch.put_file(str(f), remote_path)

    print(f"Uploaded {len(files)} file(s) to volume '{volume_name}'")


def main():
    parser = argparse.ArgumentParser(
        description="Upload training data (.bin files) to a Modal Volume",
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Local data directory (searched recursively for .bin files)",
    )
    parser.add_argument(
        "--volume", type=str, default="ash-data",
        help="Modal volume name (default: ash-data)",
    )
    args = parser.parse_args()

    upload_data(args.data_dir, args.volume)


if __name__ == "__main__":
    main()

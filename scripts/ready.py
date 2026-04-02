"""
Prepare and train Ash models.

Usage:
    python scripts/ready.py <model>              # prepare data for a model
    python scripts/ready.py <model> --dry-run     # show what would be done
    python scripts/ready.py <model> --train       # prepare data then train
    python scripts/ready.py --config <path>       # prepare data from config file
    python scripts/ready.py --list                # list available models

Models:
    debug               Tiny smoke test (CPU, random data — nothing to prepare)
    ashy-small-wikitext  WikiText proof-of-concept (~30 min on A10G)
    ashy-small           GPT-2 117M + CER, full training on Modal
    ash-full             17Bx128E MoE, text-only
    ash-full-multimodal  17Bx128E MoE + vision + audio

Examples:
    python scripts/ready.py ashy-small            # fetch & process all data sources
    python scripts/ready.py ashy-small --train    # prepare then launch Modal training
    python scripts/ready.py --list                # show all models and their sources
"""
from __future__ import annotations

import argparse
import sys

from ash.config import load_config


# ---------------------------------------------------------------------------
# Model registry — maps short names to config files
# ---------------------------------------------------------------------------

MODELS = {
    "debug": {
        "config": "configs/ashy_small_debug.yaml",
        "description": "Tiny smoke test (CPU, random data)",
    },
    "ashy-small-wikitext": {
        "config": "configs/ashy_small_modal_wikitext.yaml",
        "description": "WikiText proof-of-concept (~30 min on A10G)",
    },
    "ashy-small": {
        "config": "configs/ashy_small_modal_s3.yaml",
        "description": "GPT-2 117M + CER, full training",
    },
    "ash-full": {
        "config": "configs/ash_full.yaml",
        "description": "17Bx128E MoE, text-only",
    },
    "ash-full-multimodal": {
        "config": "configs/ash_full_multimodal.yaml",
        "description": "17Bx128E MoE + vision + audio",
    },
}


def cmd_list():
    """List all available models and their data sources."""
    print(f"\n{'Model':<24} {'Sources':<6} {'Storage':<8} {'Description'}")
    print("-" * 80)
    for name, info in MODELS.items():
        _, train_cfg = load_config(info["config"])
        n_sources = len(train_cfg.data_sources)
        backend = train_cfg.storage.backend
        print(f"{name:<24} {n_sources:<6} {backend:<8} {info['description']}")
        if train_cfg.data_sources:
            print(f"  {'sources:':<22} {', '.join(train_cfg.data_sources)}")
    print()


def cmd_prepare(config_path: str, model_name: str | None, dry_run: bool):
    """Prepare all data sources for a model."""
    _, train_cfg = load_config(config_path)

    if not train_cfg.data_sources:
        label = model_name or config_path
        print(f"No data sources configured for '{label}'.")
        if model_name == "debug":
            print("Debug model uses random data — nothing to prepare.")
        return

    print(f"Preparing data for: {model_name or config_path}")
    print(f"  Config:     {config_path}")
    print(f"  Data dir:   {train_cfg.data_dir}")
    print(f"  Storage:    {train_cfg.storage.backend}")
    print(f"  Sources:    {', '.join(train_cfg.data_sources)}")
    print()

    from ash.pipeline.source import load_source, validate_source
    from ash.pipeline.fetch import fetch_source
    from ash.pipeline.clean import clean_source
    from ash.data.storage import PipelineFS

    # Load all requested sources
    sources = []
    for name in train_cfg.data_sources:
        try:
            ds = load_source(f"data_sources/{name}.yaml")
            sources.append(ds)
        except FileNotFoundError:
            print(f"  Warning: data source '{name}' not found, skipping")

    if not sources:
        print("No valid data sources found.")
        return

    # Validate storage backends BEFORE any expensive work.
    # This catches missing boto3, missing credentials, etc. immediately.
    if not dry_run:
        s3_sources = [ds for ds in sources if ds.storage.backend == "s3"]
        if s3_sources:
            print(f"Validating S3 access for {len(s3_sources)} remote source(s)...")
            fs = PipelineFS(s3_sources[0].storage)
            fs.validate()  # exits with clear error if boto3/creds missing
            print("  S3 access OK")
            print()

        # Also validate the merge target if training config uses S3
        if train_cfg.storage.backend == "s3":
            merge_fs = PipelineFS(train_cfg.storage)
            merge_fs.validate()

    # Phase 1: Fetch + clean each source
    token_files = []
    print(f"Phase 1: Fetch and process {len(sources)} source(s)")
    print("-" * 60)

    for ds in sources:
        errors = validate_source(ds)
        if errors:
            print(f"  [{ds.name}] INVALID: {errors}")
            continue

        storage_label = f" [s3]" if ds.storage.backend == "s3" else ""
        print(f"  [{ds.name}] Fetching{storage_label}...")

        try:
            fetch_result = fetch_source(ds, dry_run=dry_run)
            status = fetch_result.get("status", "?")
            if "documents_fetched" in fetch_result:
                print(f"    Fetched: {fetch_result['documents_fetched']} documents ({status})")
            else:
                print(f"    Fetch: {status}")
        except Exception as e:
            print(f"    Fetch ERROR: {e}")
            continue

        print(f"  [{ds.name}] Cleaning...")
        try:
            clean_result = clean_source(ds, dry_run=dry_run)
            status = clean_result.get("status", "?")
            if "documents_output" in clean_result:
                print(f"    Cleaned: {clean_result['documents_output']} documents ({status})")
            else:
                print(f"    Clean: {status}")

            if "tokens" in clean_result:
                print(f"    Tokens: {clean_result['tokens']:,}")
            if "token_file" in clean_result:
                token_files.append(clean_result["token_file"])
        except Exception as e:
            print(f"    Clean ERROR: {e}")
            continue

    print()

    # Phase 2: Merge tokens into train.bin / val.bin
    if dry_run:
        print("Phase 2: Merge (skipped — dry run)")
        if token_files:
            print(f"  Would merge {len(token_files)} token file(s) into {train_cfg.data_dir}/")
        print("\nDry run complete.")
        return

    if not token_files:
        print("Phase 2: Merge (skipped — no token files produced)")
        print("  Some sources may not include a 'tokenize' cleaning step.")
        return

    print(f"Phase 2: Merge {len(token_files)} token file(s) -> {train_cfg.data_dir}/")
    print("-" * 60)

    from ash.data.storage import PipelineFS
    from ash.pipeline.merge import merge_token_files

    # Use S3 storage if the training config specifies it
    fs = PipelineFS(train_cfg.storage) if train_cfg.storage.backend == "s3" else None
    result = merge_token_files(token_files, train_cfg.data_dir, val_ratio=0.1, fs=fs)

    print(f"  Train: {result['train_tokens']:,} tokens -> {result['train_path']}")
    print(f"  Val:   {result['val_tokens']:,} tokens  -> {result['val_path']}")
    print()
    print("Data ready for training!")


def cmd_train(config_path: str, model_name: str | None, resume: str | None):
    """Launch training (dispatches to Modal if configured)."""
    model_cfg, train_cfg = load_config(config_path)

    label = model_name or config_path
    print(f"Training: {label}")
    print(f"  Config: {config_path}")

    if train_cfg.modal.enabled:
        from ash.infra.modal_runner import run_on_modal
        run_on_modal(model_cfg, train_cfg, resume_path=resume)
    else:
        from ash.training.runner import run_training
        run_training(model_cfg, train_cfg, resume_path=resume)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data and train Ash models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models:
  debug               Tiny smoke test (CPU, random data)
  ashy-small-wikitext  WikiText proof-of-concept (~30 min)
  ashy-small           GPT-2 117M + CER, full training
  ash-full             17Bx128E MoE, text-only
  ash-full-multimodal  17Bx128E MoE + vision + audio

Examples:
  %(prog)s ashy-small              # prepare data
  %(prog)s ashy-small --train      # prepare + train
  %(prog)s --list                  # show models
""",
    )
    parser.add_argument("model", nargs="?", help="Model name (see --list)")
    parser.add_argument("--config", type=str, help="Path to YAML config (instead of model name)")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--train", action="store_true", help="Launch training after prepare")
    parser.add_argument("--resume", type=str, help="Checkpoint to resume from (with --train)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    if args.list:
        cmd_list()
        return

    # Resolve config path
    if args.config:
        config_path = args.config
        model_name = None
    elif args.model:
        if args.model not in MODELS:
            print(f"Unknown model: '{args.model}'")
            print(f"Available models: {', '.join(MODELS.keys())}")
            print("Use --list for details.")
            sys.exit(1)
        config_path = MODELS[args.model]["config"]
        model_name = args.model
    else:
        parser.print_help()
        sys.exit(1)

    # Prepare
    cmd_prepare(config_path, model_name, args.dry_run)

    # Train
    if args.train and not args.dry_run:
        print("=" * 60)
        cmd_train(config_path, model_name, args.resume)


if __name__ == "__main__":
    main()

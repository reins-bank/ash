"""
Data pipeline CLI — list, fetch, and clean data sources defined by YAML.

Usage:
    python scripts/pipeline.py list [--sources-dir data_sources/]
    python scripts/pipeline.py fetch [--source fineweb] [--all] [--dry-run]
    python scripts/pipeline.py clean [--source fineweb] [--all] [--dry-run]
    python scripts/pipeline.py run   [--source fineweb] [--all] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ash.pipeline.source import load_source, load_all_sources, validate_source
from ash.pipeline.fetch import fetch_source
from ash.pipeline.clean import clean_source
from ash.pipeline.runner import run_pipeline


def cmd_list(args: argparse.Namespace) -> None:
    """List all available data sources."""
    sources = load_all_sources(args.sources_dir)

    if not sources:
        print(f"No source YAMLs found in {args.sources_dir}")
        return

    print(f"{'Name':<20} {'Type':<14} {'License':<20} {'Use':<12} {'Description'}")
    print("-" * 90)
    for ds in sources:
        errors = validate_source(ds)
        status = "" if not errors else f" [{len(errors)} errors]"
        print(
            f"{ds.name:<20} {ds.source.type:<14} {ds.license.name:<20} "
            f"{ds.license.allowed_use:<12} {ds.description[:40]}{status}"
        )


def _resolve_sources(args: argparse.Namespace) -> list:
    """Resolve which sources to operate on from --source / --all flags."""
    if args.all:
        return load_all_sources(args.sources_dir)
    elif args.source:
        # Try exact file path first, then look in sources_dir
        path = Path(args.source)
        if not path.exists():
            path = Path(args.sources_dir) / f"{args.source}.yaml"
        if not path.exists():
            print(f"Error: source not found: {args.source}", file=sys.stderr)
            print(f"  Tried: {args.source}, {path}", file=sys.stderr)
            sys.exit(1)
        return [load_source(path)]
    else:
        print("Error: specify --source <name> or --all", file=sys.stderr)
        sys.exit(1)


def _print_result(result: dict, verbose: bool = False) -> None:
    """Print a pipeline result."""
    name = result.get("source_name", "?")
    status = result.get("status", "?")

    if "warning" in result:
        print(f"  WARNING: {result['warning']}")

    if status == "invalid":
        print(f"  [{name}] INVALID:")
        for err in result.get("errors", []):
            print(f"    - {err}")
        return

    if status == "error":
        print(f"  [{name}] ERROR: {result.get('error', '?')}")
        return

    for stage_name, stage_result in result.get("stages", {}).items():
        stage_status = stage_result.get("status", "?")
        print(f"  [{name}] {stage_name}: {stage_status}")
        if verbose:
            for k, v in stage_result.items():
                if k not in ("status", "source_name"):
                    print(f"    {k}: {v}")


def cmd_fetch(args: argparse.Namespace) -> None:
    """Fetch data from one or all sources."""
    sources = _resolve_sources(args)
    for ds in sources:
        print(f"Fetching: {ds.name} ({ds.source.type}://{ds.source.path})")
        errors = validate_source(ds)
        if errors:
            print(f"  INVALID: {errors}")
            continue
        try:
            result = fetch_source(ds, dry_run=args.dry_run, base_dir=args.base_dir)
            status = result.get("status", "?")
            print(f"  Status: {status}")
            if "documents_fetched" in result:
                print(f"  Documents: {result['documents_fetched']}")
            if "output_file" in result:
                print(f"  Output: {result['output_file']}")
        except Exception as e:
            print(f"  ERROR: {e}")


def cmd_clean(args: argparse.Namespace) -> None:
    """Clean/standardize data from one or all sources."""
    sources = _resolve_sources(args)
    for ds in sources:
        print(f"Cleaning: {ds.name} ({len(ds.cleaning.steps)} steps)")
        errors = validate_source(ds)
        if errors:
            print(f"  INVALID: {errors}")
            continue
        try:
            result = clean_source(ds, dry_run=args.dry_run, base_dir=args.base_dir)
            status = result.get("status", "?")
            print(f"  Status: {status}")
            if "documents_output" in result:
                print(
                    f"  Documents: {result['documents_read']} read, "
                    f"{result['documents_filtered']} filtered, "
                    f"{result['documents_output']} output"
                )
            if "tokens" in result:
                print(f"  Tokens: {result['tokens']:,}")
        except Exception as e:
            print(f"  ERROR: {e}")


def cmd_run(args: argparse.Namespace) -> None:
    """Run the full pipeline (fetch + clean) for one or all sources."""
    sources = _resolve_sources(args)
    print(f"Running pipeline for {len(sources)} source(s), dry_run={args.dry_run}")
    print()

    results = run_pipeline(sources, dry_run=args.dry_run, base_dir=args.base_dir)
    for result in results:
        _print_result(result, verbose=True)
    print()
    print(f"Done. {len(results)} source(s) processed.")


def main():
    parser = argparse.ArgumentParser(
        description="Ash data pipeline — manage, fetch, and clean training data sources",
    )
    parser.add_argument(
        "--sources-dir", type=str, default="data_sources",
        help="Directory containing source YAML definitions",
    )
    parser.add_argument(
        "--base-dir", type=str, default=".",
        help="Base directory for resolving output paths",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    sub_list = subparsers.add_parser("list", help="List available data sources")
    sub_list.set_defaults(func=cmd_list)

    # fetch
    sub_fetch = subparsers.add_parser("fetch", help="Fetch data from source(s)")
    sub_fetch.add_argument("--source", type=str, help="Source name or YAML path")
    sub_fetch.add_argument("--all", action="store_true", help="Fetch all sources")
    sub_fetch.add_argument("--dry-run", action="store_true", help="Show what would be done")
    sub_fetch.set_defaults(func=cmd_fetch)

    # clean
    sub_clean = subparsers.add_parser("clean", help="Clean/standardize fetched data")
    sub_clean.add_argument("--source", type=str, help="Source name or YAML path")
    sub_clean.add_argument("--all", action="store_true", help="Clean all sources")
    sub_clean.add_argument("--dry-run", action="store_true", help="Show what would be done")
    sub_clean.set_defaults(func=cmd_clean)

    # run (fetch + clean)
    sub_run = subparsers.add_parser("run", help="Run full pipeline (fetch + clean)")
    sub_run.add_argument("--source", type=str, help="Source name or YAML path")
    sub_run.add_argument("--all", action="store_true", help="Run all sources")
    sub_run.add_argument("--dry-run", action="store_true", help="Show what would be done")
    sub_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

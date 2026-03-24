"""Pipeline orchestrator — coordinates fetch and clean across sources."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ash.pipeline.source import DataSource, load_source, load_all_sources, validate_source
from ash.pipeline.fetch import fetch_source
from ash.pipeline.clean import clean_source


def run_pipeline(
    sources: list[DataSource],
    stages: list[str] | None = None,
    dry_run: bool = False,
    base_dir: str | Path = ".",
) -> list[dict[str, Any]]:
    """Run the full pipeline (fetch + clean) for a list of sources.

    Args:
        sources: List of DataSource definitions to process.
        stages: Which stages to run. None or ["fetch", "clean"] for both.
                 Can be ["fetch"] or ["clean"] for just one stage.
        dry_run: If True, show what would be done without executing.
        base_dir: Base directory for resolving relative output paths.

    Returns:
        List of result dicts, one per source, with status and metadata.
    """
    if stages is None:
        stages = ["fetch", "clean"]

    results = []
    for ds in sources:
        entry: dict[str, Any] = {"source_name": ds.name, "stages": {}}

        # Validate first
        errors = validate_source(ds)
        if errors:
            entry["status"] = "invalid"
            entry["errors"] = errors
            results.append(entry)
            continue

        if ds.license.allowed_use == "unknown":
            entry["warning"] = (
                f"Source '{ds.name}' has allowed_use=unknown — "
                "quarantine until reviewed per TRAINING.md policy"
            )

        try:
            if "fetch" in stages:
                fetch_result = fetch_source(ds, dry_run=dry_run, base_dir=base_dir)
                entry["stages"]["fetch"] = fetch_result

            if "clean" in stages:
                clean_result = clean_source(ds, dry_run=dry_run, base_dir=base_dir)
                entry["stages"]["clean"] = clean_result

            entry["status"] = "dry_run" if dry_run else "complete"
        except Exception as e:
            entry["status"] = "error"
            entry["error"] = str(e)

        results.append(entry)

    return results

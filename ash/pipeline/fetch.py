"""Fetch data from sources defined in YAML."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ash.pipeline.source import DataSource


def _fetch_huggingface(ds: DataSource, output_dir: Path, dry_run: bool) -> dict[str, Any]:
    """Fetch from a HuggingFace dataset."""
    info = {
        "method": "huggingface",
        "dataset": ds.source.path,
        "subset": ds.source.subset,
        "split": ds.source.split,
        "streaming": ds.fetch.streaming,
    }

    if dry_run:
        info["status"] = "dry_run"
        return info

    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": ds.source.path,
        "split": ds.source.split,
        "streaming": ds.fetch.streaming,
        "trust_remote_code": True,
    }
    if ds.source.subset:
        kwargs["name"] = ds.source.subset
    if ds.source.revision:
        kwargs["revision"] = ds.source.revision

    dataset = load_dataset(**kwargs)

    # Write documents as JSONL
    out_file = output_dir / "documents.jsonl"
    count = 0
    with open(out_file, "w") as f:
        for doc in dataset:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1
            if ds.fetch.max_documents and count >= ds.fetch.max_documents:
                break

    info["documents_fetched"] = count
    info["output_file"] = str(out_file)
    info["status"] = "complete"
    return info


def _fetch_url(ds: DataSource, output_dir: Path, dry_run: bool) -> dict[str, Any]:
    """Fetch from a URL (download)."""
    import urllib.request

    info = {"method": "url", "url": ds.source.path}

    if dry_run:
        info["status"] = "dry_run"
        return info

    filename = ds.source.path.rsplit("/", 1)[-1] or "download"
    out_file = output_dir / filename
    urllib.request.urlretrieve(ds.source.path, str(out_file))

    info["output_file"] = str(out_file)
    info["status"] = "complete"
    return info


def _fetch_local(ds: DataSource, output_dir: Path, dry_run: bool) -> dict[str, Any]:
    """Copy/symlink from a local path."""
    src = Path(ds.source.path)
    info = {"method": "local", "source_path": str(src)}

    if dry_run:
        info["status"] = "dry_run"
        return info

    if not src.exists():
        raise FileNotFoundError(f"Local source not found: {src}")

    if src.is_file():
        dest = output_dir / src.name
        shutil.copy2(str(src), str(dest))
        info["output_file"] = str(dest)
    elif src.is_dir():
        # Copy contents into output_dir so clean step finds files at expected paths
        for item in src.iterdir():
            dest = output_dir / item.name
            if item.is_file():
                shutil.copy2(str(item), str(dest))
            elif item.is_dir():
                if dest.exists():
                    shutil.rmtree(str(dest))
                shutil.copytree(str(item), str(dest))

    info["status"] = "complete"
    return info


_FETCH_HANDLERS = {
    "huggingface": _fetch_huggingface,
    "url": _fetch_url,
    "local": _fetch_local,
}


def fetch_source(ds: DataSource, dry_run: bool = False, base_dir: str | Path = ".") -> dict[str, Any]:
    """Fetch a data source according to its YAML definition.

    Returns a dict with status information about the fetch.
    """
    base_dir = Path(base_dir)
    output_dir = base_dir / ds.fetch.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    handler = _FETCH_HANDLERS.get(ds.source.type)
    if handler is None:
        raise ValueError(f"Unknown source type: {ds.source.type}")

    result = handler(ds, output_dir, dry_run)
    result["source_name"] = ds.name
    result["output_dir"] = str(output_dir)
    return result

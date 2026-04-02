"""Fetch data from sources defined in YAML."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ash.pipeline.source import DataSource
from ash.data.storage import PipelineFS


def _fetch_huggingface(
    ds: DataSource, output_dir: str, dry_run: bool, fs: PipelineFS,
) -> dict[str, Any]:
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

    # Write documents as JSONL (streams to S3 or local)
    out_file = fs.join(output_dir, "documents.jsonl")
    count = 0
    with fs.open_write(out_file) as f:
        for doc in dataset:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1
            if ds.fetch.max_documents and count >= ds.fetch.max_documents:
                break

    info["documents_fetched"] = count
    info["output_file"] = out_file
    info["status"] = "complete"
    return info


def _fetch_url(
    ds: DataSource, output_dir: str, dry_run: bool, fs: PipelineFS,
) -> dict[str, Any]:
    """Fetch from a URL (download)."""
    import urllib.request
    import tempfile

    info = {"method": "url", "url": ds.source.path}

    if dry_run:
        info["status"] = "dry_run"
        return info

    filename = ds.source.path.rsplit("/", 1)[-1] or "download"
    out_file = fs.join(output_dir, filename)

    if fs.is_s3:
        # Download to temp file, then upload to S3
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp:
            tmp_path = tmp.name
        try:
            urllib.request.urlretrieve(ds.source.path, tmp_path)
            fs.upload_file(tmp_path, out_file)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        urllib.request.urlretrieve(ds.source.path, out_file)

    info["output_file"] = out_file
    info["status"] = "complete"
    return info


def _fetch_local(
    ds: DataSource, output_dir: str, dry_run: bool, fs: PipelineFS,
) -> dict[str, Any]:
    """Copy/symlink from a local path."""
    src = Path(ds.source.path)
    info = {"method": "local", "source_path": str(src)}

    if dry_run:
        info["status"] = "dry_run"
        return info

    if not src.exists():
        raise FileNotFoundError(f"Local source not found: {src}")

    if fs.is_s3:
        # Upload local files to S3
        if src.is_file():
            dest = fs.join(output_dir, src.name)
            fs.upload_file(str(src), dest)
            info["output_file"] = dest
        elif src.is_dir():
            for item in src.rglob("*"):
                if item.is_file():
                    rel = str(item.relative_to(src))
                    dest = fs.join(output_dir, rel)
                    fs.upload_file(str(item), dest)
    else:
        if src.is_file():
            dest = Path(output_dir) / src.name
            shutil.copy2(str(src), str(dest))
            info["output_file"] = str(dest)
        elif src.is_dir():
            for item in src.iterdir():
                dest = Path(output_dir) / item.name
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
    fs = PipelineFS(ds.storage)
    # Validate storage backend BEFORE any expensive work (e.g. HuggingFace downloads)
    if not dry_run:
        fs.validate()
    base_dir = str(base_dir)
    output_dir = fs.join(base_dir, ds.fetch.output_dir)
    fs.makedirs(output_dir)

    handler = _FETCH_HANDLERS.get(ds.source.type)
    if handler is None:
        raise ValueError(f"Unknown source type: {ds.source.type}")

    result = handler(ds, output_dir, dry_run, fs)
    result["source_name"] = ds.name
    result["output_dir"] = output_dir
    if fs.is_s3:
        result["storage"] = "s3"
        result["bucket"] = fs._bucket
    return result

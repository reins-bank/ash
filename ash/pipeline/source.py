"""Load, validate, and represent data source definitions from YAML files."""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ash.config import StorageConfig

# Default directory for source YAML definitions, relative to project root.
DEFAULT_SOURCES_DIR = Path("data_sources")


@dataclass
class LicenseInfo:
    name: str = "unknown"
    url: str = ""
    allowed_use: str = "unknown"  # research | commercial | unknown
    attribution: str = ""


@dataclass
class ProvenanceInfo:
    dedup_status: str = "unknown"
    pii_scrub: str = "none"
    quality_filter: str = "none"


@dataclass
class SourceLocation:
    type: str = "huggingface"  # huggingface | url | local
    path: str = ""
    subset: str | None = None
    split: str = "train"
    revision: str | None = None


@dataclass
class FetchConfig:
    max_documents: int | None = None
    max_size_gb: float | None = None
    streaming: bool = True
    output_dir: str = "data/raw"


@dataclass
class CleaningStep:
    name: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CleaningConfig:
    output_dir: str = "data/clean"
    steps: list[CleaningStep] = field(default_factory=list)


@dataclass
class DataSource:
    """Complete definition of a data source, loaded from a YAML file."""
    name: str = ""
    version: str = "1.0"
    description: str = ""
    modality: str = "text"  # text | image_text | video_text | audio_text | multimodal_instruction
    source: SourceLocation = field(default_factory=SourceLocation)
    license: LicenseInfo = field(default_factory=LicenseInfo)
    provenance: ProvenanceInfo = field(default_factory=ProvenanceInfo)
    fetch: FetchConfig = field(default_factory=FetchConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    yaml_path: str = ""  # path to the YAML file this was loaded from


# ---------------------------------------------------------------------------
# Required top-level fields and valid values for validation
# ---------------------------------------------------------------------------
_REQUIRED_FIELDS = {"name", "source", "license", "fetch", "cleaning"}
_VALID_SOURCE_TYPES = {"huggingface", "url", "local"}
_VALID_ALLOWED_USE = {"research", "commercial", "unknown"}
_VALID_MODALITIES = {"text", "image_text", "video_text", "audio_text", "multimodal_instruction"}


def _build_dataclass(dc_class: type, raw: dict) -> Any:
    """Instantiate a dataclass from a raw dict, ignoring unknown keys."""
    if raw is None:
        return dc_class()
    known = {f.name for f in dc_class.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in known}
    return dc_class(**filtered)


def load_source(yaml_path: str | Path) -> DataSource:
    """Load a single DataSource from a YAML file."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Source YAML not found: {yaml_path}")

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid source YAML (expected mapping): {yaml_path}")

    # Build nested dataclasses
    source_loc = _build_dataclass(SourceLocation, raw.get("source", {}))
    license_info = _build_dataclass(LicenseInfo, raw.get("license", {}))
    provenance = _build_dataclass(ProvenanceInfo, raw.get("provenance", {}))
    fetch_cfg = _build_dataclass(FetchConfig, raw.get("fetch", {}))

    # Cleaning steps need special handling
    clean_raw = raw.get("cleaning", {}) or {}
    steps_raw = clean_raw.get("steps", []) or []
    steps = [CleaningStep(name=s["name"], params=s.get("params", {})) for s in steps_raw]
    cleaning_cfg = CleaningConfig(
        output_dir=clean_raw.get("output_dir", "data/clean"),
        steps=steps,
    )

    # Storage config (local vs s3)
    storage_cfg = _build_dataclass(StorageConfig, raw.get("storage", {}))

    return DataSource(
        name=raw.get("name", yaml_path.stem),
        version=str(raw.get("version", "1.0")),
        description=raw.get("description", ""),
        modality=raw.get("modality", "text"),
        source=source_loc,
        license=license_info,
        provenance=provenance,
        fetch=fetch_cfg,
        cleaning=cleaning_cfg,
        storage=storage_cfg,
        yaml_path=str(yaml_path),
    )


def load_all_sources(sources_dir: str | Path | None = None) -> list[DataSource]:
    """Load all YAML source definitions from a directory."""
    sources_dir = Path(sources_dir) if sources_dir else DEFAULT_SOURCES_DIR
    if not sources_dir.is_dir():
        raise FileNotFoundError(f"Sources directory not found: {sources_dir}")

    sources = []
    for yaml_file in sorted(sources_dir.glob("*.yaml")):
        sources.append(load_source(yaml_file))
    return sources


def validate_source(ds: DataSource) -> list[str]:
    """Validate a DataSource and return a list of error messages (empty = valid)."""
    errors = []

    if not ds.name:
        errors.append("Missing required field: name")
    if not ds.source.path:
        errors.append("Missing required field: source.path")
    if ds.source.type not in _VALID_SOURCE_TYPES:
        errors.append(
            f"Invalid source.type '{ds.source.type}', must be one of {_VALID_SOURCE_TYPES}"
        )
    if ds.license.allowed_use not in _VALID_ALLOWED_USE:
        errors.append(
            f"Invalid license.allowed_use '{ds.license.allowed_use}', "
            f"must be one of {_VALID_ALLOWED_USE}"
        )
    if ds.modality not in _VALID_MODALITIES:
        errors.append(
            f"Invalid modality '{ds.modality}', must be one of {_VALID_MODALITIES}"
        )
    if not ds.fetch.output_dir:
        errors.append("Missing required field: fetch.output_dir")
    if not ds.cleaning.output_dir:
        errors.append("Missing required field: cleaning.output_dir")

    for i, step in enumerate(ds.cleaning.steps):
        if not step.name:
            errors.append(f"Cleaning step {i} has no name")

    return errors

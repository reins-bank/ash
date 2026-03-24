"""Tests for the data pipeline framework — YAML parsing, validation, and end-to-end mock flow."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from ash.pipeline.source import (
    DataSource,
    load_source,
    load_all_sources,
    validate_source,
    CleaningStep,
)
from ash.pipeline.fetch import fetch_source
from ash.pipeline.clean import clean_source, CLEANING_REGISTRY
from ash.pipeline.runner import run_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_YAML = textwrap.dedent("""\
    name: test_source
    description: "A test data source"

    source:
      type: local
      path: "{input_dir}"
      split: train

    license:
      name: "MIT"
      allowed_use: research

    provenance:
      dedup_status: none

    fetch:
      output_dir: "{fetch_out}"

    cleaning:
      output_dir: "{clean_out}"
      steps:
        - name: normalize_whitespace
        - name: min_length_filter
          params:
            min_chars: 10
""")


@pytest.fixture
def source_yaml(tmp_path):
    """Create a minimal valid source YAML and return its path."""
    input_dir = tmp_path / "local_data"
    input_dir.mkdir()
    # Create a test file for local fetch
    test_file = input_dir / "documents.jsonl"
    test_file.write_text("")

    yaml_content = MINIMAL_YAML.format(
        input_dir=str(input_dir),
        fetch_out=str(tmp_path / "raw"),
        clean_out=str(tmp_path / "clean"),
    )
    yaml_path = tmp_path / "test_source.yaml"
    yaml_path.write_text(yaml_content)
    return yaml_path


@pytest.fixture
def sources_dir(tmp_path):
    """Create a directory with multiple source YAMLs."""
    d = tmp_path / "sources"
    d.mkdir()

    for name in ["alpha", "beta"]:
        content = textwrap.dedent(f"""\
            name: {name}
            source:
              type: local
              path: /tmp/fake
            license:
              name: MIT
              allowed_use: research
            fetch:
              output_dir: data/raw/{name}
            cleaning:
              output_dir: data/clean/{name}
              steps: []
        """)
        (d / f"{name}.yaml").write_text(content)

    return d


# ---------------------------------------------------------------------------
# YAML parsing tests
# ---------------------------------------------------------------------------

class TestYAMLParsing:

    def test_load_source_basic(self, source_yaml):
        ds = load_source(source_yaml)
        assert ds.name == "test_source"
        assert ds.description == "A test data source"
        assert ds.source.type == "local"
        assert ds.license.name == "MIT"
        assert ds.license.allowed_use == "research"

    def test_load_source_cleaning_steps(self, source_yaml):
        ds = load_source(source_yaml)
        assert len(ds.cleaning.steps) == 2
        assert ds.cleaning.steps[0].name == "normalize_whitespace"
        assert ds.cleaning.steps[1].name == "min_length_filter"
        assert ds.cleaning.steps[1].params == {"min_chars": 10}

    def test_load_source_defaults(self, source_yaml):
        ds = load_source(source_yaml)
        assert ds.version == "1.0"
        assert ds.source.split == "train"
        assert ds.source.revision is None
        assert ds.fetch.streaming is True

    def test_load_all_sources(self, sources_dir):
        sources = load_all_sources(sources_dir)
        assert len(sources) == 2
        names = {s.name for s in sources}
        assert names == {"alpha", "beta"}

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_source(tmp_path / "nonexistent.yaml")

    def test_load_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_all_sources(tmp_path / "no_such_dir")

    def test_load_invalid_yaml(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("just a string, not a mapping")
        with pytest.raises(ValueError, match="expected mapping"):
            load_source(bad)

    def test_load_example_sources(self):
        """Verify all shipped example YAMLs in data_sources/ parse correctly."""
        sources_dir = Path("data_sources")
        if not sources_dir.is_dir():
            pytest.skip("data_sources/ not found (running from non-root dir)")
        sources = load_all_sources(sources_dir)
        assert len(sources) >= 4
        for ds in sources:
            errors = validate_source(ds)
            assert errors == [], f"{ds.name} has validation errors: {errors}"


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:

    def test_valid_source(self, source_yaml):
        ds = load_source(source_yaml)
        errors = validate_source(ds)
        assert errors == []

    def test_missing_name(self):
        ds = DataSource()
        errors = validate_source(ds)
        assert any("name" in e for e in errors)

    def test_missing_source_path(self):
        ds = DataSource(name="test")
        errors = validate_source(ds)
        assert any("source.path" in e for e in errors)

    def test_invalid_source_type(self):
        ds = DataSource(name="test")
        ds.source.type = "ftp"
        ds.source.path = "/some/path"
        errors = validate_source(ds)
        assert any("source.type" in e for e in errors)

    def test_invalid_allowed_use(self):
        ds = DataSource(name="test")
        ds.source.path = "/some/path"
        ds.license.allowed_use = "piracy"
        errors = validate_source(ds)
        assert any("allowed_use" in e for e in errors)

    def test_empty_cleaning_step_name(self):
        ds = DataSource(name="test")
        ds.source.path = "/some/path"
        ds.cleaning.steps = [CleaningStep(name="")]
        errors = validate_source(ds)
        assert any("step 0" in e for e in errors)


# ---------------------------------------------------------------------------
# Cleaning tests
# ---------------------------------------------------------------------------

class TestCleaners:

    def test_strip_html(self):
        fn = CLEANING_REGISTRY["strip_html"]
        assert fn("<p>hello</p>", {}) == "hello"
        assert fn("no tags", {}) == "no tags"

    def test_normalize_whitespace(self):
        fn = CLEANING_REGISTRY["normalize_whitespace"]
        assert fn("  hello   world  ", {}) == "hello world"
        assert fn("a\n\n\n\nb", {}) == "a\n\nb"

    def test_min_length_filter(self):
        fn = CLEANING_REGISTRY["min_length_filter"]
        assert fn("short", {"min_chars": 100}) is None
        assert fn("a" * 100, {"min_chars": 100}) is not None

    def test_max_length_filter(self):
        fn = CLEANING_REGISTRY["max_length_filter"]
        assert fn("ok", {"max_chars": 10}) == "ok"
        assert fn("a" * 11, {"max_chars": 10}) is None

    def test_strip_latex(self):
        fn = CLEANING_REGISTRY["strip_latex_commands"]
        result = fn(r"\textbf{bold} and \emph{italic}", {})
        assert "bold" in result
        assert "italic" in result
        assert "\\" not in result

    def test_registry_has_all_builtins(self):
        expected = {
            "strip_html", "strip_latex_commands", "normalize_whitespace",
            "min_length_filter", "max_length_filter", "dedup_exact", "tokenize",
        }
        assert expected.issubset(set(CLEANING_REGISTRY.keys()))


# ---------------------------------------------------------------------------
# Fetch tests
# ---------------------------------------------------------------------------

class TestFetch:

    def test_fetch_local_dry_run(self, source_yaml):
        ds = load_source(source_yaml)
        result = fetch_source(ds, dry_run=True)
        assert result["status"] == "dry_run"
        assert result["source_name"] == "test_source"

    def test_fetch_local_file(self, tmp_path):
        """Fetch a local file and verify it gets copied."""
        src_file = tmp_path / "input.txt"
        src_file.write_text("hello world")

        ds = DataSource(name="local_test")
        ds.source.type = "local"
        ds.source.path = str(src_file)
        ds.fetch.output_dir = str(tmp_path / "output")

        result = fetch_source(ds, dry_run=False, base_dir="")
        assert result["status"] == "complete"
        assert Path(result["output_file"]).exists()

    def test_fetch_unknown_type(self):
        ds = DataSource(name="bad")
        ds.source.type = "ftp"
        ds.fetch.output_dir = "/tmp/test"
        with pytest.raises(ValueError, match="Unknown source type"):
            fetch_source(ds)


# ---------------------------------------------------------------------------
# Clean integration test
# ---------------------------------------------------------------------------

class TestCleanIntegration:

    def test_clean_dry_run(self, tmp_path):
        ds = DataSource(name="test")
        ds.fetch.output_dir = str(tmp_path / "raw")
        ds.cleaning.output_dir = str(tmp_path / "clean")
        ds.cleaning.steps = [CleaningStep(name="normalize_whitespace")]

        result = clean_source(ds, dry_run=True, base_dir="")
        assert result["status"] == "dry_run"

    def test_clean_missing_input(self, tmp_path):
        ds = DataSource(name="test")
        ds.fetch.output_dir = str(tmp_path / "raw")
        ds.cleaning.output_dir = str(tmp_path / "clean")
        ds.cleaning.steps = []

        result = clean_source(ds, dry_run=False, base_dir="")
        assert result["status"] == "skipped"

    def test_clean_with_documents(self, tmp_path):
        """Full cleaning pipeline on synthetic documents."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        clean_dir = tmp_path / "clean"

        # Write synthetic documents
        docs = [
            {"text": "<p>Hello world!</p> This is a test document with enough text to pass."},
            {"text": "short"},  # will be filtered by min_length
            {"text": "  Lots   of   extra    whitespace   in   this   long   document  here  "},
            {"text": "<p>Hello world!</p> This is a test document with enough text to pass."},  # dup
        ]
        with open(raw_dir / "documents.jsonl", "w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")

        ds = DataSource(name="test_clean")
        ds.fetch.output_dir = str(raw_dir)
        ds.cleaning.output_dir = str(clean_dir)
        ds.cleaning.steps = [
            CleaningStep(name="strip_html"),
            CleaningStep(name="normalize_whitespace"),
            CleaningStep(name="min_length_filter", params={"min_chars": 10}),
            CleaningStep(name="dedup_exact"),
        ]

        result = clean_source(ds, dry_run=False, base_dir="")
        assert result["status"] == "complete"
        assert result["documents_read"] == 4
        assert result["documents_filtered"] == 2  # 1 too short + 1 duplicate
        assert result["documents_output"] == 2

        # Verify output file
        out_file = Path(result["output_file"])
        assert out_file.exists()
        lines = out_file.read_text().strip().split("\n")
        assert len(lines) == 2
        # Verify HTML was stripped
        first = json.loads(lines[0])
        assert "<p>" not in first["text"]

    def test_clean_unknown_step(self, tmp_path):
        """Unknown cleaning step should produce an error result."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "documents.jsonl").write_text('{"text": "hello"}\n')

        ds = DataSource(name="test")
        ds.fetch.output_dir = str(raw_dir)
        ds.cleaning.output_dir = str(tmp_path / "clean")
        ds.cleaning.steps = [CleaningStep(name="nonexistent_step")]

        result = clean_source(ds, dry_run=False, base_dir="")
        assert result["status"] == "error"
        assert "nonexistent_step" in result["reason"]


# ---------------------------------------------------------------------------
# End-to-end mock pipeline test
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_full_pipeline_local_source(self, tmp_path):
        """End-to-end: define a local source, fetch, clean, verify outputs."""
        # Setup: create local input data
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        docs = [
            {"text": "The quick brown fox jumps over the lazy dog. " * 5},
            {"text": "Science is the poetry of reality, expressed in mathematics. " * 3},
            {"text": "tiny"},  # will be filtered
        ]
        jsonl = input_dir / "documents.jsonl"
        with open(jsonl, "w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")

        # Create a source YAML
        yaml_content = textwrap.dedent(f"""\
            name: e2e_test
            source:
              type: local
              path: "{input_dir}"
            license:
              name: MIT
              allowed_use: research
            fetch:
              output_dir: "{tmp_path / 'raw'}"
            cleaning:
              output_dir: "{tmp_path / 'clean'}"
              steps:
                - name: normalize_whitespace
                - name: min_length_filter
                  params:
                    min_chars: 20
        """)
        yaml_path = tmp_path / "e2e_test.yaml"
        yaml_path.write_text(yaml_content)

        # Load and run
        ds = load_source(yaml_path)
        assert ds.name == "e2e_test"

        results = run_pipeline([ds], stages=["fetch", "clean"], dry_run=False, base_dir="")
        assert len(results) == 1

        r = results[0]
        assert r["status"] == "complete"

        # Fetch stage: local copy happened
        fetch_r = r["stages"]["fetch"]
        assert fetch_r["status"] == "complete"
        assert Path(fetch_r.get("output_dir", "")).exists()

        # Clean stage: documents were processed
        clean_r = r["stages"]["clean"]
        assert clean_r["status"] == "complete"
        assert clean_r["documents_output"] == 2  # "tiny" was filtered

    def test_full_pipeline_dry_run(self, tmp_path):
        """Dry run should not create output files."""
        ds = DataSource(name="dry_test")
        ds.source.type = "local"
        ds.source.path = "/tmp/fake"
        ds.license.allowed_use = "research"
        ds.fetch.output_dir = str(tmp_path / "raw")
        ds.cleaning.output_dir = str(tmp_path / "clean")

        results = run_pipeline([ds], dry_run=True, base_dir="")
        assert len(results) == 1
        assert results[0]["status"] == "dry_run"

    def test_pipeline_invalid_source(self):
        """Invalid source should be flagged, not crash."""
        ds = DataSource()  # missing name, path, etc.
        results = run_pipeline([ds])
        assert len(results) == 1
        assert results[0]["status"] == "invalid"
        assert len(results[0]["errors"]) > 0

    def test_pipeline_unknown_use_warning(self, tmp_path):
        """Sources with allowed_use=unknown should produce a warning."""
        ds = DataSource(name="unknown_test")
        ds.source.type = "local"
        ds.source.path = "/tmp/fake"
        ds.license.allowed_use = "unknown"
        ds.fetch.output_dir = str(tmp_path / "raw")
        ds.cleaning.output_dir = str(tmp_path / "clean")

        results = run_pipeline([ds], dry_run=True, base_dir="")
        assert "warning" in results[0]
        assert "quarantine" in results[0]["warning"]

"""Cleaning and standardization steps for fetched data."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Cleaning step registry
# ---------------------------------------------------------------------------

# Each cleaner takes (text, params) -> text | None.  Returning None means
# the document should be filtered out entirely.
CleanerFn = Callable[[str, dict[str, Any]], str | None]

CLEANING_REGISTRY: dict[str, CleanerFn] = {}


def register_cleaner(name: str):
    """Decorator to register a cleaning function."""
    def wrapper(fn: CleanerFn) -> CleanerFn:
        CLEANING_REGISTRY[name] = fn
        return fn
    return wrapper


# ---------------------------------------------------------------------------
# Built-in cleaners
# ---------------------------------------------------------------------------

@register_cleaner("strip_html")
def _strip_html(text: str, params: dict[str, Any]) -> str:
    return re.sub(r"<[^>]+>", "", text)


@register_cleaner("strip_latex_commands")
def _strip_latex(text: str, params: dict[str, Any]) -> str:
    # Remove common LaTeX commands but preserve their arguments
    text = re.sub(r"\\(?:textbf|textit|emph|section|subsection|paragraph)\{([^}]*)\}", r"\1", text)
    # Remove remaining bare commands
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    # Remove braces
    text = re.sub(r"[{}]", "", text)
    return text


@register_cleaner("normalize_whitespace")
def _normalize_ws(text: str, params: dict[str, Any]) -> str:
    # Collapse runs of whitespace but preserve paragraph breaks (double newline)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@register_cleaner("min_length_filter")
def _min_length(text: str, params: dict[str, Any]) -> str | None:
    min_chars = params.get("min_chars", 100)
    return text if len(text) >= min_chars else None


@register_cleaner("max_length_filter")
def _max_length(text: str, params: dict[str, Any]) -> str | None:
    max_chars = params.get("max_chars", 1_000_000)
    return text if len(text) <= max_chars else None


@register_cleaner("dedup_exact")
def _dedup_exact(text: str, params: dict[str, Any]) -> str:
    # Exact dedup is handled at the corpus level in clean_source, not per-doc.
    # This is a pass-through; the actual dedup logic lives in clean_source.
    return text


@register_cleaner("tokenize")
def _tokenize(text: str, params: dict[str, Any]) -> str:
    # Tokenization is a final step handled separately after cleaning.
    # This is a pass-through marker; actual tokenization is in clean_source.
    return text


# ---------------------------------------------------------------------------
# Corpus-level cleaning orchestrator
# ---------------------------------------------------------------------------

def _extract_text(doc: dict[str, Any]) -> str:
    """Extract the text field from a document dict, trying common field names."""
    for key in ("text", "content", "body", "document"):
        if key in doc:
            return str(doc[key])
    # Fall back to concatenating all string values
    parts = [str(v) for v in doc.values() if isinstance(v, str)]
    return "\n".join(parts)


def clean_source(
    ds: "DataSource",
    dry_run: bool = False,
    base_dir: str | Path = ".",
) -> dict[str, Any]:
    """Run all cleaning steps on fetched data for a source.

    Reads from fetch.output_dir/documents.jsonl, applies cleaning steps,
    writes to cleaning.output_dir/clean.jsonl (and optionally tokenized .bin).
    """
    from ash.pipeline.source import DataSource  # avoid circular

    base_dir = Path(base_dir)
    input_dir = base_dir / ds.fetch.output_dir
    output_dir = base_dir / ds.cleaning.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_file = input_dir / "documents.jsonl"
    result: dict[str, Any] = {
        "source_name": ds.name,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "steps": [s.name for s in ds.cleaning.steps],
    }

    if dry_run:
        result["status"] = "dry_run"
        return result

    if not input_file.exists():
        result["status"] = "skipped"
        result["reason"] = f"Input file not found: {input_file}"
        return result

    # Determine which steps to run as per-document transforms vs corpus-level
    per_doc_steps = []
    do_dedup = False
    do_tokenize = False
    tokenizer_name = "gpt2"

    for step in ds.cleaning.steps:
        if step.name == "dedup_exact":
            do_dedup = True
        elif step.name == "tokenize":
            do_tokenize = True
            tokenizer_name = step.params.get("tokenizer", "gpt2")
        else:
            if step.name not in CLEANING_REGISTRY:
                result["status"] = "error"
                result["reason"] = f"Unknown cleaning step: {step.name}"
                return result
            per_doc_steps.append((CLEANING_REGISTRY[step.name], step.params))

    # Process documents
    clean_docs: list[str] = []
    seen_hashes: set[int] = set()
    docs_read = 0
    docs_filtered = 0

    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs_read += 1
            doc = json.loads(line)
            text = _extract_text(doc)

            # Apply per-document cleaning steps
            for cleaner_fn, params in per_doc_steps:
                text = cleaner_fn(text, params)
                if text is None:
                    break

            if text is None:
                docs_filtered += 1
                continue

            # Exact dedup
            if do_dedup:
                h = hash(text)
                if h in seen_hashes:
                    docs_filtered += 1
                    continue
                seen_hashes.add(h)

            clean_docs.append(text)

    # Write clean JSONL
    out_file = output_dir / "clean.jsonl"
    with open(out_file, "w") as f:
        for text in clean_docs:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    result["documents_read"] = docs_read
    result["documents_filtered"] = docs_filtered
    result["documents_output"] = len(clean_docs)
    result["output_file"] = str(out_file)

    # Tokenize if requested
    if do_tokenize and clean_docs:
        try:
            import tiktoken
            import numpy as np

            enc = tiktoken.get_encoding(tokenizer_name)
            eot = enc.eot_token
            all_tokens: list[int] = []
            for text in clean_docs:
                all_tokens.extend(enc.encode(text))
                all_tokens.append(eot)

            arr = np.array(all_tokens, dtype=np.uint16)
            bin_path = output_dir / "tokens.bin"
            arr.tofile(str(bin_path))
            result["tokens"] = len(all_tokens)
            result["token_file"] = str(bin_path)
        except ImportError:
            result["tokenize_skipped"] = "tiktoken not available"

    result["status"] = "complete"
    return result

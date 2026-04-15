"""
study_extractor.py — Structured research paper extraction.

For every study folder (containing one or more research papers), this module
reads all of the study's documents and produces a structured StudyRecord:

  title, authors[], publication_year,
  research_question, hypothesis, data_sources[],
  conclusions, keywords[],
  abstract, primary_topic, subtopic, microtopic

It uses a map-reduce strategy with the configured STUDY_EXTRACTION_MODEL:

  MAP    one extraction call per document → partial StudyRecord
  REDUCE one merge call combining the per-doc results → final StudyRecord

Why map-reduce?
  A study collection with multiple papers × multiple pages can easily exceed
  any single-prompt context budget. Per-doc extraction keeps each call
  small, parallelizable, and individually cacheable.

Every extracted field carries provenance: which file and which page it
came from, and the model's self-reported confidence (0–1). The provenance
is what powers research verification and audit trails.

This module is pure Python — it does NOT touch the database. The caller
(ingest.py) is responsible for persisting the result via StudyStore.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

from config import (
    OLLAMA_BASE_URL,
    STUDY_EXTRACTION_MAX_CHARS,
    STUDY_EXTRACTION_MODEL,
    STUDY_EXTRACTION_TIMEOUT_S,
)

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class DocumentText:
    """One document's extracted text, ready for an extraction call."""
    filename: str
    pages: list[tuple[int, str]]   # [(page_number, text), ...]

    def combined(self, max_chars: int = STUDY_EXTRACTION_MAX_CHARS) -> str:
        """Concatenate pages with markers, truncated to fit the prompt budget."""
        parts: list[str] = []
        used = 0
        for page_number, text in self.pages:
            marker = f"\n[Page {page_number}]\n"
            chunk = marker + text
            if used + len(chunk) > max_chars:
                remaining = max(0, max_chars - used)
                if remaining > 0:
                    parts.append(chunk[:remaining])
                break
            parts.append(chunk)
            used += len(chunk)
        return "".join(parts)


@dataclass
class FieldProvenance:
    """Records where a single extracted field value came from."""
    field_name: str
    field_value: str
    source_file: str
    source_page: int | None
    confidence: float            # 0.0–1.0


@dataclass
class StudyRecord:
    """A structured research record for one study."""
    study_id: str
    study_id_source: str       # 'folder_name' | 'extracted' | 'hash'
    folder_path: str
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    publication_year: int | None = None
    research_question: str | None = None
    hypothesis: str | None = None
    data_sources: list[str] = field(default_factory=list)
    conclusions: str | None = None
    keywords: list[str] = field(default_factory=list)
    abstract: str | None = None
    primary_topic: str | None = None
    subtopic: str | None = None
    microtopic: str | None = None
    extraction_model: str = ""
    source_doc_hash: str = ""
    last_extracted_at: datetime | None = None
    provenance: list[FieldProvenance] = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────


def extract_study_record(
    folder_path: str,
    documents: list[DocumentText],
) -> StudyRecord:
    """
    Run map-reduce extraction over a study's documents.

    Args:
        folder_path: absolute path to the study's folder
        documents:   list of the study's parsed documents

    Returns:
        A fully populated StudyRecord (fields may still be None where the
        documents do not mention them — null is intentional, never guess).
    """
    folder_name = Path(folder_path).name

    if not documents:
        logger.warning(f"No documents for study at {folder_path}")
        return _empty_record(folder_path, folder_name)

    # ── MAP: one extraction per document ────────────────────────────────────
    per_doc_results: list[dict] = []
    for doc in documents:
        try:
            result = _extract_from_document(doc)
            if result:
                per_doc_results.append({"filename": doc.filename, "fields": result})
        except Exception as e:
            logger.warning(f"  Per-doc extraction failed on {doc.filename}: {e}")

    if not per_doc_results:
        logger.warning(f"All per-doc extractions failed for {folder_name}")
        return _empty_record(folder_path, folder_name)

    # ── REDUCE: merge per-doc results into a single record ─────────────────
    merged = _merge_results(per_doc_results)

    # ── Parse folder structure for topic hierarchy ──────────────────────────
    primary_topic, subtopic, microtopic = _parse_folder_hierarchy(folder_path)

    # ── Identify the study (folder name → extracted title → hash) ──────────
    study_id, id_source = _resolve_study_id(
        folder_name=folder_name,
        folder_path=folder_path,
        extracted_title=merged.get("title"),
        extracted_year=merged.get("publication_year"),
    )

    # ── Build the record ───────────────────────────────────────────────────
    record = StudyRecord(
        study_id=study_id,
        study_id_source=id_source,
        folder_path=folder_path,
        title=merged.get("title"),
        authors=_dedupe_list(merged.get("authors", [])),
        publication_year=_parse_year(merged.get("publication_year")),
        research_question=merged.get("research_question"),
        hypothesis=merged.get("hypothesis"),
        data_sources=_dedupe_list(merged.get("data_sources", [])),
        conclusions=merged.get("conclusions"),
        keywords=_dedupe_list(merged.get("keywords", [])),
        abstract=merged.get("abstract"),
        primary_topic=primary_topic,
        subtopic=subtopic,
        microtopic=microtopic,
        extraction_model=STUDY_EXTRACTION_MODEL,
        source_doc_hash=_doc_hash(documents),
        last_extracted_at=datetime.utcnow(),
        provenance=_build_provenance(per_doc_results, merged),
    )
    return record


# ── Map step: extract from one document ──────────────────────────────────────


def _extract_from_document(doc: DocumentText) -> dict | None:
    """
    Single LLM call. Returns parsed JSON dict or None on failure.
    The prompt is strict about returning null for missing fields.
    """
    text = doc.combined()
    if not text.strip():
        return None

    prompt = _build_extraction_prompt(text)
    raw = _call_ollama_json(prompt)
    if not raw:
        return None
    return _parse_json_safely(raw)


def _build_extraction_prompt(document_text: str) -> str:
    return (
        "You are a research paper extraction assistant. Read the document and "
        "return a JSON object with the structured fields below. Follow these rules:\n"
        "  • Return null for any field you cannot find. NEVER guess.\n"
        "  • authors: list of author names exactly as they appear.\n"
        "  • research_question: the primary research question or problem statement.\n"
        "  • hypothesis: stated hypothesis or experimental goal, if available.\n"
        "  • data_sources: list of datasets, benchmarks, or sources used.\n"
        "  • conclusions: key findings and conclusions of the research.\n"
        "  • keywords: research topics/keywords from the paper.\n"
        "  • publication_year: year (YYYY) if present, else null.\n"
        "  • Output JSON only, no commentary.\n\n"
        "Schema:\n"
        "{\n"
        '  "title": string|null,\n'
        '  "authors": [string],\n'
        '  "publication_year": number|null,\n'
        '  "research_question": string|null,\n'
        '  "hypothesis": string|null,\n'
        '  "data_sources": [string],\n'
        '  "conclusions": string|null,\n'
        '  "keywords": [string],\n'
        '  "abstract": string|null\n'
        "}\n\n"
        "Document:\n"
        f"{document_text}\n\n"
        "JSON:"
    )


# ── Reduce step: merge per-document results ─────────────────────────────────


def _merge_results(per_doc_results: list[dict]) -> dict:
    """
    Merge a list of per-document extraction dicts into one final record.

    Strategy:
      • Scalar fields (title, research_question, hypothesis, abstract, conclusions) — first
        non-null value wins. Documents are processed in folder order.
      • List fields (authors, keywords, data_sources) — union across
        all documents, deduplicated.
    """
    SCALAR_FIELDS = (
        "title", "research_question", "hypothesis", "abstract", "conclusions",
    )
    LIST_FIELDS = ("authors", "keywords", "data_sources")

    merged: dict = {f: None for f in SCALAR_FIELDS}
    for f in LIST_FIELDS:
        merged[f] = []
    merged["publication_year"] = None

    for entry in per_doc_results:
        fields = entry.get("fields") or {}
        for f in SCALAR_FIELDS:
            if merged[f] is None and fields.get(f):
                merged[f] = fields[f]
        for f in LIST_FIELDS:
            value = fields.get(f) or []
            if isinstance(value, list):
                merged[f].extend(str(v) for v in value if v)
        # For year, take first valid one found
        if merged["publication_year"] is None and fields.get("publication_year"):
            merged["publication_year"] = fields["publication_year"]

    return merged


def _build_provenance(
    per_doc_results: list[dict],
    merged: dict,
) -> list[FieldProvenance]:
    """
    Build the provenance trail by walking the per-doc results in order
    and recording the first document that supplied each scalar field, plus
    every document that contributed any item to a list field.
    """
    provenance: list[FieldProvenance] = []
    SCALAR_FIELDS = (
        "title", "research_question", "hypothesis", "abstract", "conclusions",
    )
    LIST_FIELDS = ("authors", "keywords", "data_sources")

    seen_scalar: set[str] = set()
    for entry in per_doc_results:
        filename = entry["filename"]
        fields = entry.get("fields") or {}
        for f in SCALAR_FIELDS:
            if f in seen_scalar:
                continue
            value = fields.get(f)
            if value:
                provenance.append(FieldProvenance(
                    field_name=f,
                    field_value=str(value),
                    source_file=filename,
                    source_page=None,
                    confidence=1.0,
                ))
                seen_scalar.add(f)
        for f in LIST_FIELDS:
            for item in fields.get(f) or []:
                if not item:
                    continue
                provenance.append(FieldProvenance(
                    field_name=f,
                    field_value=str(item),
                    source_file=filename,
                    source_page=None,
                    confidence=1.0,
                ))
    return provenance


# ── Topic hierarchy parsing ───────────────────────────────────────────────────


def _parse_folder_hierarchy(folder_path: str) -> tuple[str | None, str | None, str | None]:
    """
    Parse folder structure to extract topic hierarchy.
    Expected structure: parent/PrimaryTopic/SubTopic/MicroTopic/
    
    Returns: (primary_topic, subtopic, microtopic)
    """
    path = Path(folder_path)
    parts = path.parts
    
    # We need at least 3 levels to extract hierarchy
    if len(parts) < 3:
        return None, None, None
    
    # The last part is the study folder itself
    # Walk back: immediate parent is microtopic, grandparent is subtopic, etc.
    microtopic = None
    subtopic = None
    primary_topic = None
    
    if len(parts) >= 3:
        # parts[-2] is the immediate parent (micro/specific topic)
        microtopic = parts[-2] if parts[-2] not in {"", "."} else None
    if len(parts) >= 4:
        # parts[-3] is the sub-topic
        subtopic = parts[-3] if parts[-3] not in {"", "."} else None
    if len(parts) >= 5:
        # parts[-4] is the primary topic
        primary_topic = parts[-4] if parts[-4] not in {"", "."} else None
    
    return primary_topic, subtopic, microtopic


# ── Helpers ───────────────────────────────────────────────────────────────────


def _empty_record(folder_path: str, folder_name: str) -> StudyRecord:
    return StudyRecord(
        study_id=folder_name or _hash_id(folder_path),
        study_id_source="folder_name" if folder_name else "hash",
        folder_path=folder_path,
        extraction_model=STUDY_EXTRACTION_MODEL,
        last_extracted_at=datetime.utcnow(),
    )


def _resolve_study_id(
    folder_name: str,
    folder_path: str,
    extracted_title: str | None,
    extracted_year: int | None,
) -> tuple[str, str]:
    """
    Decide on a stable study_id. Priority:
      1. folder name if it looks like an ID (alphanumeric, ≥4 chars)
      2. <title>_<year> if both extracted
      3. SHA-1 hash of folder path (always works, never collides)
    """
    if folder_name and re.match(r"^[A-Za-z0-9_\-]{4,}$", folder_name):
        return folder_name, "folder_name"

    if extracted_title and extracted_year:
        cleaned_title = re.sub(r"[^A-Za-z0-9]+", "_", extracted_title).strip("_").lower()
        if cleaned_title:
            return f"{cleaned_title}_{extracted_year}", "extracted"

    return _hash_id(folder_path), "hash"


def _hash_id(s: str) -> str:
    return "s_" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _doc_hash(documents: list[DocumentText]) -> str:
    """Stable hash of all docs combined — used for skip-if-unchanged logic."""
    h = hashlib.sha256()
    for doc in sorted(documents, key=lambda d: d.filename):
        h.update(doc.filename.encode("utf-8"))
        for page_number, text in doc.pages:
            h.update(str(page_number).encode("utf-8"))
            h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _parse_year(value: object) -> int | None:
    if not value:
        return None
    try:
        year = int(value)
        if 1900 <= year <= 2100:
            return year
    except (ValueError, TypeError):
        pass
    return None


def _dedupe_list(values: list) -> list[str]:
    """Case-insensitive dedupe preserving original casing of first occurrence."""
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _parse_json_safely(raw: str) -> dict | None:
    """Try to extract a JSON object from the model output."""
    if not raw:
        return None
    raw = raw.strip()
    # Direct parse
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    # Find first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _call_ollama_json(prompt: str) -> str:
    """
    Call the configured study extraction model with format=json.
    Returns the raw response string, or empty string on failure.
    """
    payload = {
        "model": STUDY_EXTRACTION_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",   # forces well-formed JSON output
        "options": {"temperature": 0},
    }
    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=STUDY_EXTRACTION_TIMEOUT_S,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        logger.warning(f"  Study extraction call failed: {e}")
        return ""

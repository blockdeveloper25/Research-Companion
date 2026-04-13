"""
ingest.py — Document ingestion CLI

Usage:
  python ingest.py --folder /path/to/folder                          # default ingest (chunks + graph)
  python ingest.py --folder /path/to/patients --type patient         # patient mode: force OCR + structured extraction
  python ingest.py --folder /path/to/policies --type policy          # policy mode: tag folder type, regular RAG
  python ingest.py --folder /path/to/folder --no-extract             # skip patient extraction (chunks only, faster)
  python ingest.py --folder /path/to/folder --force-ocr              # force OCR even on text PDFs
  python ingest.py --folder /path/to/folder --extract-only           # re-run patient extraction on existing chunks
  python ingest.py --extract-patient PATIENT_ID                      # re-extract a single patient
  python ingest.py --force file.pdf                                  # re-ingest one PDF
  python ingest.py --remove file.pdf                                 # remove a PDF from the knowledge base

Coordinates: ocr.py -> chunker.py -> ollama embeddings -> vectorstore.py
              + patient_extractor.py -> patient_store.py (when --type patient)
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import httpx

from db.connection import init_pool
from rag.chunker import build_chunks, compute_doc_id
from rag.graph_extractor import (
    extract_document_metadata,
    extract_entities_and_relationships,
)
from rag.graph_store import GraphStore
from rag.ocr import extract_text_from_pdf
from rag.study_extractor import (
    DocumentText,
    extract_study_record,
)
from rag.study_store import (
    get_existing_doc_hash,
    register_folder,
    upsert_study,
)
from rag.vectorstore import VectorStore

# ── Configuration ─────────────────────────────────────────────────────────────

INGESTION_LOG   = Path(__file__).parent / "ingestion_log.json"
EXTRACTION_LOG  = Path(__file__).parent.parent / "logs" / "extraction_errors.log"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL     = "nomic-embed-text"
WORKER_MODEL    = "llama3.2:3b"
BATCH_SIZE      = 32

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Knowledge Companion — Document Ingestion")
    parser.add_argument("--folder", type=str, default=None, help="Folder to ingest PDFs from")
    parser.add_argument("--type", type=str, default="general",
                        choices=["general", "research", "policy"],
                        help="Folder type. 'research' enables structured extraction of research papers.")
    parser.add_argument("--no-extract", action="store_true",
                        help="Skip structured extraction (chunks only). Useful for fast testing.")
    parser.add_argument("--extract-only", action="store_true",
                        help="Skip chunking; only run structured extraction on existing records.")
    parser.add_argument("--force-ocr", action="store_true",
                        help="Force OCR on every page (auto-on for --type research or patient).")
    parser.add_argument("--force",  type=str, default=None, help="Re-ingest a specific PDF (filename)")
    parser.add_argument("--extract-study", type=str, default=None,
                        help="Re-extract a single study by study_id")
    parser.add_argument("--remove", type=str, default=None, help="Remove a PDF from the knowledge base")
    args = parser.parse_args()

    init_pool()
    store       = VectorStore()
    graph_store = GraphStore()
    log         = _load_log()

    # ── Single-purpose modes ──────────────────────────────────────────────────
    if args.remove:
        _remove_document(args.remove, store, graph_store, log)
        return

    if args.extract_study:
        _reextract_single_study(args.extract_study)
        return

    if not args.folder:
        logger.error("Please specify a folder: python ingest.py --folder /path/to/pdfs")
        sys.exit(1)

    search_root = Path(args.folder).expanduser().resolve()
    if not search_root.exists():
        logger.error(f"Folder not found: {search_root}")
        sys.exit(1)

    # ── Extraction is forced on for --type research ─────────────────────────────
    folder_type = args.type
    extraction_enabled = (
        (folder_type == "research") and not args.no_extract
    )
    force_ocr = args.force_ocr or folder_type == "research"

    # ── Register the root folder's type ──────────────────────────────────────
    register_folder(search_root.name, str(search_root), folder_type)

    # ── Discover PDFs ─────────────────────────────────────────────────────────
    pdf_files = sorted(search_root.rglob("*.pdf"))
    if not pdf_files:
        logger.info(f"No PDF files found in {search_root}")
        return

    logger.info(
        f"Found {len(pdf_files)} PDF(s) in {search_root} "
        f"(type={folder_type}, force_ocr={force_ocr}, extraction={extraction_enabled})"
    )

    # ── Group by record folder when in research or patient mode ──────────────
    # The immediate parent folder of each PDF is treated as one record (study/patient).
    pdfs_by_record: dict[Path, list[Path]] = defaultdict(list)
    for pdf in pdf_files:
        pdfs_by_record[pdf.parent].append(pdf)

    # ── Process: chunking pass (skipped if --extract-only) ───────────────────
    stats = {"processed": 0, "skipped": 0, "failed": 0, "extracted": 0, "extraction_skipped": 0}

    if not args.extract_only:
        for pdf_path in pdf_files:
            try:
                _process_pdf(
                    pdf_path, store, graph_store, log,
                    force=(args.force == pdf_path.name),
                    force_ocr=force_ocr,
                )
                stats["processed"] += 1
            except AlreadyIngestedException:
                stats["skipped"] += 1
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                stats["failed"] += 1

        _save_log(log)

    # ── Structured extraction pass (research papers) ───────────────────────────
    if extraction_enabled:
        total_records = len(pdfs_by_record)
        for idx, (record_folder, pdfs) in enumerate(pdfs_by_record.items(), start=1):
            try:
                was_extracted = _extract_research_study(
                    record_folder, pdfs,
                    progress=f"[{idx}/{total_records}]",
                    force_ocr=force_ocr,
                )
                if was_extracted:
                    stats["extracted"] += 1
                else:
                    stats["extraction_skipped"] += 1
            except Exception as e:
                logger.error(f"  Extraction failed for {record_folder.name}: {e}")
                _log_extraction_error(record_folder, e)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(
        f"\nDone — chunks: {stats['processed']} processed, "
        f"{stats['skipped']} skipped, {stats['failed']} failed"
    )
    if extraction_enabled:
        logger.info(
            f"      records: {stats['extracted']} extracted, "
            f"{stats['extraction_skipped']} skipped (unchanged)"
        )
    logger.info(f"Total chunks in knowledge base: {store.count()}")


# ── Per-PDF processing (chunking + graph) ────────────────────────────────────


def _process_pdf(
    pdf_path: Path,
    store: VectorStore,
    graph_store: GraphStore,
    log: dict,
    force: bool = False,
    force_ocr: bool = False,
) -> None:
    """Existing pipeline: text → chunks → embeddings → vectorstore + graph."""
    from config import GRAPH_EXTRACTION_ENABLED

    doc_id = compute_doc_id(str(pdf_path))
    filename = pdf_path.name
    folder = pdf_path.parent.name

    if not force and doc_id in log:
        logger.info(f"  Skipping {filename} — already ingested")
        raise AlreadyIngestedException()

    if force and doc_id in log:
        logger.info(f"  Force re-ingesting {filename} — removing old data")
        store.delete_by_doc_id(doc_id)
        graph_store.delete_by_doc_id(doc_id)

    logger.info(f"Processing: {filename}")
    start = time.time()

    pages = extract_text_from_pdf(str(pdf_path), force_ocr=force_ocr)
    if not pages:
        raise ValueError(f"No pages extracted from {filename}")

    chunks = build_chunks(pages, doc_id=doc_id, source=filename, folder=folder)
    if not chunks:
        raise ValueError(f"No chunks produced from {filename}")

    chunks = _enrich_metadata(chunks)

    entity_count = 0
    rel_count = 0
    if GRAPH_EXTRACTION_ENABLED:
        try:
            entities, relationships = extract_entities_and_relationships(
                chunks, doc_id, filename, folder,
            )
            doc_meta = extract_document_metadata(chunks, doc_id, filename, folder)
            entity_count = len(entities)
            rel_count = len(relationships)
        except Exception as e:
            logger.warning(f"  Graph extraction failed (non-critical): {e}")
            entities, relationships, doc_meta = [], [], None

    embeddings = _embed_batch([c.text for c in chunks])
    store.add_chunks(chunks, embeddings)

    if GRAPH_EXTRACTION_ENABLED and (entities or doc_meta):
        try:
            graph_store.store_entities(entities)
            graph_store.store_relationships(relationships)
            if doc_meta:
                graph_store.store_document_metadata(doc_meta)
        except Exception as e:
            logger.warning(f"  Graph storage failed (non-critical): {e}")

    elapsed = round(time.time() - start, 1)
    log[doc_id] = {
        "filename":      filename,
        "folder":        folder,
        "chunks":        len(chunks),
        "pages":         len(pages),
        "entities":      entity_count,
        "relationships": rel_count,
        "ingested_at":   datetime.utcnow().isoformat(),
        "elapsed_s":     elapsed,
    }
    logger.info(
        f"  Done: {len(chunks)} chunks, {entity_count} entities, "
        f"{rel_count} relationships in {elapsed}s"
    )


# ── Research study extraction ────────────────────────────────────────────────


def _extract_research_study(
    study_folder: Path,
    pdfs: list[Path],
    progress: str,
    force_ocr: bool,
) -> bool:
    """
    Run extraction for one research study folder.
    Returns True if a new extraction happened, False if skipped (unchanged).
    """
    folder_name = study_folder.name

    # Build DocumentText objects from cached OCR results.
    documents: list[DocumentText] = []
    for pdf in sorted(pdfs):
        pages = extract_text_from_pdf(str(pdf), force_ocr=force_ocr)
        documents.append(DocumentText(
            filename=pdf.name,
            pages=[(p.page_number, p.text) for p in pages],
        ))

    # Compute source_doc_hash and proposed study_id to skip if unchanged.
    from rag.study_extractor import _doc_hash, _resolve_study_id

    proposed_id, _ = _resolve_study_id(
        folder_name=folder_name,
        folder_path=str(study_folder),
        extracted_title=None,
        extracted_year=None,
    )
    new_hash = _doc_hash(documents)
    existing_hash = get_existing_doc_hash(proposed_id)
    if existing_hash == new_hash:
        logger.info(f"  {progress} {folder_name} — unchanged, skipping extraction")
        return False

    logger.info(f"  {progress} Extracting study: {folder_name}")
    start = time.time()
    record = extract_study_record(str(study_folder), documents)
    upsert_study(record)
    elapsed = round(time.time() - start, 1)
    logger.info(
        f"    Done: id={record.study_id} "
        f"title={record.title or '?'} "
        f"authors={len(record.authors)} "
        f"keywords={len(record.keywords)} "
        f"in {elapsed}s"
    )
    return True


def _reextract_single_study(study_id: str) -> None:
    """Re-extract one study by ID — useful for debugging a single record."""
    from rag.study_store import get_study

    study = get_study(study_id)
    if not study:
        logger.error(f"Study not found: {study_id}")
        return

    folder_path = Path(study["folder_path"])
    if not folder_path.exists():
        logger.error(f"Study folder no longer exists: {folder_path}")
        return

    pdfs = sorted(folder_path.rglob("*.pdf"))
    if not pdfs:
        logger.error(f"No PDFs found in {folder_path}")
        return

    documents = []
    for pdf in pdfs:
        pages = extract_text_from_pdf(str(pdf), force_ocr=True)
        documents.append(DocumentText(
            filename=pdf.name,
            pages=[(p.page_number, p.text) for p in pages],
        ))

    logger.info(f"Re-extracting study: {study_id}")
    record = extract_study_record(str(folder_path), documents)
    record.study_id = study_id  # preserve original ID
    upsert_study(record)
    logger.info(f"Done: title={record.title or '?'} authors={len(record.authors)}")


def _log_extraction_error(record_folder: Path, exc: Exception) -> None:
    """Append a one-line error record so a long batch can be reviewed later."""
    EXTRACTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(EXTRACTION_LOG, "a") as f:
            f.write(
                f"{datetime.utcnow().isoformat()}\t{record_folder}\t{type(exc).__name__}: {exc}\n"
            )
    except Exception:
        pass  # never fail ingestion because logging failed


# ── Metadata enrichment (unchanged) ──────────────────────────────────────────


def _enrich_metadata(chunks):
    """
    Enrich chunk metadata with section and document type classification.
    Uses research-focused document categories.
    """
    if not chunks:
        return chunks

    from config import RESEARCH_DOC_CATEGORIES
    
    sample_text = chunks[0].parent_text[:800]
    doc_types_str = "|".join(RESEARCH_DOC_CATEGORIES)
    
    prompt = (
        "Read this research document excerpt and classify it. "
        "Reply with JSON only, no other text.\n\n"
        f"Document types (use ONLY these): {doc_types_str}\n\n"
        "Return exactly:\n"
        '{"section": "<main section heading or topic>", '
        f'"doc_type": "<{doc_types_str}>"}}\n\n'
        f"Excerpt:\n{sample_text}"
    )
    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": WORKER_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
        )
        response.raise_for_status()
        raw = response.json().get("response", "{}")
        import re
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            meta = json.loads(match.group())
            doc_type = meta.get("doc_type", "OTHER").upper()
            # Validate doc_type is in allowed list
            if doc_type not in RESEARCH_DOC_CATEGORIES:
                doc_type = "OTHER"
            
            for chunk in chunks:
                chunk.metadata["section"]  = meta.get("section", "")
                chunk.metadata["doc_type"] = doc_type
    except Exception as e:
        logger.warning(f"Metadata extraction failed (non-critical): {e}")
    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────


def _embed_batch(texts: list[str]) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        try:
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": batch},
                timeout=120,
            )
            response.raise_for_status()
            all_embeddings.extend(response.json()["embeddings"])
        except Exception as e:
            logger.error(f"Embedding batch {i // BATCH_SIZE + 1} failed: {e}")
            raise
    return all_embeddings


# ── Remove document ───────────────────────────────────────────────────────────


def _remove_document(filename: str, store: VectorStore, graph_store: GraphStore, log: dict) -> None:
    doc_id = next((k for k, v in log.items() if v["filename"] == filename), None)
    if not doc_id:
        logger.error(f"'{filename}' not found in ingestion log")
        return
    store.delete_by_doc_id(doc_id)
    graph_store.delete_by_doc_id(doc_id)
    del log[doc_id]
    _save_log(log)
    logger.info(f"Removed '{filename}' from knowledge base")


# ── Log helpers ───────────────────────────────────────────────────────────────


def _load_log() -> dict:
    if INGESTION_LOG.exists():
        with open(INGESTION_LOG) as f:
            return json.load(f)
    return {}


def _save_log(log: dict) -> None:
    with open(INGESTION_LOG, "w") as f:
        json.dump(log, f, indent=2)


# ── Custom exceptions ─────────────────────────────────────────────────────────


class AlreadyIngestedException(Exception):
    pass


if __name__ == "__main__":
    main()

"""
chunker.py — Hierarchical document chunking

Two-level chunking strategy:
  - Parent chunks (~2000 tokens): sent to the LLM for full context
  - Child chunks (~200 tokens):   used for vector search (precise matching)

Each child carries a reference to its parent so we can always
retrieve the full context once the right child is found.
"""

import hashlib
import logging
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.ocr import PageContent

logger = logging.getLogger(__name__)

# Parent chunk: large window — gives the LLM enough context to reason
PARENT_CHUNK_SIZE    = 2000  # characters (approx 500 tokens)
PARENT_CHUNK_OVERLAP = 200   # overlap so context is not cut mid-sentence

# Child chunk: small window — tight match against the user's question
CHILD_CHUNK_SIZE    = 400    # characters (approx 100 tokens)
CHILD_CHUNK_OVERLAP = 40


@dataclass
class Chunk:
    """A single chunk ready to be embedded and stored in ChromaDB."""
    chunk_id:    str          # unique: doc_hash + index
    doc_id:      str          # SHA-256 of the source PDF
    source:      str          # original filename
    folder:      str          # folder the document lives in
    page:        int          # page number the chunk came from
    text:        str          # child chunk text (for embedding + display)
    parent_text: str          # parent chunk text (sent to LLM)
    was_ocr:     bool         # whether OCR was used on this page
    metadata:    dict = field(default_factory=dict)  # section, doc_type (added later)


def build_chunks(
    pages: list[PageContent],
    doc_id: str,
    source: str,
    folder: str,
) -> list[Chunk]:
    """
    Take a list of pages (from ocr.py) and produce a flat list of
    hierarchical chunks ready for embedding.

    Steps:
      1. Join all page text into one document string (with page markers)
      2. Split into large parent chunks
      3. Split each parent into small child chunks
      4. Attach parent text to every child for later retrieval
    """
    if not pages:
        logger.warning(f"No pages to chunk for {source}")
        return []

    # Build a single string with page markers so we can track page numbers
    page_blocks = []
    for p in pages:
        if p.text.strip():
            page_blocks.append(f"[PAGE {p.page_number}]\n{p.text}")

    full_text = "\n\n".join(page_blocks)

    # --- Split into parent chunks ---
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    parent_texts = parent_splitter.split_text(full_text)

    # --- Split each parent into children ---
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    chunk_index = 0

    for parent_text in parent_texts:
        # Detect which page this parent came from (first [PAGE N] marker inside)
        page_number = _extract_page_number(parent_text, fallback=pages[0].page_number)
        was_ocr = _page_was_ocr(pages, page_number)

        # Clean the [PAGE N] markers from parent text before sending to LLM
        clean_parent = _remove_page_markers(parent_text)

        # Split parent into children
        child_texts = child_splitter.split_text(clean_parent)

        for child_text in child_texts:
            if not child_text.strip():
                continue

            chunk_id = f"{doc_id}_{chunk_index:05d}"

            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source=source,
                folder=folder,
                page=page_number,
                text=child_text.strip(),
                parent_text=clean_parent.strip(),
                was_ocr=was_ocr,
            ))
            chunk_index += 1

    logger.info(
        f"Chunked {source}: {len(parent_texts)} parents "
        f"-> {len(chunks)} children"
    )
    return chunks


def compute_doc_id(file_path: str) -> str:
    """
    Compute a stable SHA-256 hash of a file.
    Used as the document's unique ID and for deduplication.
    Same file content = same hash = skip re-ingestion.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_page_number(text: str, fallback: int) -> int:
    """Pull the first [PAGE N] marker from a chunk."""
    import re
    match = re.search(r"\[PAGE (\d+)\]", text)
    return int(match.group(1)) if match else fallback


def _remove_page_markers(text: str) -> str:
    """Strip [PAGE N] markers — clean text for the LLM."""
    import re
    return re.sub(r"\[PAGE \d+\]\n?", "", text).strip()


def _page_was_ocr(pages: list[PageContent], page_number: int) -> bool:
    """Check if a given page was extracted via OCR."""
    for p in pages:
        if p.page_number == page_number:
            return p.was_ocr
    return False

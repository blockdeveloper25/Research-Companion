"""
ocr.py — PDF text extraction with OCR fallback

Handles two types of PDFs:
  - Text PDFs: extract text directly with PyMuPDF (fast, accurate)
  - Scanned PDFs: convert each page to an image, run Tesseract OCR

Features added for medical/scanned document quality:
  - File-hash-keyed cache: re-ingestion never re-OCRs the same file
  - force_ocr=True bypasses the heuristic and OCRs every page (used for
    patient documents where embedded text often contains only page numbers)
  - Image preprocessing (grayscale + autocontrast + binarization) before OCR
  - Per-page confidence scoring; low-confidence pages logged for review
  - Optional thread-parallel OCR within a single PDF (Tesseract releases the
    GIL during the C call, so threads scale well)

Returns a list of pages, each with its text, page number, and OCR confidence.
"""

import hashlib
import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageOps

from config import (
    MIN_TEXT_CHARS,
    OCR_CACHE_DIR,
    OCR_CACHE_ENABLED,
    OCR_DPI,
    OCR_LOW_CONFIDENCE_LOG,
    OCR_LOW_CONFIDENCE_THRESHOLD,
    OCR_MAX_WORKERS,
    OCR_PARALLEL_PAGES,
    OCR_PREPROCESS,
    OCR_TESSERACT_PSM,
)

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    page_number: int          # 1-indexed
    text: str
    was_ocr: bool             # True if OCR was used for this page
    ocr_confidence: float = 0.0  # 0–100; only meaningful when was_ocr is True


# ── Public API ────────────────────────────────────────────────────────────────


def extract_text_from_pdf(
    pdf_path: str,
    force_ocr: bool = False,
) -> list[PageContent]:
    """
    Open a PDF and return text for every page.

    Args:
        pdf_path:  path to the PDF on disk
        force_ocr: if True, every page is OCR'd regardless of embedded text.
                   Use this for scanned medical documents where the embedded
                   text layer is unreliable (page numbers, headers only).

    Cached: results are stored under backend/ocr_cache/<file_hash>.json
    keyed by SHA-256 of the file bytes plus the force_ocr flag. A second
    call with the same file (or a re-ingest) returns instantly.
    """
    pdf_path = str(pdf_path)

    # ── Cache lookup ────────────────────────────────────────────────────────
    cache_key = _cache_key(pdf_path, force_ocr=force_ocr)
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.info(f"OCR cache hit: {os.path.basename(pdf_path)}")
        return cached

    # ── Open and process ────────────────────────────────────────────────────
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path}: {e}")
        raise

    logger.info(
        f"Processing {os.path.basename(pdf_path)} — {len(doc)} pages"
        + (" (force_ocr=True)" if force_ocr else "")
    )

    # First pass: decide native vs OCR for every page (cheap, sequential).
    # We collect render-ready handles for OCR pages so the second pass can
    # parallelize the slow Tesseract calls.
    decisions: list[tuple[int, str, bool]] = []  # (page_num, native_text, needs_ocr)
    ocr_jobs: list[tuple[int, fitz.Page]] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1

        if force_ocr:
            decisions.append((page_number, "", True))
            ocr_jobs.append((page_number, page))
            continue

        raw_text = page.get_text("text")
        clean_text = _clean_text(raw_text)
        raw_stripped = raw_text.strip()

        # If we stripped >40% of raw content, font encoding is unreliable
        heavy_garble = (
            len(raw_stripped) > MIN_TEXT_CHARS
            and len(clean_text) < len(raw_stripped) * 0.60
        )

        if (
            len(clean_text) >= MIN_TEXT_CHARS
            and not _is_garbled(clean_text)
            and not heavy_garble
        ):
            # Native text page — also try to extract structured tables
            table_md = _extract_tables_as_markdown(page)
            final_text = clean_text + ("\n\n" + table_md if table_md else "")
            decisions.append((page_number, final_text, False))
        else:
            reason = (
                "scanned" if len(raw_stripped) < MIN_TEXT_CHARS
                else "garbled encoding"
            )
            logger.info(f"  Page {page_number}: {reason} → OCR")
            decisions.append((page_number, clean_text, True))
            ocr_jobs.append((page_number, page))

    # Second pass: OCR everything that needs it. Tesseract releases the GIL
    # during its C call, so a thread pool gives a real speedup.
    ocr_results: dict[int, tuple[str, float]] = {}
    if ocr_jobs:
        ocr_results = _run_ocr_jobs(ocr_jobs, pdf_path)

    # Stitch the final page list together
    pages: list[PageContent] = []
    for page_number, native_text, needs_ocr in decisions:
        if needs_ocr:
            ocr_text, confidence = ocr_results.get(page_number, ("", 0.0))
            # If OCR produced less text than the (garbled) native text, keep native
            final_text = ocr_text if len(ocr_text) > len(native_text) else native_text
            pages.append(PageContent(
                page_number=page_number,
                text=final_text,
                was_ocr=True,
                ocr_confidence=confidence,
            ))
        else:
            pages.append(PageContent(
                page_number=page_number,
                text=native_text,
                was_ocr=False,
                ocr_confidence=0.0,
            ))

    doc.close()

    total_ocr = sum(1 for p in pages if p.was_ocr)
    low_conf = sum(
        1 for p in pages
        if p.was_ocr and p.ocr_confidence < OCR_LOW_CONFIDENCE_THRESHOLD
    )
    logger.info(
        f"Done — {len(pages)} pages, {total_ocr} via OCR, "
        f"{len(pages) - total_ocr} native"
        + (f", {low_conf} low-confidence" if low_conf else "")
    )

    _cache_put(cache_key, pages)
    return pages


# ── OCR execution ─────────────────────────────────────────────────────────────


def _run_ocr_jobs(
    jobs: list[tuple[int, fitz.Page]],
    pdf_path: str,
) -> dict[int, tuple[str, float]]:
    """
    Run Tesseract on a batch of pages, optionally in parallel.
    Returns {page_number: (text, mean_confidence)}.
    """
    # Render all page pixmaps first (must happen on the main thread because
    # fitz.Page objects are not thread-safe). OCR happens on the workers.
    rendered: list[tuple[int, bytes]] = []
    for page_number, page in jobs:
        matrix = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        rendered.append((page_number, pixmap.tobytes("png")))

    def _ocr_one(item: tuple[int, bytes]) -> tuple[int, str, float]:
        page_number, png_bytes = item
        try:
            text, confidence = _ocr_image_bytes(png_bytes)
        except Exception as e:
            logger.warning(f"  OCR failed on page {page_number}: {e}")
            return (page_number, "", 0.0)
        return (page_number, text, confidence)

    results: dict[int, tuple[str, float]] = {}
    if OCR_PARALLEL_PAGES and len(rendered) > 1:
        with ThreadPoolExecutor(max_workers=OCR_MAX_WORKERS) as pool:
            for page_number, text, confidence in pool.map(_ocr_one, rendered):
                results[page_number] = (text, confidence)
    else:
        for item in rendered:
            page_number, text, confidence = _ocr_one(item)
            results[page_number] = (text, confidence)

    # Log low-confidence pages so they can be reviewed later
    for page_number, (_, confidence) in results.items():
        if confidence and confidence < OCR_LOW_CONFIDENCE_THRESHOLD:
            _log_low_confidence(pdf_path, page_number, confidence)

    return results


def _ocr_image_bytes(png_bytes: bytes) -> tuple[str, float]:
    """
    OCR a single page image. Returns (text, mean_word_confidence_0_to_100).
    """
    image = Image.open(io.BytesIO(png_bytes))
    if OCR_PREPROCESS:
        image = _preprocess_for_ocr(image)

    config = f"--oem 3 --psm {OCR_TESSERACT_PSM}"

    # image_to_data gives per-word confidence, which we average for the page
    data = pytesseract.image_to_data(
        image,
        lang="eng",
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    words: list[str] = []
    confidences: list[float] = []
    for word, conf in zip(data.get("text", []), data.get("conf", [])):
        if not word or not word.strip():
            continue
        try:
            conf_num = float(conf)
        except (TypeError, ValueError):
            continue
        if conf_num < 0:  # -1 means "no confidence reported"
            continue
        words.append(word)
        confidences.append(conf_num)

    text = " ".join(words)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return text, mean_conf


def _preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """
    Cheap, dependency-free image cleanup that meaningfully improves Tesseract
    accuracy on scanned medical documents:
      1. Convert to grayscale
      2. Autocontrast (stretch histogram across full 0–255 range)
      3. Otsu-like binarization via point() with the mean as threshold
    """
    img = ImageOps.grayscale(image)
    img = ImageOps.autocontrast(img, cutoff=1)
    # Simple thresholding — pixels above the mean become white, below black.
    # This isn't true Otsu but works well for printed forms without OpenCV.
    histogram = img.histogram()
    total_pixels = sum(histogram)
    weighted_sum = sum(i * count for i, count in enumerate(histogram))
    mean_value = int(weighted_sum / total_pixels) if total_pixels else 128
    threshold = max(120, min(200, mean_value))  # clamp to a sensible range
    img = img.point(lambda px: 255 if px > threshold else 0, mode="1")
    return img


# ── Cache ─────────────────────────────────────────────────────────────────────


def _cache_dir() -> Path:
    """Resolve the cache directory under backend/, creating it if needed."""
    backend_root = Path(__file__).resolve().parent.parent
    path = backend_root / OCR_CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_key(pdf_path: str, *, force_ocr: bool) -> str:
    """SHA-256 of file bytes + force_ocr flag — uniquely identifies the run."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    h.update(b"\x01" if force_ocr else b"\x00")
    return h.hexdigest()


def _cache_get(key: str) -> list[PageContent] | None:
    if not OCR_CACHE_ENABLED:
        return None
    cache_file = _cache_dir() / f"{key}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file) as f:
            payload = json.load(f)
        return [PageContent(**page) for page in payload]
    except Exception as e:
        logger.warning(f"OCR cache read failed for {key}: {e}")
        return None


def _cache_put(key: str, pages: list[PageContent]) -> None:
    if not OCR_CACHE_ENABLED:
        return
    cache_file = _cache_dir() / f"{key}.json"
    try:
        with open(cache_file, "w") as f:
            json.dump([asdict(p) for p in pages], f)
    except Exception as e:
        logger.warning(f"OCR cache write failed for {key}: {e}")


def _log_low_confidence(pdf_path: str, page_number: int, confidence: float) -> None:
    """Append a low-confidence page record to the review log."""
    backend_root = Path(__file__).resolve().parent.parent
    log_path = backend_root / OCR_LOW_CONFIDENCE_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "a") as f:
            f.write(f"{pdf_path}\tpage={page_number}\tconf={confidence:.1f}\n")
    except Exception:
        pass  # never fail ingestion because logging failed


# ── Table extraction (unchanged from previous version) ───────────────────────


def _extract_tables_as_markdown(page: fitz.Page) -> str | None:
    """
    Use PyMuPDF's built-in table detector to extract tables as markdown.
    Preserves column order and special characters that flat text loses.
    """
    try:
        tabs = page.find_tables()
        if not tabs.tables:
            return None

        md_tables = []
        for table in tabs.tables:
            rows = table.extract()
            if not rows or len(rows) < 2:
                continue

            md_rows = []
            for i, row in enumerate(rows):
                cells = [_normalise_cell(cell) for cell in row]
                md_rows.append("| " + " | ".join(cells) + " |")
                if i == 0:
                    md_rows.append("| " + " | ".join(["---"] * len(cells)) + " |")

            md_tables.append("\n".join(md_rows))

        return "\n\n".join(md_tables) if md_tables else None
    except Exception as e:
        logger.debug(f"Table extraction skipped: {e}")
        return None


def _normalise_cell(cell: object) -> str:
    if cell is None:
        return ""
    text = str(cell).strip()
    text = _normalise_tick_cross(text)
    return text.replace("|", "\\|")


def _normalise_tick_cross(text: str) -> str:
    stripped = text.strip()
    if stripped in {"v", "V", "y", "Y", "J", "√", "\u2713", "\u2714", "✓"}:
        return "✓"
    if stripped in {"x", "X", "×", "\u2717", "\u2718", "✗", "✕"}:
        return "✗"
    return text


# ── Text cleanup helpers (unchanged) ─────────────────────────────────────────


def _is_garbled(text: str) -> bool:
    """Detect pages where PDF font encoding produced garbage characters."""
    if not text:
        return True
    readable = sum(
        1 for c in text
        if c.isalpha() or c.isspace() or c in ".,;:!?-()'\"[]0123456789"
    )
    return (readable / len(text)) < 0.70


def _clean_text(text: str) -> str:
    """Drop blank lines and lines that look like garbled font-encoding artefacts."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        percent_count = stripped.count("%")
        alpha_space = sum(1 for c in stripped if c.isalpha() or c.isspace())
        if len(stripped) > 10:
            if percent_count / len(stripped) > 0.08:
                continue
            if alpha_space / len(stripped) < 0.50:
                continue
        cleaned.append(stripped)
    return "\n".join(cleaned)

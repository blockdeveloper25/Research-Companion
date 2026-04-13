# Changelog

All notable changes to this project are documented in this file.

The format follows the "Keep a Changelog" convention and uses Semantic Versioning.

## [2.3.0] - 2026-04-13

### Added

- In-app document ingestion (no CLI/command panel required)
  - Upload PDFs directly from the UI via the Upload modal (`frontend/src/components/UploadModal.tsx`)
  - Drag-and-drop or file picker selection; supports multiple PDFs per upload
  - Optional folder name to organize uploaded documents in the sidebar
  - Backend upload endpoint `POST /api/documents/upload` saves files and queues ingestion as a background task

## [2.2.0] - 2026-04-07

### Added

- Track 1: structured study extraction layer for research papers. Enables reliable metadata + field-level provenance so you can ask aggregation questions like "list studies published after 2020" or "how many papers used dataset X".
- OCR pipeline upgrades (`backend/rag/ocr.py`)
  - File-hash-keyed cache so re-ingestion never re-OCRs the same file (cache at `backend/ocr_cache/`, gitignored)
  - `force_ocr=True` bypasses the embedded-text heuristic; enabled automatically for `--type research` because many PDFs have unreliable embedded text layers
  - Image preprocessing: grayscale + autocontrast + binarization
  - Per-page OCR confidence scoring; pages below 60 are logged to `logs/ocr_low_confidence.log` for review
  - Thread-parallel, page-level OCR (Tesseract releases the GIL)
  - Tesseract `--psm 6` for dense, form-like layouts
- New schema (`backend/db/connection.py`)
  - `studies` table with GIN indexes on `authors`, `keywords`, `data_sources` for fast multi-value filters
  - `study_field_provenance` table: every extracted field links back to its source file and page
  - `folders` table registers each ingested folder as `research`, `policy`, or `general`
- Map-reduce extractor (`backend/rag/study_extractor.py`)
  - One LLM call per document (MAP) plus a merge step (REDUCE)
  - Strict JSON output via Ollama `format=json`
  - Configurable model via `STUDY_EXTRACTION_MODEL` (default `llama3.1:8b`)
  - Study ID resolution: folder name → extracted title+year → hash fallback
  - Prompt instructs the model to return `null` for missing fields (never guess)
- Persistence layer (`backend/rag/study_store.py`)
  - Idempotent UPSERT keyed on `study_id`
  - `source_doc_hash` enables skip-if-unchanged for fast re-runs
- Ingest CLI changes (`backend/ingest.py`)
  - `--type general|research|policy` (default: `general`)
  - `--no-extract` to skip structured extraction (chunks only; fast for testing)
  - `--extract-only` to re-run extraction without re-chunking
  - `--force-ocr` to bypass the OCR heuristic
  - `--extract-study STUDY_ID` to re-extract a single study
  - Research mode groups PDFs by parent folder and runs extraction per study
  - Resumable ingestion: skips studies whose source documents have not changed
  - Extraction errors logged to `logs/extraction_errors.log` so long batches can continue
- Validation API
  - `GET /api/studies/stats`: field completeness across all extracted studies
  - `GET /api/studies/{study_id}`: full study record plus provenance trail

### Changed

- Configuration (`backend/config.py`)
  - Added OCR settings: `OCR_CACHE_ENABLED`, `OCR_PARALLEL_PAGES`, `OCR_MAX_WORKERS`, `OCR_PREPROCESS`, `OCR_TESSERACT_PSM`, `OCR_LOW_CONFIDENCE_THRESHOLD`, `OCR_ENGINE`
  - Added study extraction settings: `STUDY_EXTRACTION_ENABLED`, `STUDY_EXTRACTION_MODEL`, `STUDY_EXTRACTION_MAX_CHARS`, `STUDY_EXTRACTION_TIMEOUT_S`

### Notes

- Use the validation API endpoints to spot-check extracted fields and provenance.
- Tesseract is the only OCR engine wired in this release. The `OCR_ENGINE` flag exists so PaddleOCR/docTR/Surya can be added later without changing call sites.

## [2.1.0] - 2026-04-05

### Fixed

- Backend port mismatch: `start.sh` launched backend on port 8001 but the Vite proxy targeted port 8000; unified to port 8000
- Empty state flash on startup: frontend now retries API connection with a loading spinner instead of showing "No conversations yet" while backend starts

### Changed

- Frontend dev server port changed from 5173 to 5457
- Added a re-ingest hint to the `start.sh` startup message for when entity/relation types are changed in `config.py`

## [2.0.0] - 2026-03-31

### Added

- Initial release with PDF ingestion, a hybrid RAG pipeline, a knowledge graph, and an entity browser

"""
config.py — Central configuration for the Knowledge Companion backend.
All environment-driven settings live here.
"""

import os

# ── Database ──────────────────────────────────────────────────────────────────

# Full PostgreSQL connection URL
# Format: postgresql://user:password@host:port/dbname
# Override via environment variable in production
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/knowledge_companion",
)

# ── Ollama ────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL     = "nomic-embed-text"
WORKER_MODEL    = "llama3.2:3b"
REASONER_MODEL  = "llama3.1:8b"

# ── Ingestion ─────────────────────────────────────────────────────────────────

EMBED_BATCH_SIZE = 32     # chunks per Ollama embedding call
MIN_TEXT_CHARS   = 50     # below this → treat page as scanned
OCR_DPI          = 300    # render DPI for scanned pages

# ── OCR ───────────────────────────────────────────────────────────────────────

OCR_ENGINE              = "tesseract"   # "tesseract" — only engine wired today.
                                        # Future: "paddle" | "doctr" | "surya"
OCR_CACHE_ENABLED       = True          # Cache OCR results by file hash so
                                        # re-ingestion never re-OCRs the same file
OCR_CACHE_DIR           = "ocr_cache"   # Created under backend/ if missing
OCR_PARALLEL_PAGES      = True          # OCR multiple pages of one PDF in parallel
OCR_MAX_WORKERS         = 4             # Thread pool size for page-level parallelism
OCR_PREPROCESS          = True          # Grayscale + autocontrast + binarize
OCR_TESSERACT_PSM       = 6             # Page segmentation mode 6 = uniform block,
                                        # works better for medical forms than auto
OCR_LOW_CONFIDENCE_THRESHOLD = 60.0     # Pages below this mean confidence get
                                        # logged to logs/ocr_low_confidence.log
OCR_LOW_CONFIDENCE_LOG  = "logs/ocr_low_confidence.log"

# ── Study extraction ────────────────────────────────────────────────────────

STUDY_EXTRACTION_ENABLED   = True
STUDY_EXTRACTION_MODEL     = "llama3.1:8b"   # Accuracy > speed for research papers
STUDY_EXTRACTION_MAX_CHARS = 30000           # Truncate huge studies
STUDY_EXTRACTION_TIMEOUT_S = 120             # Per-study hard cap

# ── Retrieval ─────────────────────────────────────────────────────────────────

SEMANTIC_TOP_K  = 20      # candidates from pgvector search
BM25_TOP_K      = 20      # candidates from BM25
FINAL_TOP_K     = 5       # survivors after re-ranking
SCORE_THRESHOLD = 0.45    # minimum re-rank score to proceed

# ── Pipeline ──────────────────────────────────────────────────────────────────

MAX_HISTORY_TURNS    = 12   # sliding window of chat history
PARENT_CHUNK_SIZE    = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE     = 400
CHILD_CHUNK_OVERLAP  = 40

# ── Knowledge Graph ──────────────────────────────────────────────────────────

GRAPH_EXTRACTION_ENABLED = True   # kill switch — set False to skip entirely

# Research-focused entity and relationship types for academic papers and studies
# Fixed vocabulary forces the 3b model to categorize consistently.
# These types are optimized for research papers, not policy documents.
GRAPH_ENTITY_TYPES = [
    # Core research concepts
    "RESEARCH_QUESTION",      # Central question being investigated
    "HYPOTHESIS",             # Proposed answer or theoretical prediction
    "FINDING",                # Key result or discovery
    "CONCLUSION",             # Final interpretation of findings
    
    # Methodology & data
    "METHODOLOGY",            # Experimental or analytical method
    "DATASET",                # Data source or collection
    "ALGORITHM",              # Computational or analytical algorithm
    "STATISTICAL_MEASURE",    # Specific statistical result (p-value, correlation, etc.)
    
    # Research structure
    "LIMITATION",             # Known constraints or weaknesses
    "ASSUMPTION",             # Underlying assumptions made
    "KEY_FINDING",            # Highlighted significant result
    "RESEARCH_DOMAIN",        # Field of study (e.g., neuroscience, ML)
    
    # People & institutions
    "AUTHOR",                 # Researcher/author name
    "AFFILIATION",            # Institution or organization
    "FUNDING_BODY",           # Funding source or grant program
    
    # References & citations
    "CITATION",               # Referenced work or study
    "PUBLICATION_INFO",       # Journal, conference, or publication metadata
    
    # Fallback
    "OTHER",
]

GRAPH_RELATION_TYPES = [
    # Research logic & validation
    "ADDRESSES",              # Hypothesis addresses research question
    "VALIDATES",              # Finding validates hypothesis
    "CONTRADICTS",            # Finding contradicts prior work
    "SUPPORTS",               # Finding supports conclusion
    "CONTRADICTS_FINDING",    # Two findings are contradictory
    "DEMONSTRATES",           # Result demonstrates principle
    
    # Knowledge building & citations
    "BUILDS_ON",              # Extends prior research
    "REPLICATES_STUDY",       # Replicates or validates prior study
    "DISPUTES",               # Disputes or challenges prior finding
    "CITES",                  # References or cites work
    "EXTENDS",                # Extends methodology or theory
    "PROPOSES",               # Proposes new approach/method
    
    # Method & data
    "USES_METHODOLOGY",       # Applies specific method
    "USES_DATASET",           # Analysis uses this data
    "MEASURES",               # Statistical measure quantifies finding
    "COMPARES_TO",            # Comparison with other studies/methods
    
    # Attribution
    "AUTHORED_BY",            # Paper authored by person
    "AFFILIATED_WITH",        # Author affiliated with institution
    "FUNDED_BY",              # Research funded by organization
    
    # General
    "RELATED_TO",             # General relationship
]

GRAPH_MAX_ENTITIES_PER_CHUNK  = 10   # cap per parent chunk
GRAPH_MAX_RELS_PER_CHUNK      = 15
GRAPH_EXTRACTION_BATCH_SIZE   = 5    # parent chunks per LLM call
GRAPH_TRAVERSAL_MAX_DEPTH     = 3    # hops in recursive CTE
GRAPH_TRAVERSAL_MAX_NODES     = 50   # total nodes returned from traversal

# ── Research Document Categories ──────────────────────────────────────────────
#
# Fixed vocabulary for document types ensures 3b model categorizes consistently.
# All categories are research-focused.

RESEARCH_DOC_CATEGORIES = [
    "JOURNAL_ARTICLE",      # Peer-reviewed published journal articles
    "CONFERENCE_PAPER",     # Papers in conference proceedings / workshops
    "PREPRINT",             # ArXiv, bioRxiv, medRxiv, etc.
    "DISSERTATION",         # PhD dissertation, Masters thesis, etc.
    "TECHNICAL_REPORT",     # Technical reports from institutions/labs/companies
    "REVIEW_PAPER",         # Literature reviews, systematic reviews, surveys
    "CASE_STUDY",           # Case study documentation and analysis
    "WHITE_PAPER",          # Technical/research white papers
    "BENCHMARK_DATASET",    # Dataset papers, benchmark documentation
    "OTHER",                # Fallback for unclassified research documents
]

# Human-readable descriptions for categories
RESEARCH_DOC_CATEGORY_DESCRIPTIONS = {
    "JOURNAL_ARTICLE": "Peer-reviewed article published in a scientific journal",
    "CONFERENCE_PAPER": "Research paper from a conference or workshop proceedings",
    "PREPRINT": "Pre-publication preprint (ArXiv, bioRxiv, etc.)",
    "DISSERTATION": "PhD dissertation or Masters thesis",
    "TECHNICAL_REPORT": "Technical or research report from an institution or company",
    "REVIEW_PAPER": "Survey, literature review, or systematic review paper",
    "CASE_STUDY": "Case study or detailed analysis publication",
    "WHITE_PAPER": "Technical white paper or position paper",
    "BENCHMARK_DATASET": "Dataset paper or benchmark documentation",
    "OTHER": "Other research-related document",
}

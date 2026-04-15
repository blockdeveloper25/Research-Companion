"""
connection.py — PostgreSQL connection pool shared across the entire backend.

One pool serves both the vector store and the session store.
Uses psycopg2 ThreadedConnectionPool — safe for FastAPI's threaded workers.

pgvector's vector type is registered on every connection so Postgres
knows how to serialise/deserialise the 768-dim float arrays.

Usage:
    from db.connection import get_conn, release_conn, init_db

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(...)
        conn.commit()
    finally:
        release_conn(conn)

Or use the context manager:
    with db_conn() as conn:
        ...
"""

import logging
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool
from pgvector.psycopg2 import register_vector

from config import DB_URL

logger = logging.getLogger(__name__)

# Connection pool — min 2 connections always open, max 10
# For a single-user local app, 10 is more than enough
_pool: ThreadedConnectionPool | None = None


def init_pool() -> None:
    """
    Initialise the connection pool. Called once at application startup.
    Creates all tables and indexes if they don't exist.
    """
    global _pool

    _pool = ThreadedConnectionPool(
        minconn=2,
        maxconn=10,
        dsn=DB_URL,
    )
    logger.info("PostgreSQL connection pool initialised")

    # Bootstrap the schema on first run
    init_db()


def get_conn() -> psycopg2.extensions.connection:
    """
    Borrow a connection from the pool.
    Always pair with release_conn() in a finally block,
    or use the db_conn() context manager instead.
    """
    if _pool is None:
        raise RuntimeError("Connection pool not initialised — call init_pool() first")

    conn = _pool.getconn()

    # Register pgvector type so Python lists become Postgres vectors
    register_vector(conn)

    # Return dicts instead of tuples for cursor rows
    conn.cursor_factory = psycopg2.extras.RealDictCursor

    return conn


def release_conn(conn: psycopg2.extensions.connection) -> None:
    """Return a connection to the pool."""
    if _pool:
        _pool.putconn(conn)


@contextmanager
def db_conn():
    """
    Context manager — borrows a connection, commits on success,
    rolls back on exception, always releases back to pool.

    Usage:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ...")
    """
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        release_conn(conn)


def init_db() -> None:
    """
    Create all tables, indexes, and extensions if they don't exist.
    Safe to call on every startup — uses CREATE IF NOT EXISTS throughout.
    """
    with db_conn() as conn:
        with conn.cursor() as cur:

            # ── pgvector extension ────────────────────────────────────────────
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # ── Chunks table (replaces ChromaDB) ─────────────────────────────
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id           TEXT PRIMARY KEY,
                    doc_id       TEXT        NOT NULL,
                    source       TEXT        NOT NULL,
                    folder       TEXT        NOT NULL,
                    page         INTEGER     NOT NULL,
                    content      TEXT        NOT NULL,
                    parent_text  TEXT        NOT NULL,
                    embedding    vector(768) NOT NULL,
                    was_ocr      BOOLEAN     NOT NULL DEFAULT FALSE,
                    section      TEXT        NOT NULL DEFAULT '',
                    doc_type     TEXT        NOT NULL DEFAULT '',
                    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            # HNSW index — best accuracy for cosine similarity search
            # Builds incrementally as rows are inserted (no training needed)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                ON chunks USING hnsw (embedding vector_cosine_ops);
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
                ON chunks(doc_id);
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_folder
                ON chunks(folder);
            """)

            # ── Sessions table (replaces SQLite sessions) ─────────────────────
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id          TEXT PRIMARY KEY,
                    title       TEXT        NOT NULL,
                    folders     JSONB       NOT NULL DEFAULT '[]',
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    is_active   BOOLEAN     NOT NULL DEFAULT TRUE
                );
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_updated
                ON sessions(updated_at DESC);
            """)

            # ── Messages table (replaces SQLite messages) ─────────────────────
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id          TEXT PRIMARY KEY,
                    session_id  TEXT        NOT NULL REFERENCES sessions(id),
                    role        TEXT        NOT NULL,
                    content     TEXT        NOT NULL,
                    sources     JSONB       NOT NULL DEFAULT '[]',
                    confidence  JSONB       NOT NULL DEFAULT '{}',
                    model_used  TEXT        NOT NULL DEFAULT '',
                    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages(session_id);
            """)

            # ── Knowledge Graph: Entities ────────────────────────────────────
            # Each row is a "noun" extracted from a document chunk:
            # a person, policy, department, date, etc.
            # The LLM reads each chunk during ingestion and pulls these out.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id              TEXT PRIMARY KEY,
                    name            TEXT        NOT NULL,
                    name_normalized TEXT        NOT NULL,
                    entity_type     TEXT        NOT NULL,
                    doc_id          TEXT        NOT NULL,
                    source          TEXT        NOT NULL,
                    folder          TEXT        NOT NULL,
                    chunk_id        TEXT        NOT NULL,
                    page            INTEGER     NOT NULL,
                    description     TEXT        NOT NULL DEFAULT '',
                    confidence      REAL        NOT NULL DEFAULT 0.7,
                    section         TEXT        NOT NULL DEFAULT '',
                    properties      JSONB       NOT NULL DEFAULT '{}',
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            # Add columns if they don't exist (for backwards compatibility)
            cur.execute("""
                ALTER TABLE entities
                ADD COLUMN IF NOT EXISTS confidence REAL NOT NULL DEFAULT 0.7;
            """)
            cur.execute("""
                ALTER TABLE entities
                ADD COLUMN IF NOT EXISTS section TEXT NOT NULL DEFAULT '';
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_entities_name_normalized ON entities(name_normalized);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_entities_doc_id ON entities(doc_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_entities_folder ON entities(folder);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_entities_confidence ON entities(confidence);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_entities_section ON entities(section);")

            # ── Knowledge Graph: Relationships ───────────────────────────────
            # Each row is a directed edge: entity A --[relation]--> entity B
            # e.g. "Leave Policy v3" --SUPERSEDES--> "Leave Policy v2"
            cur.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id                TEXT PRIMARY KEY,
                    source_entity_id  TEXT        NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    target_entity_id  TEXT        NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    relation_type     TEXT        NOT NULL,
                    description       TEXT        NOT NULL DEFAULT '',
                    confidence        REAL        NOT NULL DEFAULT 0.5,
                    doc_id            TEXT        NOT NULL,
                    chunk_id          TEXT        NOT NULL,
                    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(source_entity_id, target_entity_id, relation_type)
                );
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_entity_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_entity_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relation_type);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_doc_id ON relationships(doc_id);")

            # ── Knowledge Graph: Document Metadata ───────────────────────────
            # One row per ingested PDF — title, type, version, summary.
            # Supersession links connect versions of the same document.
            # ── Folders: tracks ingest type (research | policy | general) ──
            # Knowing whether a folder is a research folder lets the ingest
            # pipeline route documents through the study extractor.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS folders (
                    name        TEXT PRIMARY KEY,
                    path        TEXT        NOT NULL,
                    folder_type TEXT        NOT NULL DEFAULT 'general',
                    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            # ── Studies: one row per study (research paper extraction) ──
            # Study_id is the folder name when it looks like an ID, otherwise
            # synthesized from extracted title + year. JSONB arrays for multi-valued
            # research fields are indexed with GIN for fast queries.
            # Topic fields allow hierarchical organization (Primary/Sub/Micro).
            cur.execute("""
                CREATE TABLE IF NOT EXISTS studies (
                    study_id            TEXT PRIMARY KEY,
                    folder_path         TEXT        NOT NULL,
                    study_id_source     TEXT        NOT NULL DEFAULT 'folder_name',
                    title               TEXT,
                    authors             JSONB       NOT NULL DEFAULT '[]',
                    publication_year    INTEGER,
                    research_question   TEXT,
                    hypothesis          TEXT,
                    data_sources        JSONB       NOT NULL DEFAULT '[]',
                    conclusions         TEXT,
                    keywords            JSONB       NOT NULL DEFAULT '[]',
                    abstract            TEXT,
                    primary_topic       TEXT,
                    subtopic            TEXT,
                    microtopic          TEXT,
                    last_extracted_at   TIMESTAMPTZ,
                    source_doc_hash     TEXT,
                    extraction_model    TEXT        NOT NULL DEFAULT '',
                    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_studies_title ON studies(title);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_studies_publication_year ON studies(publication_year);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_studies_primary_topic ON studies(primary_topic);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_studies_subtopic ON studies(subtopic);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_studies_microtopic ON studies(microtopic);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_studies_authors ON studies USING GIN (authors);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_studies_keywords ON studies USING GIN (keywords);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_studies_data_sources ON studies USING GIN (data_sources);")

            # ── Study field provenance ───────────────────────────────────────
            # Every extracted field links back to its source document and page.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS study_field_provenance (
                    id              SERIAL PRIMARY KEY,
                    study_id        TEXT        NOT NULL REFERENCES studies(study_id) ON DELETE CASCADE,
                    field_name      TEXT        NOT NULL,
                    field_value     TEXT        NOT NULL,
                    source_file     TEXT        NOT NULL,
                    source_page     INTEGER,
                    confidence      REAL        NOT NULL DEFAULT 0.0,
                    extracted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_study_provenance_id ON study_field_provenance(study_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_study_provenance_field ON study_field_provenance(field_name);")

            # ── Patients: legacy table (data preservation only) ──────────────
            # Kept for backwards compatibility with existing patient data.
            # For new projects, use the studies table instead.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id          TEXT PRIMARY KEY,
                    folder_path         TEXT        NOT NULL,
                    patient_id_source   TEXT        NOT NULL DEFAULT 'folder_name',
                    full_name           TEXT,
                    date_of_birth       DATE,
                    gender              TEXT,
                    city                TEXT,
                    state               TEXT,
                    insurance_provider  TEXT,
                    insurance_id        TEXT,
                    icd10_codes         JSONB       NOT NULL DEFAULT '[]',
                    diagnoses           JSONB       NOT NULL DEFAULT '[]',
                    medications         JSONB       NOT NULL DEFAULT '[]',
                    medical_history     TEXT        NOT NULL DEFAULT '',
                    last_extracted_at   TIMESTAMPTZ,
                    source_doc_hash     TEXT,
                    extraction_model    TEXT        NOT NULL DEFAULT '',
                    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_patients_dob ON patients(date_of_birth);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_patients_state_city ON patients(state, city);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_patients_gender ON patients(gender);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_patients_insurance ON patients(insurance_provider);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_patients_icd10 ON patients USING GIN (icd10_codes);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_patients_diagnoses ON patients USING GIN (diagnoses);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_patients_medications ON patients USING GIN (medications);")

            # ── Patient field provenance (legacy) ─────────────────────────────
            # Legacy table for data preservation. For new projects, use
            # study_field_provenance instead.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS patient_field_provenance (
                    id              SERIAL PRIMARY KEY,
                    patient_id      TEXT        NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
                    field_name      TEXT        NOT NULL,
                    field_value     TEXT        NOT NULL,
                    source_file     TEXT        NOT NULL,
                    source_page     INTEGER,
                    confidence      REAL        NOT NULL DEFAULT 0.0,
                    extracted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_provenance_patient ON patient_field_provenance(patient_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_provenance_field ON patient_field_provenance(field_name);")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_metadata (
                    doc_id          TEXT PRIMARY KEY,
                    source          TEXT        NOT NULL,
                    folder          TEXT        NOT NULL,
                    title           TEXT        NOT NULL DEFAULT '',
                    doc_type        TEXT        NOT NULL DEFAULT '',
                    version         TEXT        NOT NULL DEFAULT '',
                    effective_date  TEXT        NOT NULL DEFAULT '',
                    superseded_by   TEXT        DEFAULT NULL,
                    supersedes      TEXT        DEFAULT NULL,
                    summary         TEXT        NOT NULL DEFAULT '',
                    entity_count    INTEGER     NOT NULL DEFAULT 0,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

    logger.info("Database schema ready")

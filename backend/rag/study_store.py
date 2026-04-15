"""
study_store.py — DB writes for the studies / provenance / folders tables.

Companion to study_extractor.py. The extractor produces a StudyRecord
in pure Python; this module persists it to PostgreSQL.

Idempotency contract:
  upsert_study() is safe to call repeatedly. If a study with the same
  study_id already exists, the row is updated and old provenance entries
  are replaced. The source_doc_hash on the record is what callers use to
  decide whether re-extraction is needed at all (skip if unchanged).
"""

import json
import logging
from dataclasses import asdict

from db.connection import db_conn
from rag.study_extractor import StudyRecord

logger = logging.getLogger(__name__)


# ── Folder type registry ──────────────────────────────────────────────────────


def register_folder(name: str, path: str, folder_type: str) -> None:
    """
    Record a folder's type. UPSERT — safe to call on every ingest.
    folder_type must be one of: 'research' | 'general'.
    """
    if folder_type not in {"research", "general", "policy"}:
        raise ValueError(f"invalid folder_type: {folder_type}")

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO folders (name, path, folder_type)
                VALUES (%s, %s, %s)
                ON CONFLICT (name) DO UPDATE
                  SET path = EXCLUDED.path,
                      folder_type = EXCLUDED.folder_type
                """,
                (name, path, folder_type),
            )


def get_folder_type(name: str) -> str | None:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT folder_type FROM folders WHERE name = %s", (name,))
            row = cur.fetchone()
            return row["folder_type"] if row else None


# ── Study upsert ──────────────────────────────────────────────────────────────


def get_existing_doc_hash(study_id: str) -> str | None:
    """Return the previously stored source_doc_hash, or None if not present."""
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT source_doc_hash FROM studies WHERE study_id = %s",
                (study_id,),
            )
            row = cur.fetchone()
            return row["source_doc_hash"] if row else None


def upsert_study(record: StudyRecord) -> None:
    """
    Insert or update a study row, then replace its provenance entries.
    """
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO studies (
                    study_id, folder_path, study_id_source,
                    title, authors, publication_year,
                    research_question, hypothesis, data_sources,
                    conclusions, keywords, abstract,
                    primary_topic, subtopic, microtopic,
                    last_extracted_at, source_doc_hash, extraction_model
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (study_id) DO UPDATE SET
                    folder_path        = EXCLUDED.folder_path,
                    study_id_source    = EXCLUDED.study_id_source,
                    title              = EXCLUDED.title,
                    authors            = EXCLUDED.authors,
                    publication_year   = EXCLUDED.publication_year,
                    research_question  = EXCLUDED.research_question,
                    hypothesis         = EXCLUDED.hypothesis,
                    data_sources       = EXCLUDED.data_sources,
                    conclusions        = EXCLUDED.conclusions,
                    keywords           = EXCLUDED.keywords,
                    abstract           = EXCLUDED.abstract,
                    primary_topic      = EXCLUDED.primary_topic,
                    subtopic           = EXCLUDED.subtopic,
                    microtopic         = EXCLUDED.microtopic,
                    last_extracted_at  = EXCLUDED.last_extracted_at,
                    source_doc_hash    = EXCLUDED.source_doc_hash,
                    extraction_model   = EXCLUDED.extraction_model
                """,
                (
                    record.study_id, record.folder_path, record.study_id_source,
                    record.title, json.dumps(record.authors), record.publication_year,
                    record.research_question, record.hypothesis, json.dumps(record.data_sources),
                    record.conclusions, json.dumps(record.keywords), record.abstract,
                    record.primary_topic, record.subtopic, record.microtopic,
                    record.last_extracted_at, record.source_doc_hash, record.extraction_model,
                ),
            )

            # Replace provenance for this study
            cur.execute(
                "DELETE FROM study_field_provenance WHERE study_id = %s",
                (record.study_id,),
            )
            for entry in record.provenance:
                cur.execute(
                    """
                    INSERT INTO study_field_provenance
                        (study_id, field_name, field_value, source_file, source_page, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        record.study_id,
                        entry.field_name,
                        entry.field_value,
                        entry.source_file,
                        entry.source_page,
                        entry.confidence,
                    ),
                )


# ── Read API ──────────────────────────────────────────────────────────────────


def get_study(study_id: str) -> dict | None:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM studies WHERE study_id = %s", (study_id,))
            row = cur.fetchone()
            if not row:
                return None
            study = dict(row)
            cur.execute(
                """
                SELECT field_name, field_value, source_file, source_page, confidence
                FROM study_field_provenance
                WHERE study_id = %s
                ORDER BY field_name
                """,
                (study_id,),
            )
            study["provenance"] = [dict(r) for r in cur.fetchall()]
            return study


def study_stats() -> dict:
    """High-level extraction quality stats for the validation API."""
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM studies")
            total = cur.fetchone()["n"]

            stats: dict = {"total_studies": total, "field_completeness": {}}
            if total == 0:
                return stats

            for field_name in (
                "title", "publication_year", "research_question", "hypothesis",
                "conclusions", "abstract", "primary_topic", "subtopic", "microtopic",
            ):
                cur.execute(
                    f"SELECT COUNT(*) AS n FROM studies "
                    f"WHERE {field_name} IS NOT NULL AND {field_name}::text <> ''"
                )
                stats["field_completeness"][field_name] = cur.fetchone()["n"]

            for field_name in ("authors", "data_sources", "keywords"):
                cur.execute(
                    f"SELECT COUNT(*) AS n FROM studies "
                    f"WHERE jsonb_array_length({field_name}) > 0"
                )
                stats["field_completeness"][field_name] = cur.fetchone()["n"]

            return stats

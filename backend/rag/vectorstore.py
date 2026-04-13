"""
vectorstore.py — pgvector-backed vector store

Responsibilities:
  - Store child chunk embeddings with full metadata in PostgreSQL
  - Query by vector similarity using pgvector cosine distance (HNSW index)
  - Support folder-scoped filtering per session
  - Handle deduplication via ON CONFLICT DO NOTHING

Replaces the previous ChromaDB implementation.
All data lives in the `chunks` table created by db/connection.py.
"""

import logging
from dataclasses import dataclass

from db.connection import db_conn
from rag.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single result returned from a similarity search."""
    chunk_id:    str
    source:      str        # filename
    folder:      str        # folder path
    page:        int        # page number
    child_text:  str        # small chunk — what was matched
    parent_text: str        # large chunk — what gets sent to the LLM
    score:       float      # similarity score 0.0–1.0 (higher = more relevant)
    was_ocr:     bool


class VectorStore:
    """
    Thin wrapper around pgvector queries.
    One instance shared across the whole backend application.
    Stateless — all persistent state lives in PostgreSQL.
    """

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """
        Store a batch of chunks with their pre-computed embeddings.
        ON CONFLICT DO NOTHING makes this idempotent — safe to re-run on the
        same folder without creating duplicates.
        """
        if not chunks:
            return

        rows = [
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.source,
                chunk.folder,
                chunk.page,
                chunk.text,
                chunk.parent_text,
                embedding,
                chunk.was_ocr,
                chunk.metadata.get("section", ""),
                chunk.metadata.get("doc_type", ""),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks
                      (id, doc_id, source, folder, page, content, parent_text,
                       embedding, was_ocr, section, doc_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    rows,
                )
        logger.info(f"Stored {len(chunks)} chunks (duplicates silently skipped)")

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document (used on re-ingestion)."""
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        logger.info(f"Deleted all chunks for doc_id={doc_id}")

    # ── Read ──────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        folders: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Find the top_k most semantically similar chunks to the query embedding.

        Uses pgvector's <=> cosine distance operator on the HNSW index.
        Score = 1 - cosine_distance, so 1.0 = identical, 0.0 = unrelated.

        folders: restrict search to those folders only. None = search everything.
        """
        # psycopg2 sends Python lists as numeric[], not vector.
        # Format as a pgvector string literal and cast with ::vector.
        vec_str = "[" + ",".join(map(str, query_embedding)) + "]"

        if folders:
            placeholders = ", ".join(["%s"] * len(folders))
            query = f"""
                SELECT
                    id,
                    source,
                    folder,
                    page,
                    content,
                    parent_text,
                    was_ocr,
                    1 - (embedding <=> %s::vector) AS score
                FROM chunks
                WHERE folder IN ({placeholders})
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            # vec_str appears twice: once in SELECT (score), once in ORDER BY
            params: list = [vec_str] + folders + [vec_str, top_k]
        else:
            query = """
                SELECT
                    id,
                    source,
                    folder,
                    page,
                    content,
                    parent_text,
                    was_ocr,
                    1 - (embedding <=> %s::vector) AS score
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params = [vec_str, vec_str, top_k]

        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        return [
            SearchResult(
                chunk_id=row["id"],
                source=row["source"],
                folder=row["folder"],
                page=row["page"],
                child_text=row["content"],
                parent_text=row["parent_text"],
                score=round(float(row["score"]), 4),
                was_ocr=row["was_ocr"],
            )
            for row in rows
        ]

    def keyword_search(
        self,
        keywords: list[str],
        top_k: int = 10,
        folders: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Exact keyword search using PostgreSQL full-text matching.

        Used as a third retrieval lane alongside semantic and BM25 search.
        Critical for short queries containing acronyms (HyDE, RAG, BM25) or
        proper nouns that embedding models may not recognise.

        Returns chunks where ALL keywords appear in the content, scored 0.5
        (a neutral score — re-ranking will determine final relevance).
        """
        if not keywords:
            return []

        # Build a WHERE clause that requires all keywords to appear
        keyword_clauses = " AND ".join(["content ILIKE %s"] * len(keywords))
        keyword_params = [f"%{kw}%" for kw in keywords]

        if folders:
            placeholders = ", ".join(["%s"] * len(folders))
            query = f"""
                SELECT id, source, folder, page, content, parent_text, was_ocr
                FROM chunks
                WHERE ({keyword_clauses})
                  AND folder IN ({placeholders})
                LIMIT %s
            """
            params: list = keyword_params + folders + [top_k]
        else:
            query = f"""
                SELECT id, source, folder, page, content, parent_text, was_ocr
                FROM chunks
                WHERE {keyword_clauses}
                LIMIT %s
            """
            params = keyword_params + [top_k]

        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        return [
            SearchResult(
                chunk_id=row["id"],
                source=row["source"],
                folder=row["folder"],
                page=row["page"],
                child_text=row["content"],
                parent_text=row["parent_text"],
                score=0.50,   # neutral — re-ranker will score properly
                was_ocr=row["was_ocr"],
            )
            for row in rows
        ]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[SearchResult]:
        """
        Load full SearchResult objects for a list of chunk IDs.

        Used by the graph retriever: graph traversal discovers chunk IDs
        that are related to the query, but the pipeline needs full
        SearchResult objects (text, source, page, etc.) to include them
        in the LLM context.

        Score is set to 0.0 — the graph retriever assigns its own scores
        based on graph distance.  These chunks compete on merit through
        re-ranking, not on a pre-assigned score.
        """
        if not chunk_ids:
            return []

        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, source, folder, page, content, parent_text, was_ocr
                    FROM chunks
                    WHERE id = ANY(%s)
                    """,
                    (chunk_ids,),
                )
                rows = cur.fetchall()

        return [
            SearchResult(
                chunk_id=row["id"],
                source=row["source"],
                folder=row["folder"],
                page=row["page"],
                child_text=row["content"],
                parent_text=row["parent_text"],
                score=0.0,
                was_ocr=row["was_ocr"],
            )
            for row in rows
        ]

    def doc_exists(self, doc_id: str) -> bool:
        """Check if any chunk from this document is already stored."""
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM chunks WHERE doc_id = %s LIMIT 1",
                    (doc_id,),
                )
                return cur.fetchone() is not None

    def count(self) -> int:
        """Total number of chunks stored across all documents."""
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS n FROM chunks")
                row = cur.fetchone()
                return row["n"] if row else 0

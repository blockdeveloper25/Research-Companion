"""
graph_store.py — CRUD and traversal queries for the knowledge graph

All graph data lives in PostgreSQL (no separate graph database).
Multi-hop traversal uses recursive CTEs — a SQL feature that lets
you "walk" relationships iteratively in a single query.

Design decisions:
  - ON CONFLICT DO NOTHING for idempotent writes (safe re-ingestion)
  - Recursive CTEs capped at depth 3, max 50 nodes (prevents runaway)
  - Bidirectional traversal — follows edges in both directions
  - Uses the shared connection pool from db/connection.py
"""

import logging
from dataclasses import asdict

from db.connection import db_conn
from config import GRAPH_TRAVERSAL_MAX_DEPTH, GRAPH_TRAVERSAL_MAX_NODES
from rag.graph_extractor import DocumentMetadata, Entity, Relationship

logger = logging.getLogger(__name__)


class GraphStore:
    """Read/write interface for the knowledge graph tables."""

    # ── Write operations ─────────────────────────────────────────────────

    def store_entities(self, entities: list[Entity]) -> int:
        """
        Bulk-insert entities. Returns count of NEW entities stored.

        ON CONFLICT DO NOTHING — if the entity already exists (same
        deterministic ID), it's silently skipped.  This makes
        re-ingestion safe without needing to delete first.
        """
        if not entities:
            return 0

        with db_conn() as conn:
            with conn.cursor() as cur:
                stored = 0
                for entity in entities:
                    cur.execute("""
                        INSERT INTO entities
                            (id, name, name_normalized, entity_type, doc_id,
                             source, folder, chunk_id, page, description, 
                             confidence, section, properties)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        ON CONFLICT (id) DO NOTHING
                    """, (
                        entity.id,
                        entity.name,
                        entity.name_normalized,
                        entity.entity_type,
                        entity.doc_id,
                        entity.source,
                        entity.folder,
                        entity.chunk_id,
                        entity.page,
                        entity.description,
                        entity.confidence,
                        entity.section,
                        _json_str(entity.properties),
                    ))
                    stored += cur.rowcount  # 1 if inserted, 0 if conflict

        return stored

    def store_relationships(self, relationships: list[Relationship]) -> int:
        """
        Bulk-insert relationships. Returns count of NEW relationships stored.

        UNIQUE constraint on (source, target, type) prevents duplicate
        edges.  Same ON CONFLICT DO NOTHING pattern as entities.
        """
        if not relationships:
            return 0

        with db_conn() as conn:
            with conn.cursor() as cur:
                stored = 0
                for rel in relationships:
                    cur.execute("""
                        INSERT INTO relationships
                            (id, source_entity_id, target_entity_id, relation_type,
                             description, confidence, doc_id, chunk_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (
                        rel.id,
                        rel.source_entity_id,
                        rel.target_entity_id,
                        rel.relation_type,
                        rel.description,
                        rel.confidence,
                        rel.doc_id,
                        rel.chunk_id,
                    ))
                    stored += cur.rowcount

        return stored

    def store_document_metadata(self, meta: DocumentMetadata) -> None:
        """Insert or update the metadata summary for a document."""
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO document_metadata
                        (doc_id, source, folder, title, doc_type, version,
                         effective_date, summary)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        title          = EXCLUDED.title,
                        doc_type       = EXCLUDED.doc_type,
                        version        = EXCLUDED.version,
                        effective_date = EXCLUDED.effective_date,
                        summary        = EXCLUDED.summary
                """, (
                    meta.doc_id, meta.source, meta.folder, meta.title,
                    meta.doc_type, meta.version, meta.effective_date,
                    meta.summary,
                ))

    def delete_by_doc_id(self, doc_id: str) -> None:
        """
        Remove all graph data for a document.

        Order matters: relationships reference entities via FK with
        ON DELETE CASCADE, so deleting entities auto-deletes their
        relationships.  We delete relationships first anyway for
        clarity and to handle any edge cases.
        """
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM relationships WHERE doc_id = %s", (doc_id,)
                )
                cur.execute(
                    "DELETE FROM entities WHERE doc_id = %s", (doc_id,)
                )
                cur.execute(
                    "DELETE FROM document_metadata WHERE doc_id = %s", (doc_id,)
                )
        logger.info(f"  Deleted graph data for doc_id={doc_id[:12]}...")

    # ── Read operations ──────────────────────────────────────────────────

    def find_entities_by_name(
        self,
        name: str,
        entity_type: str | None = None,
        folders: list[str] | None = None,
    ) -> list[dict]:
        """
        Search entities by name (case-insensitive, substring match).
        Optional filters by type and folder.
        """
        conditions = ["name_normalized ILIKE %s"]
        params: list = [f"%{name.lower().strip()}%"]

        if entity_type:
            conditions.append("entity_type = %s")
            params.append(entity_type)

        if folders:
            conditions.append("folder = ANY(%s)")
            params.append(folders)

        where = " AND ".join(conditions)

        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM entities WHERE {where} LIMIT 50", params
                )
                return cur.fetchall()

    def find_entities_by_type(
        self,
        entity_type: str,
        folders: list[str] | None = None,
    ) -> list[dict]:
        """List all entities of a given type, optionally scoped to folders."""
        if folders:
            query = "SELECT * FROM entities WHERE entity_type = %s AND folder = ANY(%s) LIMIT 100"
            params = [entity_type, folders]
        else:
            query = "SELECT * FROM entities WHERE entity_type = %s LIMIT 100"
            params = [entity_type]

        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()

    def get_related_entities(
        self,
        entity_id: str,
        max_depth: int | None = None,
        max_nodes: int | None = None,
    ) -> list[dict]:
        """
        Walk the graph from a starting entity using a recursive CTE.

        How it works:
          1. Base case: find all entities directly connected to the start
          2. Recursive step: for each discovered entity, find ITS connections
          3. Repeat up to max_depth times
          4. Collect all unique entities found

        This runs as ONE SQL query — the database handles the recursion
        internally, which is faster than multiple round-trips from Python.

        Bidirectional: follows edges in both directions (outgoing AND incoming)
        so "Policy A → APPLIES_TO → Department B" is discoverable starting
        from either Policy A or Department B.
        """
        depth = max_depth or GRAPH_TRAVERSAL_MAX_DEPTH
        limit = max_nodes or GRAPH_TRAVERSAL_MAX_NODES

        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    WITH RECURSIVE graph AS (
                        -- Base case: direct outgoing relationships
                        SELECT r.target_entity_id AS entity_id,
                               r.relation_type,
                               r.description AS rel_description,
                               1 AS depth
                        FROM relationships r
                        WHERE r.source_entity_id = %s

                        UNION

                        -- Base case: direct incoming relationships
                        SELECT r.source_entity_id AS entity_id,
                               r.relation_type,
                               r.description AS rel_description,
                               1 AS depth
                        FROM relationships r
                        WHERE r.target_entity_id = %s

                        UNION

                        -- Recursive step: follow outgoing edges from discovered nodes
                        SELECT r.target_entity_id,
                               r.relation_type,
                               r.description,
                               g.depth + 1
                        FROM relationships r
                        JOIN graph g ON r.source_entity_id = g.entity_id
                        WHERE g.depth < %s

                        UNION

                        -- Recursive step: follow incoming edges from discovered nodes
                        SELECT r.source_entity_id,
                               r.relation_type,
                               r.description,
                               g.depth + 1
                        FROM relationships r
                        JOIN graph g ON r.target_entity_id = g.entity_id
                        WHERE g.depth < %s
                    )
                    SELECT DISTINCT
                        e.*,
                        g.relation_type,
                        g.rel_description,
                        g.depth
                    FROM graph g
                    JOIN entities e ON e.id = g.entity_id
                    WHERE g.entity_id != %s
                    ORDER BY g.depth, e.name
                    LIMIT %s
                """, (entity_id, entity_id, depth, depth, entity_id, limit))

                return cur.fetchall()

    def get_entities_for_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Given retrieved chunks, find their associated entities."""
        if not chunk_ids:
            return []

        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM entities WHERE chunk_id = ANY(%s)",
                    (chunk_ids,)
                )
                return cur.fetchall()

    def find_related_chunk_ids(
        self,
        seed_chunk_ids: list[str],
        max_depth: int = 2,
        folders: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Bridge function: graph world → retrieval world.

        Given chunk_ids from vector/BM25 search:
          1. Find entities associated with those chunks
          2. Walk the graph to find related entities
          3. Return THEIR chunk_ids (with scores based on distance)

        This is how graph knowledge enhances retrieval — it discovers
        chunks that are *related* to the retrieved chunks, not just
        *similar* to the query.
        """
        if not seed_chunk_ids:
            return []

        # Step 1: Find entities in the seed chunks
        seed_entities = self.get_entities_for_chunk_ids(seed_chunk_ids)
        if not seed_entities:
            return []

        # Step 2: Walk the graph from each seed entity
        related_chunk_scores: dict[str, float] = {}

        for entity in seed_entities:
            related = self.get_related_entities(
                entity["id"], max_depth=max_depth
            )
            for rel in related:
                cid = rel["chunk_id"]
                # Skip chunks we already have
                if cid in seed_chunk_ids:
                    continue

                # Apply folder filter
                if folders and rel["folder"] not in folders:
                    continue

                # Score decays with distance: depth 1 → 1.0, depth 2 → 0.5, depth 3 → 0.33
                score = 1.0 / rel["depth"]
                # Keep the best score if this chunk was found via multiple paths
                related_chunk_scores[cid] = max(
                    related_chunk_scores.get(cid, 0), score
                )

        # Sort by score descending
        return sorted(related_chunk_scores.items(), key=lambda x: x[1], reverse=True)

    def get_document_metadata(self, doc_id: str) -> dict | None:
        """Get the metadata summary for a document."""
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM document_metadata WHERE doc_id = %s",
                    (doc_id,)
                )
                return cur.fetchone()

    # ── Stats ────────────────────────────────────────────────────────────

    def entity_count(self) -> int:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS count FROM entities")
                return cur.fetchone()["count"]

    def relationship_count(self) -> int:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS count FROM relationships")
                return cur.fetchone()["count"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _json_str(d: dict) -> str:
    """Convert a dict to a JSON string for PostgreSQL JSONB columns."""
    import json
    return json.dumps(d) if d else "{}"

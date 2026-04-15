"""
graph_retriever.py — Graph-augmented retrieval

Orchestrates entity extraction, graph traversal, and chunk loading
to discover document chunks that are RELATED to the query via the
knowledge graph — not just SIMILAR to it (which vector search does).

This is the key difference:
  Vector search:  "find chunks that mean something similar"
  Graph retrieval: "find chunks connected via entity relationships"

Example:
  Question: "Which policy supersedes the 2023 leave policy?"
  Vector search finds: chunks about leave policies (by meaning)
  Graph retrieval finds: the specific chunk that mentions the
    SUPERSEDES relationship between v2 and v3

Design:
  - Two-path entity discovery (question + seed chunks)
  - Graph context text injected into system prompt
  - Fail-safe: returns empty result on any failure
  - All graph chunks enter the RRF merge — they compete on merit
"""

import logging
from dataclasses import dataclass, field

from rag.graph_extractor import extract_entities_from_question, resolve_question_entities
from rag.graph_store import GraphStore
from rag.vectorstore import SearchResult, VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphRetrievalResult:
    """Output from graph-augmented retrieval."""
    graph_chunks:      list[SearchResult]  # chunks found via graph traversal
    graph_context:     str                 # relationship text for the system prompt
    is_graph_enhanced: bool                # whether graph found useful results
    entity_names:      tuple[str, ...]     # entity names for UI display


class GraphRetriever:
    """
    Discovers related chunks via the knowledge graph.

    Two-path entity discovery:
      Path 1 — Extract entities from the question → resolve against graph
      Path 2 — Find entities associated with seed chunks (from vector search)
    Both paths feed into graph traversal to discover related chunks.
    """

    def __init__(self, graph_store: GraphStore, vector_store: VectorStore):
        self._graph = graph_store
        self._vectors = vector_store

    def retrieve(
        self,
        question: str,
        seed_results: list[SearchResult],
        folders: list[str] | None = None,
    ) -> GraphRetrievalResult:
        """
        Find chunks related to the query via the knowledge graph.

        Args:
            question:      the user's raw question
            seed_results:  chunks already found by vector/BM25 search
            folders:       optional folder scope

        Returns:
            GraphRetrievalResult — chunks and context text to merge
            into the main retrieval pipeline
        """
        empty = GraphRetrievalResult(
            graph_chunks=[], graph_context="", is_graph_enhanced=False,
            entity_names=(),
        )

        try:
            return self._retrieve_inner(question, seed_results, folders)
        except Exception as e:
            logger.warning(f"Graph retrieval failed (non-critical): {e}")
            return empty

    def _retrieve_inner(
        self,
        question: str,
        seed_results: list[SearchResult],
        folders: list[str] | None,
    ) -> GraphRetrievalResult:
        """Core retrieval logic — separated for clean error handling."""

        # ── Path 1: Extract entities from the question ───────────────────
        # The LLM reads the question and identifies named entities.
        # Then we look up those names in the graph.
        question_entities_raw = extract_entities_from_question(question)
        question_entities = resolve_question_entities(
            question_entities_raw, self._graph, folders,
        )
        logger.info(
            f"  Graph: {len(question_entities_raw)} entities extracted from question, "
            f"{len(question_entities)} resolved in graph"
        )

        # ── Path 2: Find entities from seed chunks ───────────────────────
        # Chunks found by vector/BM25 search are associated with entities
        # in the graph.  This discovers connections the user didn't
        # explicitly mention.
        seed_chunk_ids = [r.chunk_id for r in seed_results]
        seed_entities = self._graph.get_entities_for_chunk_ids(seed_chunk_ids)
        logger.info(f"  Graph: {len(seed_entities)} entities from seed chunks")

        # Combine both paths, deduplicate by entity ID
        all_entity_ids: set[str] = set()
        all_entities: list[dict] = []

        for e in question_entities + seed_entities:
            eid = e["id"]
            if eid not in all_entity_ids:
                all_entity_ids.add(eid)
                all_entities.append(e)

        if not all_entities:
            return GraphRetrievalResult(
                graph_chunks=[], graph_context="", is_graph_enhanced=False,
                entity_names=(),
            )

        # ── Walk the graph from discovered entities ──────────────────────
        # find_related_chunk_ids handles traversal + scoring + dedup
        related = self._graph.find_related_chunk_ids(
            seed_chunk_ids=seed_chunk_ids,
            max_depth=2,
            folders=folders,
        )
        logger.info(f"  Graph: {len(related)} related chunks discovered")

        # ── Load full chunk data for discovered chunks ───────────────────
        if related:
            related_ids = [chunk_id for chunk_id, _ in related[:10]]  # cap at 10
            score_map = dict(related)
            graph_chunks = self._vectors.get_chunks_by_ids(related_ids)

            # Assign graph-based scores
            for chunk in graph_chunks:
                chunk.score = round(score_map.get(chunk.chunk_id, 0.3), 4)
        else:
            graph_chunks = []

        # ── Build context text for the system prompt ─────────────────────
        # This tells the LLM about entity relationships explicitly,
        # so it doesn't have to infer them from chunk text alone.
        graph_context = self._build_context(all_entities, folders)

        entity_names = tuple(e["name"] for e in all_entities[:10])

        return GraphRetrievalResult(
            graph_chunks=graph_chunks,
            graph_context=graph_context,
            is_graph_enhanced=bool(graph_chunks or graph_context),
            entity_names=entity_names,
        )

    def _build_context(
        self,
        entities: list[dict],
        folders: list[str] | None,
    ) -> str:
        """
        Format entity relationships as plain text for the system prompt.

        Why inject this into the prompt?
          The LLM sees chunk text but doesn't know the relationships
          between entities ACROSS chunks.  This context text makes
          cross-document relationships explicit:

          "Annual Leave Policy v3 SUPERSEDES Annual Leave Policy v2"

          Without this, the LLM would need to infer the supersession
          from two separate chunks that might not mention each other.
        """
        lines: list[str] = []

        for entity in entities[:5]:  # cap to avoid prompt bloat
            related = self._graph.get_related_entities(
                entity["id"], max_depth=1, max_nodes=10,
            )
            for rel in related:
                line = (
                    f"- {entity['name']} ({entity['entity_type']}) "
                    f"→ {rel['relation_type']} → "
                    f"{rel['name']} ({rel['entity_type']})"
                )
                lines.append(line)

        if not lines:
            return ""

        # Deduplicate (same relationship can appear from both ends)
        unique_lines = list(dict.fromkeys(lines))
        return "Entity relationships:\n" + "\n".join(unique_lines[:15])

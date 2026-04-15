"""
retriever.py — Hybrid search and Ollama-powered re-ranking

Two-stage retrieval:
  Stage 1 — Candidate retrieval (fast, broad)
    a. Semantic search via ChromaDB (meaning-based, top 20)
    b. BM25 keyword search (exact-term-based, top 20)
    c. Merge and deduplicate (up to 40 candidates)

  Stage 2 — Re-ranking (precise, llama3.2:3b)
    a. Score each candidate 1-10 for relevance to the question
    b. Keep top 5 by score
    c. Check top 2 for contradictions
    d. Apply score threshold — below 0.7 means no answer

Final output: up to 5 SearchResult objects ready for the LLM prompt.
"""

import logging
import math
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from models.ollama import rerank_chunks, detect_conflict
from rag.vectorstore import VectorStore, SearchResult

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

SEMANTIC_TOP_K    = 20     # candidates from vector search
BM25_TOP_K        = 20     # candidates from BM25
FINAL_TOP_K       = 5      # survivors after re-ranking
SCORE_THRESHOLD   = 0.45   # minimum re-rank score (normalised 0-1) to proceed
                           # 0.45 = 4.5/10 on the LLM scoring scale
                           # Chunks below this are not relevant enough to answer from


@dataclass
class RetrievalResult:
    """Final output from the retriever — ready for the LLM prompt."""
    results:           list[SearchResult]  # top-k chunks, re-ranked
    has_conflict:      bool                # did top 2 chunks contradict?
    best_score:        float               # highest re-rank score (0-1)
    below_threshold:   bool               # True = no relevant docs found
    keyword_chunk_ids: set[str] = None    # IDs of keyword-boosted chunks (diversity-exempt)


class Retriever:
    """
    Stateless retriever — create once, call retrieve() for each query.
    Holds a reference to the VectorStore for semantic search.
    BM25 index is built on-the-fly from the candidate pool (no pre-indexing needed).
    """

    def __init__(self, vector_store: VectorStore):
        self._store = vector_store

    def retrieve(
        self,
        question: str,
        query_vector: list[float],
        folders: list[str] | None = None,
    ) -> RetrievalResult:
        """
        Full retrieval pipeline for one question.

        Args:
            question:     raw user question (used for BM25 + re-ranking)
            query_vector: HyDE-expanded embedding (used for semantic search)
            folders:      optional folder scope from session settings

        Returns:
            RetrievalResult with up to FINAL_TOP_K chunks
        """

        # ── Stage 1a: Semantic search ─────────────────────────────────────────
        semantic_results = self._store.similarity_search(
            query_embedding=query_vector,
            top_k=SEMANTIC_TOP_K,
            folders=folders,
        )
        logger.info(f"Semantic search: {len(semantic_results)} candidates")

        # ── Stage 1b: BM25 keyword search (on semantic candidates) ────────────
        bm25_results = self._bm25_search(question, semantic_results)
        logger.info(f"BM25 re-ordered candidates")

        # ── Stage 1c: Keyword search (full corpus, exact match) ───────────────
        # Catches acronyms and proper nouns the embedding model may not recognise.
        keywords = _extract_keywords(question)
        if keywords:
            keyword_results = self._store.keyword_search(
                keywords=keywords,
                top_k=10,
                folders=folders,
            )
            logger.info(f"Keyword search: {len(keyword_results)} candidates for {keywords}")
        else:
            keyword_results = []

        # ── Stage 1d: Merge and deduplicate all three lanes ───────────────────
        candidates = _merge_results(semantic_results, bm25_results, keyword_results)
        logger.info(f"Merged pool: {len(candidates)} unique candidates")

        if not candidates:
            return RetrievalResult(
                results=[],
                has_conflict=False,
                best_score=0.0,
                below_threshold=True,
                keyword_chunk_ids=set(),
            )

        # ── Stage 2a: Re-rank with llama3.2:3b ───────────────────────────────
        chunk_texts = [r.child_text for r in candidates]
        raw_scores  = rerank_chunks(question, chunk_texts)

        # Normalise scores from 1-10 to 0-1
        normalised = [s / 10.0 for s in raw_scores]

        # Sort candidates by re-rank score descending
        ranked = sorted(
            zip(candidates, normalised),
            key=lambda x: x[1],
            reverse=True,
        )

        best_score = ranked[0][1] if ranked else 0.0
        logger.info(f"Best re-rank score: {best_score:.2f}")

        # ── Stage 2b: Apply threshold ─────────────────────────────────────────
        # If keyword search found exact matches, relax the threshold slightly —
        # the re-ranker (3b) may not recognise synonym gaps like "types" vs
        # "techniques" but the keyword lane confirms the document is relevant.
        keyword_ids = {r.chunk_id for r in keyword_results}
        effective_threshold = SCORE_THRESHOLD * 0.8 if keyword_ids else SCORE_THRESHOLD

        if best_score < effective_threshold:
            logger.info("Best score below threshold — no relevant documents")
            return RetrievalResult(
                results=[],
                has_conflict=False,
                best_score=best_score,
                below_threshold=True,
                keyword_chunk_ids=set(),
            )

        # ── Stage 2c: Keep top FINAL_TOP_K, but guarantee keyword matches ─────
        # The re-ranker may score generic chunks higher than specific keyword-
        # matched chunks (the 3b model doesn't know niche terms like HyDE).
        # Ensure at least one keyword-matched chunk is in the final context
        # if it scored above threshold, even if it fell below FINAL_TOP_K.
        top_results = [result for result, _ in ranked[:FINAL_TOP_K]]

        if keyword_ids:
            kw_in_top = any(r.chunk_id in keyword_ids for r in top_results)
            if not kw_in_top:
                # Find the best keyword-matched chunk that passed threshold
                for result, score in ranked[FINAL_TOP_K:]:
                    if result.chunk_id in keyword_ids and score >= SCORE_THRESHOLD:
                        # Swap out the last result (lowest re-rank score)
                        top_results[-1] = result
                        logger.info(
                            f"Keyword-boost: swapped in p{result.page} "
                            f"({result.source[:30]}) score={score:.2f}"
                        )
                        break

        # Update scores with re-rank values for display
        score_map = {result.chunk_id: score for result, score in ranked}
        for result in top_results:
            result.score = round(score_map.get(result.chunk_id, result.score), 4)

        # ── Stage 2d: Conflict detection on top 2 ────────────────────────────
        has_conflict = False
        if len(top_results) >= 2:
            has_conflict = detect_conflict(
                top_results[0].child_text,
                top_results[1].child_text,
            )
            if has_conflict:
                logger.warning("Conflict detected between top 2 chunks")

        # Track which chunks were keyword-boosted — pipeline uses this to exempt
        # them from the per-source diversity cap in _diversify_sources
        boosted_ids = {r.chunk_id for r in top_results if r.chunk_id in keyword_ids}

        return RetrievalResult(
            results=top_results,
            has_conflict=has_conflict,
            best_score=best_score,
            below_threshold=False,
            keyword_chunk_ids=boosted_ids,
        )


    # ── BM25 ──────────────────────────────────────────────────────────────────

    def _bm25_search(
        self,
        question: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Re-order candidates using BM25 keyword scoring.

        BM25 (Best Match 25) is the algorithm behind most search engines.
        It scores documents by how often query terms appear, weighted by
        how rare those terms are across all documents.

        We run BM25 over the semantic candidates (not the full corpus)
        to avoid pre-indexing thousands of chunks — keeps things simple
        and still gives us the keyword boost we need.
        """
        if not candidates:
            return []

        # Tokenise each candidate's child text
        tokenised_corpus = [_tokenise(r.child_text) for r in candidates]
        tokenised_query  = _tokenise(question)

        bm25 = BM25Okapi(tokenised_corpus)
        scores = bm25.get_scores(tokenised_query)

        # Sort candidates by BM25 score descending
        bm25_ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [result for result, _ in bm25_ranked]


# ── Merge helpers ─────────────────────────────────────────────────────────────

def _merge_results(
    semantic: list[SearchResult],
    bm25: list[SearchResult],
    keyword: list[SearchResult] | None = None,
) -> list[SearchResult]:
    """
    Combine semantic, BM25, and keyword results using Reciprocal Rank Fusion (RRF).

    RRF score(chunk) = sum of 1/(rank_in_list + 60) across all lists it appears in.
    Chunks appearing high in multiple lists score highest.
    Chunks appearing in only one list (e.g. keyword-only) still get credit.
    The constant 60 dampens rank dominance.
    """
    rrf_scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}

    for rank, result in enumerate(semantic):
        rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + _rrf(rank)
        result_map[result.chunk_id] = result

    for rank, result in enumerate(bm25):
        rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + _rrf(rank)
        result_map[result.chunk_id] = result

    for rank, result in enumerate(keyword or []):
        rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + _rrf(rank)
        result_map[result.chunk_id] = result

    sorted_ids = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
    return [result_map[chunk_id] for chunk_id in sorted_ids]


def _rrf(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a given rank position."""
    return 1.0 / (rank + k)


def _extract_keywords(question: str) -> list[str]:
    """
    Extract distinctive keywords from a question for exact-match search.

    Targets:
      - Acronyms (ALL_CAPS or MixedCase like HyDE, BM25, RAG, HNSW)
      - Short technical tokens that embedding models may not recognise well

    Common question words (what, is, how, the, a, an, ...) are excluded.
    Returns at most 2 keywords to keep the SQL ILIKE query precise.
    """
    import re
    _STOP_WORDS = {
        "what", "is", "are", "how", "does", "do", "the", "a", "an",
        "in", "of", "for", "to", "and", "or", "it", "this", "that",
        "why", "when", "where", "which", "who", "can", "could", "would",
        "should", "will", "explain", "describe", "tell", "me", "please",
        "difference", "between", "vs", "versus",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]*", question)
    keywords = []
    for tok in tokens:
        if tok.lower() in _STOP_WORDS:
            continue
        # Keep: acronyms (≥2 uppercase letters), camelCase/MixedCase, or length ≥6
        is_acronym = re.match(r"[A-Z]{2,}", tok)
        is_mixed = re.match(r"[A-Z][a-z]+[A-Z]", tok)  # e.g. HyDE
        if is_acronym or is_mixed or len(tok) >= 6:
            keywords.append(tok)
    return keywords[:2]


def _tokenise(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokeniser for BM25.
    Lowercases and splits on non-alphanumeric characters.
    """
    import re
    return re.findall(r"[a-z0-9]+", text.lower())

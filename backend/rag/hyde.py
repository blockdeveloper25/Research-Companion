"""
hyde.py — Hypothetical Document Embedding (HyDE) query expansion

Problem:
  User questions and document chunks live in different parts of vector space.
  "How many holidays do I get?" is far from "Employees are entitled to 20 days
  of annual leave" even though they mean the same thing.

Solution:
  Before searching, ask llama3.2:3b to write a short document excerpt that
  would answer the question. Search with that hypothetical answer instead.
  Hypothetical answers sound like real document chunks — so they match better.

Pipeline:
  raw question
      → llama3.2:3b generates hypothetical answer
      → embed hypothetical answer with nomic-embed-text
      → use that vector for ChromaDB search
"""

import logging

from models.ollama import embed_single, generate_hyde

logger = logging.getLogger(__name__)


def expand_query(question: str) -> tuple[str, list[float]]:
    """
    Takes a raw user question and returns:
      - hypothetical_answer: the generated document-style excerpt (for logging/debug)
      - query_vector:        a blended vector (70% HyDE + 30% original question)

    Blending ensures that even when HyDE produces a poor expansion (e.g. the model
    doesn't recognise the topic), the original question still anchors the search.
    """
    hypothetical_answer = _generate_hypothetical(question)

    hyde_vector = _embed_query(hypothetical_answer)
    orig_vector = _embed_query(question)

    # Blend: HyDE carries most weight when it works, original anchors when it doesn't
    query_vector = _blend_vectors(hyde_vector, orig_vector, hyde_weight=0.7)

    return hypothetical_answer, query_vector


def expand_query_multi(question: str, n: int = 3) -> list[float]:
    """
    Advanced variant: generate N hypothetical answers, embed each,
    then average the vectors. A single HyDE vector captures one
    interpretation of the question. Averaging N captures more angles.

    Example:
      "What happens if I'm sick on a public holiday?"
      → HyDE 1: focuses on sick leave rules
      → HyDE 2: focuses on public holiday policy
      → HyDE 3: focuses on overlap between the two
      → average vector: captures all three interpretations

    Use this for ambiguous or multi-part questions.
    Only called when the router classifies a query as COMPLEX.
    """
    vectors = []

    for i in range(n):
        hypothetical = _generate_hypothetical(question, variant=i)
        vector = _embed_query(hypothetical)
        vectors.append(vector)

    if not vectors:
        # Fallback: embed original question
        return _embed_query(question)

    # Average the vectors element-by-element
    averaged = _average_vectors(vectors)
    logger.info(f"Multi-HyDE: averaged {len(vectors)} vectors for query")
    return averaged


# ── Internal helpers ──────────────────────────────────────────────────────────


# Phrases that indicate the model confused the HyDE task with something else.
# When these appear, the generated text is useless as a search vector.
_META_RESPONSE_SIGNALS = (
    "i don't see",
    "i'm happy to help",
    "please provide",
    "could you provide",
    "i need more information",
    "i cannot see",
    "no excerpt",
    "no document",
)


def _generate_hypothetical(question: str, variant: int = 0) -> str:
    """
    Ask llama3.2:3b to write a document-style excerpt that answers the question.
    variant > 0 nudges the model to approach from a different angle.
    Falls back to the original question if:
      - generation raises an exception
      - the model returned a meta-response (confused about the task)
      - the response is too short to be meaningful
    """
    try:
        hypothetical = generate_hyde(question)

        if not hypothetical or len(hypothetical.strip()) < 20:
            logger.warning("HyDE returned empty response, falling back to question")
            return question

        # Detect when the model didn't understand the task and returned a
        # meta-response instead of a hypothetical document excerpt
        lower = hypothetical.lower()
        if any(signal in lower for signal in _META_RESPONSE_SIGNALS):
            logger.warning(f"HyDE returned meta-response, falling back to question")
            return question

        logger.debug(f"HyDE generated: {hypothetical[:100]}...")
        return hypothetical

    except Exception as e:
        logger.warning(f"HyDE generation failed ({e}), falling back to question")
        return question


def _embed_query(text: str) -> list[float]:
    """
    Embed a text string using nomic-embed-text.
    Returns a 768-dimensional vector.
    Falls back gracefully on error.
    """
    try:
        return embed_single(text)
    except Exception as e:
        logger.error(f"Query embedding failed: {e}")
        raise


def _blend_vectors(
    a: list[float],
    b: list[float],
    hyde_weight: float = 0.7,
) -> list[float]:
    """
    Weighted blend of two equal-dimension vectors.
    Default: 70% HyDE vector + 30% original question vector.
    """
    w_b = 1.0 - hyde_weight
    return [hyde_weight * x + w_b * y for x, y in zip(a, b)]


def _average_vectors(vectors: list[list[float]]) -> list[float]:
    """
    Element-wise average of multiple vectors.
    All vectors must be the same dimension (768 for nomic-embed-text).

    Example with 2 vectors of dim 3:
      [1.0, 2.0, 3.0]
      [3.0, 4.0, 5.0]
      → average: [2.0, 3.0, 4.0]
    """
    if not vectors:
        raise ValueError("Cannot average empty list of vectors")

    dim = len(vectors[0])
    averaged = [0.0] * dim

    for vec in vectors:
        for i, val in enumerate(vec):
            averaged[i] += val

    n = len(vectors)
    return [v / n for v in averaged]

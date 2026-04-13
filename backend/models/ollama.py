"""
ollama.py — Single interface to all Ollama models

All communication with Ollama goes through this file.
Three models, each with a defined role:

  nomic-embed-text  →  convert text to vectors (embeddings)
  llama3.2:3b       →  fast worker: HyDE, routing, re-ranking, confidence
  llama3.2:8b       →  deep reasoner: complex query answers only

Design principles:
  - Every call has a timeout so nothing hangs forever
  - Streaming is handled via a generator (yields tokens one by one)
  - Temperature=0 on all generation calls (no creativity, maximum accuracy)
  - Single retry on transient errors before raising
"""

import json
import logging
from collections.abc import Generator
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL  = "http://localhost:11434"
EMBED_MODEL      = "nomic-embed-text"
WORKER_MODEL     = "llama3.2:3b"
REASONER_MODEL   = "llama3.1:8b"

# Generation options applied to every LLM call
GENERATION_OPTIONS = {
    "temperature":    0,      # deterministic — no creativity, no hallucination risk
    "top_p":          1,      # no nucleus sampling
    "repeat_penalty": 1.1,    # mild penalty to avoid repetitive filler
}

EMBED_TIMEOUT    = 60    # seconds — embedding is fast
GENERATE_TIMEOUT = 120   # seconds — generation can take longer
STREAM_TIMEOUT   = 300   # seconds — streaming responses can be slow for complex queries


# ── Embeddings ────────────────────────────────────────────────────────────────

def embed(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into 768-dim embedding vectors.
    Used for both document ingestion and query embedding.

    Returns one vector per input text, in the same order.
    """
    if not texts:
        return []

    try:
        response = _post(
            "/api/embed",
            {"model": EMBED_MODEL, "input": texts},
            timeout=EMBED_TIMEOUT,
        )
        return response["embeddings"]
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise


def embed_single(text: str) -> list[float]:
    """Convenience wrapper — embed a single string."""
    return embed([text])[0]


# ── Fast worker — llama3.2:3b ─────────────────────────────────────────────────

def generate_worker(prompt: str) -> str:
    """
    Run a single prompt through llama3.2:3b and return the full response.
    Used for: HyDE, routing, re-ranking scores, conflict detection,
              confidence scoring, metadata extraction, simple answers.
    """
    return _generate(WORKER_MODEL, prompt)


def stream_worker(
    system_prompt: str,
    messages: list[dict],
) -> Generator[str, None, None]:
    """
    Stream a chat response from llama3.2:3b token by token.
    Used for simple query answers where speed matters most.
    Yields each token as it arrives.
    """
    yield from _stream_chat(WORKER_MODEL, system_prompt, messages)


# ── Deep reasoner — llama3.2:8b ──────────────────────────────────────────────

def stream_reasoner(
    system_prompt: str,
    messages: list[dict],
) -> Generator[str, None, None]:
    """
    Stream a chat response from llama3.2:8b token by token.
    Used only for complex queries requiring multi-document reasoning.
    Yields each token as it arrives.
    """
    yield from _stream_chat(REASONER_MODEL, system_prompt, messages)


# ── Structured helpers ────────────────────────────────────────────────────────

def classify_complexity(question: str) -> str:
    """
    Ask llama3.2:3b to classify a question as SIMPLE or COMPLEX.

    SIMPLE  → single fact, one document, direct answer
    COMPLEX → comparison, multi-document, reasoning required

    Returns "SIMPLE" or "COMPLEX".
    """
    prompt = (
        "Classify this question as SIMPLE or COMPLEX.\n"
        "SIMPLE: a single fact lookup with a direct answer from one document.\n"
        "COMPLEX: requires comparison, reasoning across multiple documents, "
        "or a step-by-step explanation.\n\n"
        f"Question: {question}\n\n"
        "Reply with one word only — SIMPLE or COMPLEX."
    )
    result = generate_worker(prompt).strip().upper()
    # Guard against unexpected model output
    return "COMPLEX" if "COMPLEX" in result else "SIMPLE"


def classify_graph_relevance(question: str) -> bool:
    """
    Should this question use the knowledge graph?

    Returns True for relational/comparative questions:
      - "Which policy supersedes X?"        → YES (relationship traversal)
      - "What department does X apply to?"  → YES (entity connection)
      - "Compare old and new versions"      → YES (document comparison)

    Returns False for simple fact lookups:
      - "What is the notice period?"        → NO (single chunk answer)
      - "How do I submit expenses?"         → NO (procedural lookup)

    Why an LLM call instead of keyword heuristics?
      Heuristics catch "compare" and "supersede" but miss questions like
      "What department is responsible for data protection?" which is
      clearly relational but uses no obvious keywords.
    """
    prompt = (
        "Does this question ask about relationships between entities, "
        "comparisons between documents, policy versions, who authored or "
        "approved something, which department something applies to, or "
        "what supersedes/replaces what?\n\n"
        f"Question: {question}\n\n"
        "Reply with one word only — YES or NO."
    )
    try:
        result = generate_worker(prompt).strip().upper()
        return "YES" in result
    except Exception:
        return False  # safe default — skip graph, use normal retrieval


def generate_hyde(question: str) -> str:
    """
    Generate a Hypothetical Document Excerpt (HyDE) for a question.

    Instead of searching with the raw question, we search with a
    hypothetical answer — this produces much better vector matches
    because the hypothetical answer is in the same style as real chunks.
    """
    prompt = (
        "Write a short document excerpt (2-3 sentences) that would directly "
        "answer the following question. Be factual and concise. "
        "Write as if you are quoting from an official policy or manual.\n\n"
        f"Question: {question}\n\n"
        "Excerpt:"
    )
    return generate_worker(prompt).strip()


def rerank_chunks(question: str, chunks: list[str]) -> list[float]:
    """
    Ask llama3.2:3b to score how relevant each chunk is to the question.
    Returns a list of float scores (1.0 to 10.0) in the same order as chunks.

    We call the model once per chunk (fast at 3b size) to get precise scores.
    """
    scores = []
    for chunk in chunks:
        prompt = (
            "You are a relevance judge. Score how well this text answers the question.\n\n"
            "Scoring guide:\n"
            "9-10 = directly and completely answers the question\n"
            "7-8  = contains key facts that help answer the question\n"
            "5-6  = loosely related but misses the main point\n"
            "3-4  = same topic area but does not address the question\n"
            "1-2  = unrelated or garbled/unreadable text\n\n"
            "IMPORTANT: Give a 1 if the text contains mostly symbols, percent signs, "
            "or non-English characters — it is corrupted and useless.\n\n"
            f"Question: {question}\n\n"
            f"Text: {chunk[:600]}\n\n"
            "Reply with a single integer (1-10) and nothing else.\n"
            "Score:"
        )
        try:
            raw = generate_worker(prompt).strip()
            # Extract first number from response
            import re
            match = re.search(r"\d+\.?\d*", raw)
            score = float(match.group()) if match else 1.0
            scores.append(min(max(score, 1.0), 10.0))  # clamp to 1-10
        except Exception:
            scores.append(1.0)  # default low score on failure

    return scores


def check_confidence(answer: str, context_chunks: list[str]) -> dict:
    """
    After an answer is generated, ask llama3.2:3b to rate how well
    the answer is grounded in the source chunks.

    Returns: {"level": "HIGH"|"MEDIUM"|"LOW", "reason": "..."}
    """
    context_preview = "\n---\n".join(c[:400] for c in context_chunks[:3])
    prompt = (
        "Given this answer and the source passages it was based on, "
        "rate the confidence that the answer is fully supported by the sources.\n\n"
        f"Answer: {answer}\n\n"
        f"Sources:\n{context_preview}\n\n"
        "Reply in this exact format:\n"
        "CONFIDENCE: HIGH\n"
        "REASON: one sentence\n\n"
        "Use HIGH if the answer is clearly supported, "
        "MEDIUM if partially supported, LOW if weakly supported."
    )
    try:
        raw = generate_worker(prompt)
        import re
        level_match  = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", raw, re.IGNORECASE)
        reason_match = re.search(r"REASON:\s*(.+)", raw, re.IGNORECASE)
        return {
            "level":  level_match.group(1).upper() if level_match else "MEDIUM",
            "reason": reason_match.group(1).strip() if reason_match else "",
        }
    except Exception:
        return {"level": "MEDIUM", "reason": ""}


def detect_conflict(chunk_a: str, chunk_b: str) -> bool:
    """
    Ask llama3.2:3b whether two chunks contradict each other.
    Used before building the prompt — conflicting sources are flagged.
    """
    prompt = (
        "Do these two passages contradict each other?\n"
        "Reply YES if they clearly disagree, NO otherwise.\n\n"
        f"Passage 1: {chunk_a[:400]}\n\n"
        f"Passage 2: {chunk_b[:400]}\n\n"
        "Answer (YES or NO):"
    )
    result = generate_worker(prompt).strip().upper()
    return result.startswith("YES")


# ── Internal HTTP helpers ─────────────────────────────────────────────────────

def _generate(model: str, prompt: str) -> str:
    """
    Non-streaming generation — waits for the full response.
    Used for short structured tasks where we need the complete output.
    """
    response = _post(
        "/api/generate",
        {
            "model":   model,
            "prompt":  prompt,
            "stream":  False,
            "options": GENERATION_OPTIONS,
        },
        timeout=GENERATE_TIMEOUT,
    )
    return response.get("response", "")


def _stream_chat(
    model: str,
    system_prompt: str,
    messages: list[dict],
) -> Generator[str, None, None]:
    """
    Streaming chat — yields tokens one by one as Ollama generates them.
    messages format: [{"role": "user"|"assistant", "content": "..."}]
    """
    payload = {
        "model":    model,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "stream":   True,
        "options":  GENERATION_OPTIONS,
    }

    try:
        with httpx.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=STREAM_TIMEOUT,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    except httpx.TimeoutException:
        logger.error(f"Stream timeout after {STREAM_TIMEOUT}s for model {model}")
        yield "\n\n[Response timed out. Please try again.]"
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield "\n\n[An error occurred while generating the response.]"


def _post(endpoint: str, payload: dict, timeout: int) -> dict[str, Any]:
    """
    Single POST call to Ollama with one retry on transient failure.
    """
    url = f"{OLLAMA_BASE_URL}{endpoint}"
    for attempt in range(2):  # try twice
        try:
            response = httpx.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error {e.response.status_code}: {e}")
            raise
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            if attempt == 0:
                logger.warning(f"Ollama call failed (attempt 1), retrying: {e}")
                continue
            logger.error(f"Ollama unreachable after 2 attempts: {e}")
            raise
    return {}

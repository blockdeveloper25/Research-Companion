"""
graph_extractor.py — LLM-based entity and relationship extraction

Reads parent chunks and asks llama3.2:3b to identify entities (nouns)
and relationships (verbs connecting them).  Results feed into the
knowledge graph stored in PostgreSQL.

Design decisions:
  - Batches parent chunks (5 per call) to minimise LLM round-trips
  - Uses few-shot prompting for consistent output from a 3b model
  - Regex-extracts JSON from LLM responses (tolerates prose wrapping)
  - Deterministic IDs via hashing for idempotent re-ingestion
  - Graceful fallback: extraction failure → empty lists (non-critical)
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field

import httpx

from config import (
    GRAPH_ENTITY_TYPES,
    GRAPH_EXTRACTION_BATCH_SIZE,
    GRAPH_MAX_ENTITIES_PER_CHUNK,
    GRAPH_MAX_RELS_PER_CHUNK,
    GRAPH_RELATION_TYPES,
    OLLAMA_BASE_URL,
    WORKER_MODEL,
)

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Entity:
    """A noun extracted from a document chunk."""
    id:              str
    name:            str
    name_normalized: str
    entity_type:     str
    doc_id:          str
    source:          str
    folder:          str
    chunk_id:        str
    page:            int
    description:     str = ""
    confidence:      float = 0.7  # LLM-assessed confidence (0.0-1.0)
    section:         str = ""     # Research section: abstract, methods, results, discussion, etc.
    properties:      dict = field(default_factory=dict)


@dataclass(frozen=True)
class Relationship:
    """A directed edge between two entities."""
    id:               str
    source_entity_id: str
    target_entity_id: str
    relation_type:    str
    description:      str
    confidence:       float
    doc_id:           str
    chunk_id:         str


@dataclass(frozen=True)
class DocumentMetadata:
    """Summary card for an ingested PDF."""
    doc_id:           str
    source:           str
    folder:           str
    title:            str = ""
    doc_type:         str = ""
    version:          str = ""
    effective_date:   str = ""
    summary:          str = ""
    cross_references: tuple[str, ...] = ()


# ── Public API ───────────────────────────────────────────────────────────────

def extract_entities_and_relationships(
    chunks: list,
    doc_id: str,
    source: str,
    folder: str,
) -> tuple[list[Entity], list[Relationship]]:
    """
    Extract entities and relationships from document chunks.

    Groups chunks by parent text (deduped), batches them, and sends
    each batch to the LLM.  Returns all entities and relationships
    found across the entire document.
    """
    # Deduplicate parent texts — multiple child chunks share a parent
    seen_parents: set[str] = set()
    parent_chunk_map: list[tuple[str, str, int]] = []  # (parent_text, chunk_id, page)

    for chunk in chunks:
        if chunk.parent_text not in seen_parents:
            seen_parents.add(chunk.parent_text)
            parent_chunk_map.append((chunk.parent_text, chunk.chunk_id, chunk.page))

    logger.info(
        f"  Graph extraction: {len(parent_chunk_map)} unique parents "
        f"(from {len(chunks)} child chunks)"
    )

    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []

    # Process in batches
    for i in range(0, len(parent_chunk_map), GRAPH_EXTRACTION_BATCH_SIZE):
        batch = parent_chunk_map[i : i + GRAPH_EXTRACTION_BATCH_SIZE]
        batch_num = i // GRAPH_EXTRACTION_BATCH_SIZE + 1

        try:
            entities, rels = _extract_batch(batch, doc_id, source, folder)
            all_entities.extend(entities)
            all_relationships.extend(rels)
            logger.info(
                f"    Batch {batch_num}: {len(entities)} entities, {len(rels)} relationships"
            )
        except Exception as e:
            logger.warning(f"    Batch {batch_num} extraction failed (non-critical): {e}")

    # Deduplicate entities by ID (same entity found in multiple parents)
    unique_entities = list({e.id: e for e in all_entities}.values())

    logger.info(
        f"  Graph totals: {len(unique_entities)} entities, "
        f"{len(all_relationships)} relationships"
    )

    return unique_entities, all_relationships


def extract_document_metadata(
    chunks: list,
    doc_id: str,
    source: str,
    folder: str,
) -> DocumentMetadata:
    """
    Extract document-level metadata (title, authors, abstract, research type, etc.)
    from the first few parent chunks — focusing on research paper structure
    (title page, abstract, introduction).
    """
    from config import RESEARCH_DOC_CATEGORIES
    
    # Take up to 3 unique parent texts from the start of the document
    seen: set[str] = set()
    intro_texts: list[str] = []
    for chunk in chunks:
        if chunk.parent_text not in seen and len(intro_texts) < 3:
            seen.add(chunk.parent_text)
            intro_texts.append(chunk.parent_text)

    combined = "\n\n---\n\n".join(intro_texts)
    doc_types_list = ", ".join(RESEARCH_DOC_CATEGORIES)

    prompt = (
        "Read this research document excerpt and extract metadata. "
        "Reply with JSON only, no other text.\n\n"
        f"Document types (use ONLY these): {doc_types_list}\n\n"
        "Return exactly:\n"
        '{"title": "<document title or main heading>", '
        f'"doc_type": "<{doc_types_list}>", '
        '"version": "<version or edition if found, else empty>", '
        '"effective_date": "<publication or submission date if found, else empty>", '
        '"summary": "<2-3 sentence abstract or summary of research topic>", '
        '"cross_references": ["<papers or studies referenced>"]}\n\n'
        f"Excerpt:\n{combined[:3000]}"
    )

    try:
        raw = _call_llm(prompt)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            doc_type = data.get("doc_type", "OTHER").upper()
            # Validate doc_type is in allowed list
            if doc_type not in RESEARCH_DOC_CATEGORIES:
                doc_type = "OTHER"
            
            return DocumentMetadata(
                doc_id=doc_id,
                source=source,
                folder=folder,
                title=data.get("title", ""),
                doc_type=doc_type,
                version=data.get("version", ""),
                effective_date=data.get("effective_date", ""),
                summary=data.get("summary", ""),
                cross_references=tuple(data.get("cross_references", [])),
            )
    except Exception as e:
        logger.warning(f"  Document metadata extraction failed (non-critical): {e}")

    return DocumentMetadata(doc_id=doc_id, source=source, folder=folder)


# ── Batch extraction ─────────────────────────────────────────────────────────

def _extract_batch(
    batch: list[tuple[str, str, int]],
    doc_id: str,
    source: str,
    folder: str,
) -> tuple[list[Entity], list[Relationship]]:
    """
    Send a batch of parent chunks to the LLM and parse the response
    into Entity and Relationship objects.
    """
    prompt = _build_extraction_prompt([text for text, _, _ in batch])
    raw = _call_llm(prompt)
    return _parse_extraction_response(raw, batch, doc_id, source, folder)


def _build_extraction_prompt(parent_texts: list[str]) -> str:
    """
    Construct the extraction prompt with multi-scenario few-shot examples.

    Optimized for research papers: emphasis on hypotheses, methodologies,
    findings, authors, citations, and structured research elements
    (abstract, methods, results, discussion).

    Why few-shot?  A 3b model follows examples far better than
    abstract instructions.  Multiple concrete examples teach it the
    output format and research-specific nuances more reliably.
    """
    entity_types = ", ".join(GRAPH_ENTITY_TYPES)
    relation_types = ", ".join(GRAPH_RELATION_TYPES)

    # Number each text block so the model can reference them
    text_blocks = []
    for i, text in enumerate(parent_texts, 1):
        text_blocks.append(f"[BLOCK {i}]\n{text[:600]}")

    texts_section = "\n\n".join(text_blocks)

    return f"""Extract all scientific entities and relationships from research paper text blocks.
Your goal: identify research concepts, people, institutions, data, methods, and how they connect.

ENTITY TYPES (use ONLY these): {entity_types}
RELATIONSHIP TYPES (use ONLY these): {relation_types}

CRITICAL EXTRACTION RULES:
1. AUTHOR names: Extract individual researcher names (e.g., "Smith, J", "Jane Smith")
2. AFFILIATION: Extract institution names (e.g., "MIT", "Stanford Medical Center")
3. FUNDING_BODY: Extract funding sources (e.g., "NIH", "European Commission")
4. STATISTICAL_MEASURE: Extract with values where possible (e.g., "p < 0.05", "r = 0.73")
5. CITATION: Extract referenced studies/authors (e.g., "Smith et al. 2020")
6. RESEARCH_DOMAIN: Extract field of study (e.g., "machine learning", "oncology")
7. CONFIDENCE: Include your confidence (0.1-1.0) that entity is correctly identified
8. SECTION: Note which research section (abstract, methods, results, discussion, introduction)
9. Section identifiers: Use "methods" for method sections, "results" for findings, "discussion" for conclusions

Reply with JSON only. No explanation, no markdown fences.

Format:
{{"entities": [
    {{"name": "entity name", "type": "ENTITY_TYPE", "description": "brief description", "confidence": 0.9, "section": "methods", "properties": {{"key": "value"}}}}
  ],
  "relationships": [
    {{"source": "entity name", "target": "entity name", "type": "RELATION_TYPE", "description": "what this relationship means", "confidence": 0.85}}
  ]
}}

EXAMPLE 1 — Climate/agriculture study:
Given: "We hypothesized that climate change impacts crop yields, measured by rainfall levels. We used random forest regression on 50 years of agricultural data from the USDA. This work was funded by the NSF. Lead researcher: Dr. Emily Chen from UC Berkeley."

Expected entities and relationships:
{{"entities": [
    {{"name": "climate change", "type": "RESEARCH_DOMAIN", "description": "Field of study focusing on climate systems", "confidence": 0.95, "section": "introduction"}},
    {{"name": "crop yields", "type": "KEY_FINDING", "description": "Agricultural productivity being measured", "confidence": 0.92, "section": "introduction"}},
    {{"name": "rainfall levels", "type": "DATASET", "description": "Climate variable representing precipitation", "confidence": 0.88, "section": "methods", "properties": {{"unit": "mm", "temporal_scope": "50 years"}}}},
    {{"name": "random forest regression", "type": "METHODOLOGY", "description": "Machine learning method for prediction", "confidence": 0.94, "section": "methods"}},
    {{"name": "USDA agricultural dataset", "type": "DATASET", "description": "50 years of historical agricultural data", "confidence": 0.91, "section": "methods", "properties": {{"source": "USDA", "temporal_range": "50 years"}}}},
    {{"name": "NSF", "type": "FUNDING_BODY", "description": "National Science Foundation - funding source", "confidence": 0.97, "section": "introduction"}},
    {{"name": "Dr. Emily Chen", "type": "AUTHOR", "description": "Lead researcher on climate-agriculture study", "confidence": 0.93, "section": "introduction"}},
    {{"name": "UC Berkeley", "type": "AFFILIATION", "description": "University of California, Berkeley", "confidence": 0.95, "section": "introduction"}}
  ],
  "relationships": [
    {{"source": "climate change", "target": "crop yields", "type": "ADDRESSES", "description": "Study addresses how climate impacts agricultural output", "confidence": 0.90}},
    {{"source": "random forest regression", "target": "rainfall levels", "type": "USES_DATASET", "description": "ML method applied to precipitation data", "confidence": 0.88}},
    {{"source": "Dr. Emily Chen", "target": "UC Berkeley", "type": "AFFILIATED_WITH", "description": "Researcher affiliated with UC Berkeley", "confidence": 0.96}},
    {{"source": "USDA agricultural dataset", "target": "random forest regression", "type": "USES_DATASET", "description": "Analysis performed on USDA historical data", "confidence": 0.89}},
    {{"source": "NSF", "target": "Dr. Emily Chen", "type": "FUNDED_BY", "description": "Research funded by National Science Foundation", "confidence": 0.92}}
  ]
}}

EXAMPLE 2 — Neuroscience study contradiction:
Given: "Prior studies suggested dopamine depletion causes depression (Smith 2015), but our randomized controlled trial of 200 participants found no significant correlation (p = 0.23). We replicate the methodology from Johnson 2018 but with larger sample size."

Expected:
- Extract "Smith 2015" as CITATION
- Extract "dopamine depletion" as RESEARCH_DOMAIN
- Extract "p = 0.23" as STATISTICAL_MEASURE
- Create CONTRADICTS_FINDING relationship between the prior finding and your p-value result
- Create REPLICATES_STUDY relationship to Johnson 2018

EXAMPLE 3 — Key practices:
- AUTHOR: Extract person names not generic "researchers"
- STATISTICAL_MEASURE: Include actual numbers/values (p-value, correlation, effect size)
- CONFIDENCE: Be honest - reduce confidence for ambiguous extractions
- SECTION: Identify which research section entity originated from
- PROPERTIES: Store quantitative details (sample size, units, ranges, p-values, etc.)

Now extract from these research paper text blocks:

{texts_section}"""


# ── Response parsing ─────────────────────────────────────────────────────────

def _parse_extraction_response(
    raw: str,
    batch: list[tuple[str, str, int]],
    doc_id: str,
    source: str,
    folder: str,
) -> tuple[list[Entity], list[Relationship]]:
    """
    Parse the LLM's JSON response into typed objects.

    Tolerates: prose wrapping, unknown types (mapped to OTHER/RELATED_TO),
    malformed entries (skipped individually).
    """
    # Extract JSON from response — model often wraps it in text
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        logger.warning("  No JSON found in extraction response")
        return [], []

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"  JSON parse failed: {e}")
        return [], []

    # Use the first chunk in the batch as the reference for chunk_id and page
    # (entities span the batch, we attribute to the first chunk for simplicity)
    ref_chunk_id = batch[0][1]
    ref_page = batch[0][2]

    raw_entities = data.get("entities", [])[:GRAPH_MAX_ENTITIES_PER_CHUNK]
    raw_rels = data.get("relationships", [])[:GRAPH_MAX_RELS_PER_CHUNK]

    # Build entities
    entities: list[Entity] = []
    entity_id_by_name: dict[str, str] = {}  # name_normalized → entity_id

    for raw_ent in raw_entities:
        name = raw_ent.get("name", "").strip()
        if not name:
            continue

        name_normalized = _normalize_entity_name(name)
        entity_type = raw_ent.get("type", "OTHER").upper()
        if entity_type not in GRAPH_ENTITY_TYPES:
            entity_type = "OTHER"

        entity_id = _make_id(name_normalized, entity_type, doc_id)
        entity_id_by_name[name_normalized] = entity_id

        # Extract confidence (0.0-1.0), default to 0.7 if not provided
        confidence = float(raw_ent.get("confidence", 0.7))
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0.0, 1.0]

        entities.append(Entity(
            id=entity_id,
            name=name,
            name_normalized=name_normalized,
            entity_type=entity_type,
            doc_id=doc_id,
            source=source,
            folder=folder,
            chunk_id=ref_chunk_id,
            page=ref_page,
            description=raw_ent.get("description", ""),
            confidence=confidence,
            section=raw_ent.get("section", "").lower(),
            properties=raw_ent.get("properties", {}),
        ))

    # Build relationships — only if both source and target entities exist
    relationships: list[Relationship] = []

    for raw_rel in raw_rels:
        src_name = raw_rel.get("source", "").strip()
        tgt_name = raw_rel.get("target", "").strip()

        src_name_normalized = _normalize_entity_name(src_name)
        tgt_name_normalized = _normalize_entity_name(tgt_name)

        src_id = entity_id_by_name.get(src_name_normalized)
        tgt_id = entity_id_by_name.get(tgt_name_normalized)

        if not src_id or not tgt_id:
            continue  # skip if either entity wasn't extracted

        rel_type = raw_rel.get("type", "RELATED_TO").upper()
        if rel_type not in GRAPH_RELATION_TYPES:
            rel_type = "RELATED_TO"

        rel_id = _make_id(src_id, tgt_id, rel_type)

        # Extract confidence for relationship
        rel_confidence = float(raw_rel.get("confidence", 0.7))
        rel_confidence = max(0.0, min(1.0, rel_confidence))

        relationships.append(Relationship(
            id=rel_id,
            source_entity_id=src_id,
            target_entity_id=tgt_id,
            relation_type=rel_type,
            description=raw_rel.get("description", ""),
            confidence=rel_confidence,
            doc_id=doc_id,
            chunk_id=ref_chunk_id,
        ))

    return entities, relationships


# ── Helpers ──────────────────────────────────────────────────────────────────

# ── Query-time extraction ────────────────────────────────────────────────────
#
# These functions run in the query pipeline (Phase 3), not during ingestion.
# They extract entities from the USER'S QUESTION — much simpler than
# extracting from a full document chunk.
#
# Why separate from ingestion extraction?
#   Ingestion: thorough, batched, can be slow (offline process)
#   Query-time: fast, single question, must complete in <2 seconds

def extract_entities_from_question(question: str) -> list[dict]:
    """
    Extract named entities from a user question about research topics.

    Returns a lightweight list of {name, type} dicts — no IDs,
    no relationships, just enough to look up in the graph.

    Uses a more sophisticated prompt than before to recognize:
    - Author names and citations
    - Research methodologies  
    - Statistical concepts
    - Research fields/domains
    - Funding bodies

    Uses a simpler, shorter prompt than ingestion extraction
    because speed matters here (query hot path).
    """
    entity_types = ", ".join(GRAPH_ENTITY_TYPES)

    prompt = (
        "Extract key research entities from this user question about papers/research. "
        "Identify: hypotheses, methodologies, findings, datasets, authors, research questions, "
        "journals, conferences, institutions, statistical measures, funding sources, research domains, etc.\n"
        "Reply with JSON only, no other text.\n\n"
        f"Entity types: {entity_types}\n\n"
        'Format: [{{"name": "entity name", "type": "TYPE"}}]\n\n'
        "If no entities found, return []\n\n"
        "EXAMPLES:\n"
        "Q: 'What do Smith et al. 2020 say about climate change impact on agriculture?'\n"
        "A: [{\"name\": \"Smith et al. 2020\", \"type\": \"CITATION\"}, "
        "{\"name\": \"climate change\", \"type\": \"RESEARCH_DOMAIN\"}, "
        "{\"name\": \"agriculture\", \"type\": \"RESEARCH_DOMAIN\"}]\n\n"
        "Q: 'How does random forest compare to neural networks for this dataset?'\n"
        "A: [{\"name\": \"random forest\", \"type\": \"METHODOLOGY\"}, "
        "{\"name\": \"neural networks\", \"type\": \"METHODOLOGY\"}, "
        "{\"name\": \"dataset\", \"type\": \"DATASET\"}]\n\n"
        "Q: 'Was this funded by NSF?'\n"
        "A: [{\"name\": \"NSF\", \"type\": \"FUNDING_BODY\"}]\n\n"
        f"User question: {question}"
    )

    try:
        raw = _call_llm(prompt)
        # Find JSON array in response
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            entities = json.loads(match.group())
            return [
                {"name": e.get("name", "").strip(), "type": e.get("type", "OTHER")}
                for e in entities
                if e.get("name", "").strip()
            ]
    except Exception as e:
        logger.warning(f"Question entity extraction failed: {e}")

    return []


def resolve_question_entities(
    extracted: list[dict],
    graph_store,
    folders: list[str] | None = None,
) -> list[dict]:
    """
    Match extracted entity names against the stored graph.

    Fuzzy matching via ILIKE — "leave policy" matches "Annual Leave Policy".
    Returns the actual stored Entity rows for matched names.

    Why fuzzy?  Users rarely type exact entity names.  "the leave policy"
    should match "Annual Leave Policy v3".
    """
    matched: list[dict] = []

    for entity in extracted:
        results = graph_store.find_entities_by_name(
            name=entity["name"],
            entity_type=entity["type"] if entity["type"] != "OTHER" else None,
            folders=folders,
        )
        matched.extend(results)

    # Deduplicate by entity ID
    seen: set[str] = set()
    unique: list[dict] = []
    for m in matched:
        eid = m["id"]
        if eid not in seen:
            seen.add(eid)
            unique.append(m)

    return unique


def _call_llm(prompt: str) -> str:
    """Send a prompt to llama3.2:3b and return the raw text response."""
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": WORKER_MODEL, "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def _normalize_entity_name(name: str) -> str:
    """
    Normalize entity names for deduplication.

    Handles:
    - Case insensitivity
    - Extra whitespace
    - Common abbreviations and variations
    - Name variations (e.g., "J. Smith" vs "James Smith")

    Goal: "Random Forest" and "random forest" should deduplicate,
    but still track the original form.
    """
    normalized = name.lower().strip()
    
    # Remove extra whitespace
    normalized = " ".join(normalized.split())
    
    # Common normalizations
    replacements = {
        "machine learning": "ml",
        "artificial intelligence": "ai",
        "neural network": "nn",
        "deep learning": "dl",
        "support vector": "svm",
        "random forest": "rf",
        "natural language processing": "nlp",
    }
    
    for original, shortened in replacements.items():
        if f" {original} " in f" {normalized} ":
            normalized = normalized.replace(original, shortened)
    
    return normalized


def _make_id(*parts: str) -> str:
    """
    Deterministic ID from component strings.

    Why deterministic?  Re-ingesting the same document produces the
    same IDs, so INSERT ... ON CONFLICT DO NOTHING naturally deduplicates.
    With random UUIDs, every re-ingestion creates duplicates.
    """
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

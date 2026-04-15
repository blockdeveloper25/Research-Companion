# Graph Extractor: Getting Started Guide

This guide helps you understand and use the enhanced graph extractor.

---

## Quick Start (5 minutes)

### Step 1: Understand What Changed

The graph extractor now:
- Extracts **24 types** of research entities (up from 10)
- Finds **22 types** of relationships (up from 11)  
- Assigns **confidence scores** to each extraction (0.0-1.0)
- Tracks the **research section** (abstract, methods, results, etc.)
- Captures **rich metadata** about findings

### Step 2: Run the System

```bash
cd ~/AI_RealWorld_Projects/Research-Companion
./start.sh
```

When ingesting documents, the enhanced extractor runs automatically.

### Step 3: Query the Results

```python
from db.connection import db_conn

with db_conn() as conn:
    with conn.cursor() as cur:
        # Find highly confident findings
        cur.execute("""
            SELECT name, description, confidence, section
            FROM entities
            WHERE entity_type = 'KEY_FINDING'
            AND confidence >= 0.8
            ORDER BY confidence DESC
            LIMIT 10
        """)
        for row in cur.fetchall():
            print(f"✓ {row['name']} (confidence: {row['confidence']})")
```

---

## New Entity Types to Know

### People & Institutions
- **AUTHOR** - Researcher names (e.g., "Dr. Jane Smith")
- **AFFILIATION** - Institutions (e.g., "MIT", "Stanford Medical")
- **FUNDING_BODY** - Funding sources (e.g., "NSF", "NIH")

### Research Concepts
- **RESEARCH_QUESTION** - Central question being asked
- **HYPOTHESIS** - Proposed answer/theory
- **FINDING** - Result or discovery
- **KEY_FINDING** - Highlighted important result
- **CONCLUSION** - Final interpretation

### Methods & Data
- **METHODOLOGY** - Research method
- **DATASET** - Data source
- **ALGORITHM** - Computational method
- **STATISTICAL_MEASURE** - Quantitative result (p-value, correlation, etc.)

### Context & References
- **RESEARCH_DOMAIN** - Field of study (neuroscience, ML, etc.)
- **CITATION** - Referenced work
- **LIMITATION** - Known weakness
- **ASSUMPTION** - Assumed condition

### Fallback
- **OTHER** - Unclassified

---

## New Relationship Types to Know

### Research Logic
- **ADDRESSES** - Hypothesis addresses research question
- **VALIDATES** - Finding validates hypothesis
- **CONTRADICTS** - Finding contradicts prior work
- **CONTRADICTS_FINDING** - Two findings contradict each other
- **SUPPORTS** - Finding supports conclusion

### Knowledge Building
- **CITES** - References or cites work
- **BUILDS_ON** - Extends prior research
- **REPLICATES_STUDY** - Replicates or validates prior study
- **DISPUTES** - Disputes or challenges finding
- **EXTENDS** - Extends methodology or theory

### Attribution
- **AUTHORED_BY** - Paper authored by person
- **AFFILIATED_WITH** - Author affiliated with institution
- **FUNDED_BY** - Research funded by organization

### Methods
- **USES_METHODOLOGY** - Uses this method
- **USES_DATASET** - Uses this data
- **MEASURES** - Quantifies finding
- **COMPARES_TO** - Comparison with other approach

---

## Understanding Confidence Scores

Each extracted entity and relationship has a **confidence score** (0.0 to 1.0):

| Score | Meaning | Use Case |
|-------|---------|----------|
| 0.95-1.0 | Very high confidence | Use directly in analysis |
| 0.85-0.95 | High confidence | Good for most uses |
| 0.70-0.85 | Good confidence | Use with caution |
| 0.50-0.70 | Medium confidence | May need verification |
| < 0.50 | Low confidence | Review manually |

**Example**: 
```python
# Get only high-confidence entities
reliable = entities.filter(e => e.confidence >= 0.85)

# Get all but review low-confidence ones
uncertain = entities.filter(e => e.confidence < 0.6)
```

---

## Understanding Research Sections

Entities are tagged with their source section:

- **abstract** - From paper abstract
- **introduction** - From background/introduction
- **methods** - From methodology description
- **results** - From results/findings section  
- **discussion** - From discussion/conclusion

**Why this matters**:
- Methodologies found in "methods" section are more reliable
- Findings in "results" section are primary results
- Discussion section contains interpretations

**Example**:
```python
# Get primary results (high confidence + results section)
primary_results = entities.filter(
    e => e.entity_type == 'FINDING' 
    and e.section == 'results' 
    and e.confidence >= 0.8
)
```

---

## Understanding Entity Properties

Entities can have rich metadata in the `properties` field:

### Statistical Measure
```json
{
  "value": "0.73",
  "type": "pearson_correlation",
  "p_value": "0.001",
  "sample_size": 150
}
```

### Dataset
```json
{
  "source": "USDA Agricultural Database",
  "temporal_range": "1970-2020",
  "sample_size": 10000,
  "url": "https://..."
}
```

### Author
```json
{
  "affiliation": "MIT",
  "email": "john@mit.edu",
  "orcid": "0000-0001-2345-6789"
}
```

### Citation
```json
{
  "authors": ["Smith", "Jones"],
  "year": 2020,
  "venue": "Nature",
  "doi": "10.1038/..."
}
```

---

## Common Query Patterns

### Find All Authors in Your Documents

```python
from db.connection import db_conn

with db_conn() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT name, COUNT(*) as mentions
            FROM entities
            WHERE entity_type = 'AUTHOR'
            GROUP BY name
            ORDER BY mentions DESC
        """)
        authors = cur.fetchall()
```

### Find Funding Sources

```python
with db_conn() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT name
            FROM entities
            WHERE entity_type = 'FUNDING_BODY'
            ORDER BY name
        """)
        funding = cur.fetchall()
```

### Find High-Confidence Contradictions

```python
with db_conn() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                e1.name as finding_1,
                e2.name as finding_2,
                r.confidence
            FROM relationships r
            JOIN entities e1 ON r.source_entity_id = e1.id
            JOIN entities e2 ON r.target_entity_id = e2.id
            WHERE r.relation_type = 'CONTRADICTS_FINDING'
            AND r.confidence >= 0.85
        """)
        contradictions = cur.fetchall()
```

### Find Who is Funded by NSF

```python
with db_conn() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT e2.name
            FROM relationships r
            JOIN entities e1 ON r.source_entity_id = e1.id
            JOIN entities e2 ON r.target_entity_id = e2.id
            WHERE e1.name = 'NSF'
            AND r.relation_type = 'FUNDED_BY'
        """)
        nsf_funded = cur.fetchall()
```

---

## Analyzing Your Extracted Graph

### Check Extraction Quality

```python
with db_conn() as conn:
    with conn.cursor() as cur:
        # Get quality metrics by entity type
        cur.execute("""
            SELECT 
                entity_type,
                COUNT(*) as count,
                ROUND(AVG(confidence)::numeric, 2) as avg_confidence,
                COUNT(*) FILTER (WHERE confidence >= 0.8) as high_conf
            FROM entities
            GROUP BY entity_type
            ORDER BY count DESC
        """)
        print(cur.fetchall())
```

### Identify Uncovered Research Domains

```python
with db_conn() as conn:
    with conn.cursor() as cur:
        # What domains are represented?
        cur.execute("""
            SELECT DISTINCT name
            FROM entities
            WHERE entity_type = 'RESEARCH_DOMAIN'
        """)
        domains = [row['name'] for row in cur.fetchall()]
        print(f"Found {len(domains)} research domains:")
        for d in domains:
            print(f"  - {d}")
```

### Find Most Connected Researchers

```python
with db_conn() as conn:
    with conn.cursor() as cur:
        # Who collaborates most?
        cur.execute("""
            SELECT 
                e.name,
                COUNT(DISTINCT r.id) as collaborations
            FROM entities e
            JOIN relationships r ON (
                e.id = r.source_entity_id 
                OR e.id = r.target_entity_id
            )
            WHERE e.entity_type = 'AUTHOR'
            GROUP BY e.name
            ORDER BY collaborations DESC
            LIMIT 10
        """)
        print(cur.fetchall())
```

---

## Advanced Usage

### Multi-Hop Query: Who Funded Studies That Cite Paper X?

```python
with db_conn() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT 
                e_fund.name as funder,
                e_author.name as researcher
            FROM relationships r1
            JOIN entities e_fund ON r1.source_entity_id = e_fund.id
            JOIN relationships r2 ON r1.target_entity_id = r2.source_entity_id
            JOIN entities e_author ON r2.target_entity_id = e_author.id
            WHERE r1.relation_type = 'FUNDED_BY'
            AND e_fund.entity_type = 'FUNDING_BODY'
            AND r2.relation_type = 'AUTHORED_BY'
            AND e_author.entity_type = 'AUTHOR'
        """)
        results = cur.fetchall()
```

### Track Research Evolution: What Builds on Study X?

```python
with db_conn() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                e1.name as original_study,
                e2.name as extending_study,
                r.description
            FROM relationships r
            JOIN entities e1 ON r.source_entity_id = e1.id
            JOIN entities e2 ON r.target_entity_id = e2.id
            WHERE r.relation_type IN (
                'BUILDS_ON', 
                'REPLICATES_STUDY', 
                'EXTENDS'
            )
        """)
        evolution = cur.fetchall()
```

---

## Troubleshooting

### Q: I'm getting empty results
**A**: 
- Make sure documents have been ingested
- Check confidence threshold isn't too high
- Verify entity_type name is correct (case-sensitive)

### Q: Confidence scores are low
**A**:
- This is normal for ambiguous text
- Use multiple sources to validate
- Filter by section for better signals

### Q: Missing certain entities
**A**:
- The LLM may not have recognized them
- Check if the section/document type matches
- Consider using a larger model (llama3.1:8b) for ingestion

### Q: How do I update the system?
**A**:
- Stop the app: `Ctrl+C`
- The code changes are already in place
- Run `./start.sh` to restart
- Re-ingest documents to get new extraction quality

---

## Next Steps

1. **Read full documentation**: See `GRAPH_EXTRACTOR_IMPROVEMENTS.md`
2. **Explore query examples**: See `GRAPH_QUERY_RECIPES.md`
3. **Build applications**: Use these queries in your backend
4. **Experiment**: Try different entity/relationship types
5. **Extend**: Add new types to config.py as needed

---

## Files Reference

| File | Purpose |
|------|---------|
| `backend/config.py` | Entity & relationship type definitions |
| `backend/rag/graph_extractor.py` | Core extraction logic |
| `backend/rag/graph_store.py` | Database storage |
| `backend/db/connection.py` | Database schema |
| `GRAPH_EXTRACTOR_IMPROVEMENTS.md` | Detailed feature guide |
| `GRAPH_QUERY_RECIPES.md` | 30+ SQL examples |
| `GRAPH_EXTRACTOR_SUMMARY.md` | Technical summary |

---

## Support

For more details:
- Check source code comments in `graph_extractor.py`
- Review database schema in `db/connection.py`
- See query examples in `GRAPH_QUERY_RECIPES.md`
- Test locally with small dataset first

Happy researching! 🔬📚


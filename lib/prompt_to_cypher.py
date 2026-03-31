import re

CYPHER_PROMPT = """
You are a Neo4j Cypher query generator for an ERP knowledge graph.

## Graph Schema

### Node label
{node_properties}

### Relation types
{relation_types}

## Node Schema
- name: entity name
- type: entity type
- mention: original mention text
- confidence: confidence level (high/medium/low)
- product: product name, example "Odoo"
- section_id: section identifier
- module: module name, example "Point of Sales"
- heading: section heading

## Rules
- MATCH and RETURN only — no MERGE, CREATE, DELETE
- Case-insensitive: toLower(n.name) CONTAINS toLower('value')
- Always LIMIT 20
- Return node names, types, relation types
- Return ONLY the Cypher query, no explanation

## Example
Question: "What features does the Accounting module have?"
MATCH (a)-[r:HAS_FEATURE]->(b)
WHERE toLower(a.name) CONTAINS toLower('accounting')
RETURN a.name, a.type, type(r) AS relation, b.name, b.type
LIMIT 20

Question: {question}
"""

# ── Safe execution with validation ────────────────────────────────

SAFE_CYPHER_PATTERN = re.compile(
    r'^\s*(MATCH|WITH|RETURN|WHERE|LIMIT|ORDER|UNWIND)',
    re.IGNORECASE
)
UNSAFE_KEYWORDS = re.compile(
    r'\b(MERGE|CREATE|DELETE|SET|REMOVE|DROP|CALL)\b',
    re.IGNORECASE
)

def is_safe_cypher(query: str) -> bool:
    if UNSAFE_KEYWORDS.search(query):
        return False
    if not SAFE_CYPHER_PATTERN.match(query):
        return False
    return True

def prompt_to_cypher(question: str, node_properties: str, relation_types: str) -> str:
    return CYPHER_PROMPT.format(question=question, node_properties=node_properties, relation_types=relation_types)

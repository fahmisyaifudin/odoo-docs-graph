import re
from openai import OpenAI
import json
import os
from dotenv import load_dotenv


load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

SCHEMA = {
    "entity_types": [
        {"type": "Product",     "description": "A top-level software product or platform",          "examples": ["Odoo", "Zoho CRM", "SAP S/4HANA"]},
        {"type": "Module",      "description": "A major functional area within a product",           "examples": ["Accounting", "HR", "Inventory"]},
        {"type": "Sub-module",  "description": "A sub-section or component within a module",         "examples": ["Bank Synchronization", "Asset Management"]},
        {"type": "Feature",     "description": "A specific capability or function",                  "examples": ["multi-currency", "double-entry bookkeeping"]},
        {"type": "Workflow",    "description": "A defined process or sequence of steps",             "examples": ["invoice approval", "bank reconciliation"]},
        {"type": "Action",      "description": "A discrete user or system operation",                "examples": ["export ZIP", "lock date", "reconcile"]},
        {"type": "Integration", "description": "A connection to an external system or service",      "examples": ["bank API sync", "payment gateway"]},
        {"type": "Report",      "description": "A generated financial or operational report",        "examples": ["Balance Sheet", "Partner Ledger", "Tax Report"]},
        {"type": "DataObject",  "description": "A structured record or document in the system",     "examples": ["invoice", "vendor bill", "journal entry"]},
        {"type": "Field",       "description": "A specific data attribute on a record",              "examples": ["VAT number", "invoice date", "fiscal period"]},
        {"type": "Setting",     "description": "A configuration option or system parameter",         "examples": ["Accounting Firms mode", "fiscal localization"]},
        {"type": "Permission",  "description": "An access control role or right",                    "examples": ["Accountant role", "bank account validation"]},
        {"type": "Standard",    "description": "An accounting or inventory valuation method",        "examples": ["FIFO", "AVCO", "accrual basis", "cash basis"]},
        {"type": "Concept",     "description": "A general business or accounting principle",         "examples": ["retained earnings", "double-entry", "fiscal year"]},
        {"type": "Actor",       "description": "A person, organization, or system entity",           "examples": ["customer", "vendor", "branch", "accountant"]},
        {"type": "Regulation",  "description": "A legal, tax, or compliance requirement",            "examples": ["VAT", "IFRS", "fiscal localization package"]},
    ],
    "relation_types": [
        {"type": "BELONGS_TO",    "description": "Entity is a member of a larger group",            "example": "Module –BELONGS_TO→ Product"},
        {"type": "CONTAINS",      "description": "Entity holds or owns another entity",             "example": "Module –CONTAINS→ Sub-module"},
        {"type": "IS_PART_OF",    "description": "Entity is a structural component of another",     "example": "Field –IS_PART_OF→ DataObject"},
        {"type": "HAS_FEATURE",   "description": "Entity provides a capability",                    "example": "Module –HAS_FEATURE→ Feature"},
        {"type": "SUPPORTS",      "description": "Entity is compatible with a standard or method",  "example": "Module –SUPPORTS→ Standard"},
        {"type": "ENABLES",       "description": "Feature makes another feature or action possible","example": "Feature –ENABLES→ Workflow"},
        {"type": "REQUIRES",      "description": "Entity depends on another to function",           "example": "Feature –REQUIRES→ Setting"},
        {"type": "GENERATES",     "description": "Entity automatically produces another entity",    "example": "Workflow –GENERATES→ DataObject"},
        {"type": "HAS_REPORT",    "description": "Entity exposes a report",                         "example": "Module –HAS_REPORT→ Report"},
        {"type": "PRODUCES",      "description": "Action results in a data output",                 "example": "Action –PRODUCES→ DataObject"},
        {"type": "PERFORMED_BY",  "description": "Action is executed by an actor",                  "example": "Action –PERFORMED_BY→ Actor"},
        {"type": "CONFIGURED_BY", "description": "Entity is set up via a setting or permission",    "example": "Feature –CONFIGURED_BY→ Setting"},
        {"type": "TRIGGERED_BY",  "description": "Workflow or action starts from an event/action",  "example": "Workflow –TRIGGERED_BY→ Action"},
        {"type": "APPLIES_TO",    "description": "Rule, tax or regulation applies to an entity",    "example": "Regulation –APPLIES_TO→ Actor"},
        {"type": "COMPLIES_WITH", "description": "Entity adheres to a regulation or standard",      "example": "Module –COMPLIES_WITH→ Regulation"},
        {"type": "USES_METHOD",   "description": "Entity applies a specific standard or method",    "example": "Feature –USES_METHOD→ Standard"},
        {"type": "GOVERNED_BY",   "description": "Entity is controlled by a regulation or setting", "example": "DataObject –GOVERNED_BY→ Regulation"},
        {"type": "RELATED_TO",    "description": "Loose semantic relationship between entities",    "example": "Feature –RELATED_TO→ Feature"},
        {"type": "SEE_ALSO",      "description": "Explicit documentation cross-reference",          "example": "Report –SEE_ALSO→ Report"},
        {"type": "EQUIVALENT_TO", "description": "Same concept in a different product/vendor",      "example": "Odoo Accounting –EQUIVALENT_TO→ Zoho Books"},
    ]
}

EXTRACTION_PROMPT = """
Extract entities and relationships from this ERP documentation section for a knowledge graph.

ENTITY TYPES:
{entity_types}

RELATION TYPES:
{relation_types}

RULES:
- Only extract what is explicitly stated
- Canonical names must be clean (no markup)
- Assign "Unknown" if no type fits — do not force a fit
- Return ONLY valid JSON, no markdown fences, do not show reasoning, do not explain


--- SECTION ---
Product: {product_name}
Title: {heading}
{cleaned_content}
--- END ---

{{"entities":[{{"name":"...","type":"...","mention":"...","confidence":"high|medium|low"}}],"relations":[{{"source":"...","relation":"...","target":"...","evidence":"..."}}]}}
"""

def parse_rst_sections(raw_rst: str) -> list[dict]:
    """
    Split RST content into sections by heading underlines.
    Returns list of {section_id, heading, raw_rst_content}
    """
    lines = raw_rst.splitlines()
    sections = []
    i = 0

    while i < len(lines):
        # Detect heading: line followed by underline of ===, ---, ~~~, ^^^
        if i + 1 < len(lines) and re.match(r'^[=\-~^]{3,}$', lines[i + 1].strip()):
            heading = lines[i].strip()
            underline_char = lines[i + 1][0]
            level = {'=': 1, '-': 2, '~': 3, '^': 4}.get(underline_char, 5)

            # Collect content until the next heading
            content_lines = []
            j = i + 2
            while j < len(lines):
                if j + 1 < len(lines) and re.match(r'^[=\-~^]{3,}$', lines[j + 1].strip()):
                    break
                content_lines.append(lines[j])
                j += 1

            section_id = heading.lower().replace(' ', '-').replace('/', '-')
            sections.append({
                "section_id": section_id,
                "heading": heading,
                "level": level,
                "raw_rst_content": "\n".join(content_lines).strip()
            })
            i = j
        else:
            i += 1

    return sections

def strip_rst_markup(text: str) -> str:
    """
    Light RST cleanup to reduce noise before sending to LLM.
    The LLM handles the rest — don't over-engineer this.
    """
    # Remove image directives and their options
    text = re.sub(r'\.\. image::.*?(?=\n\S|\Z)', '', text, flags=re.DOTALL)
    # Remove substitution definitions
    text = re.sub(r'\.\. \|.*?\|.*?\n', '', text)
    # Remove internal anchor labels
    text = re.sub(r'\.\. _[\w/\-]+:\n', '', text)
    # Extract clean text from roles: :guilabel:`X` → X
    text = re.sub(r':\w+:`([^`]+)`', r'\1', text)
    # Remove substitution references like |clock|
    text = re.sub(r'\|\w+\|', '', text)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def build_schema_blocks() -> tuple[str, str]:
    entity_block = "\n".join(
        f"- {t['type']}: {t['description']}"
        for t in SCHEMA["entity_types"]
    )
    relation_block = "\n".join(
        f"- {t['type']}: {t['description']}"
        for t in SCHEMA["relation_types"]
    )
    return entity_block, relation_block

ENTITY_BLOCK, RELATION_BLOCK = build_schema_blocks()

def build_prompt(section: dict, product_name: str) -> str:
    return EXTRACTION_PROMPT.format(
        product_name=product_name,
        heading=section["heading"],
        cleaned_content=section["cleaned_content"],
        entity_types=ENTITY_BLOCK,
        relation_types=RELATION_BLOCK
    )

def build_prompts_from_rst(raw_rst: str, product_name: str) -> list[dict]:
    sections = parse_rst_sections(raw_rst)
    prompts = []

    for section in sections:
        cleaned = strip_rst_markup(section["raw_rst_content"])
        if len(cleaned.strip()) < 50:
            continue
        section["cleaned_content"] = cleaned
        prompts.append({
            "section_id": section["section_id"],
            "heading": section["heading"],
            "prompt": build_prompt(section, product_name)
        })

    return prompts

def extract_graph_from_section(prompt: str) -> dict:
    """
    Single-turn extraction call with extended thinking enabled.
    Returns parsed JSON or an error dict.
    """
    response = client.chat.completions.create(
        model="deepseek/deepseek-v3.2",
        messages=[
            {"role": "user", "content": prompt}
        ],
        extra_body={"reasoning": {"enabled": False}}
    )

    message = response.choices[0].message
    raw_text = message.content

    try:
        # Strip accidental markdown fences if the model adds them
        clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_text.strip())
        return json.loads(clean)
    except json.JSONDecodeError as e:
        return {
            "error": str(e),
            "raw_response": raw_text
        }

from pathlib import Path

def save_output(results: list[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} section(s) → {path}")


def process_rst_file(
    rst_content: str,
    product_name: str,
    output_path: str = "output/graph.json"
) -> list[dict]:

    prompts = build_prompts_from_rst(rst_content, product_name)
    print(f"Found {len(prompts)} section(s) to process.")

    results = []

    for i, item in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] Extracting: {item['heading']}")

        graph = extract_graph_from_section(item["prompt"])

        results.append({
            "section_id": item["section_id"],
            "heading": item["heading"],
            "product": product_name,
            "graph": graph
        })

    save_output(results, output_path)
    return results

def main():
    with open("rst_loop.txt", "r") as f:
        rst_files = [line.strip() for line in f if line.strip()]

    for index, rst_file in enumerate(rst_files):
        rst_content = Path(rst_file).read_text(encoding="utf-8")

        results = process_rst_file(
            rst_content=rst_content,
            product_name="Odoo",
            output_path=f"output/{index}_{Path(rst_file).stem}_graph.json"
        )

        # Quick summary print
        for r in results:
            g = r["graph"]
            if not isinstance(g, dict):
                print(f"  SKIP {r['section_id']}: graph is not a JSON object")
                continue

            if "error" in g:
                print(f"  SKIP {r['section_id']}: {g['error']}")
                continue

            n_entities = len(g.get("entities", []))
            n_relations = len(g.get("relations", []))
            print(f"  {r['section_id']}: {n_entities} entities, {n_relations} relations")

main()
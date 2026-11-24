"""Extract knowledge graphs from requirement specifications.

Input : Markdown files in ``output/requirement/`` (``schema.md`` is skipped).
Output: One graph JSON per input file in ``output/requirement/graph/``.

Language model: DeepSeek V4.1 Flash via OpenRouter with high reasoning effort.

Usage (from the repository root):

    python cmd/prompting.py
"""

import argparse
import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "deepseek/deepseek-v4.1-flash"
REASONING_EFFORT = "high"

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = REPO_ROOT / "output" / "requirement"
OUTPUT_DIR = INPUT_DIR / "graph"
MIN_SECTION_CHARS = 50

SCHEMA = {
    "entity_types": [
        {"type": "Product", "description": "A top-level software product, platform, or system under specification", "examples": ["Odoo", "CCTNS", "UUIS", "PeaZip"]},
        {"type": "Module", "description": "A major functional area or function block within a product", "examples": ["Registration", "Investigation", "HR", "Inventory"]},
        {"type": "Sub-module", "description": "A sub-section or component within a module", "examples": ["Bank Synchronization", "Asset Management", "Access Control"]},
        {"type": "Feature", "description": "A specific capability or function the product provides or the specification requires", "examples": ["multi-currency", "role-based landing pages", "advanced search"]},
        {"type": "Workflow", "description": "A defined process or sequence of steps", "examples": ["complaint → investigation → prosecution", "asset transfer approval"]},
        {"type": "Action", "description": "A discrete user or system operation", "examples": ["export audit trail", "register complaint", "approve transfer request"]},
        {"type": "Integration", "description": "A connection to an external system or service", "examples": ["SSO", "SMS/email gateway", "PDA/mobile terminal", "payment gateway"]},
        {"type": "Report", "description": "A generated operational or compliance report", "examples": ["Balance Sheet", "audit report", "monthly MIS report", "RTI report"]},
        {"type": "DataObject", "description": "A structured record or document in the system", "examples": ["case", "suspect", "asset", "complaint", "audit record", "user profile"]},
        {"type": "Field", "description": "A specific data attribute on a record", "examples": ["event timestamp", "workstation address", "asset identifier", "VAT number"]},
        {"type": "Setting", "description": "A configuration option or system parameter", "examples": ["unauthorised-access response mode", "master data (acts, sections, castes)", "fiscal localisation"]},
        {"type": "Permission", "description": "An access control right or privilege", "examples": ["case-level access", "change security attributes", "approve transfer", "delegate permissions"]},
        {"type": "Standard", "description": "An external or industry standard the system complies with or implements", "examples": ["ISO 9241", "WCAG 1.0", "FIFO", "AVCO", "HTTPS/SSL"]},
        {"type": "Concept", "description": "A domain principle, architectural pattern, or abstract term", "examples": ["3C principle (Core–Configuration–Customisation)", "audit trail", "modus operandi", "separation of concerns"]},
        {"type": "Actor", "description": "A user category, organisational role, or external system entity that interacts with the system", "examples": ["Investigating Officer", "super-user", "citizen", "faculty group", "external auditor"]},
        {"type": "Regulation", "description": "A legal, policy, or compliance requirement the system must satisfy", "examples": ["RTI Act", "legal admissibility", "national security classification", "IFRS", "VAT"]},
        {"type": "Requirement", "description": "A prescriptive statement in the specification that defines what the system shall do, how it shall behave, or what constraints it must satisfy. Carries req_id, req_type, modality, priority and status.", "examples": ["FR-REG-01 (complaint registration)", "BR-07 (unalterable audit trail)", "NFR-PER-02 (search ≤ 8 s)", "ARC-05 (SOA design)"]},
    ],
    "requirement_properties": [
        {"name": "req_id", "type": "string", "description": "Document-level identifier (e.g. FR-REG-01, BR-15, NFR-PER-02, ARC-05); null if the requirement has no id"},
        {"name": "req_type", "type": "enum", "description": "FR (functional), BR (business rule), NFR (non-functional), ARC (architecture constraint)"},
        {"name": "modality", "type": "enum", "description": "shall (mandatory), should (recommended), may (optional)"},
        {"name": "priority", "type": "string", "description": "Stakeholder priority if stated (e.g. high, medium, low); null otherwise"},
        {"name": "status", "type": "enum", "description": "defined (fully specified), tbd (placeholder such as <xx:00>), deferred, out_of_scope"},
    ],
    "relation_types": [
        {"type": "BELONGS_TO", "description": "Entity is a member of a larger group", "example": "Module → Product; Actor → Actor (group)"},
        {"type": "CONTAINS", "description": "Entity holds or owns another entity", "example": "Module → Sub-module"},
        {"type": "IS_PART_OF", "description": "Entity is a structural component of another", "example": "Field → DataObject"},
        {"type": "HAS_FEATURE", "description": "Entity provides a capability", "example": "Module → Feature"},
        {"type": "SUPPORTS", "description": "Entity is compatible with a standard or method", "example": "Module → Standard"},
        {"type": "ENABLES", "description": "Entity makes another feature or action possible", "example": "Feature → Workflow"},
        {"type": "REQUIRES", "description": "Entity depends on another to function", "example": "Feature → Setting"},
        {"type": "GENERATES", "description": "Entity automatically produces another entity", "example": "Workflow → DataObject"},
        {"type": "HAS_REPORT", "description": "Entity exposes a report", "example": "Module → Report"},
        {"type": "PRODUCES", "description": "Action results in a data output", "example": "Action → DataObject"},
        {"type": "PERFORMED_BY", "description": "Action is executed by an actor", "example": "Action → Actor"},
        {"type": "CONFIGURED_BY", "description": "Entity is set up via a setting or permission", "example": "Feature → Setting"},
        {"type": "TRIGGERED_BY", "description": "Workflow or action starts from an event or action", "example": "Workflow → Action"},
        {"type": "APPLIES_TO", "description": "Rule, regulation, or permission applies to an entity", "example": "Regulation → Actor; Permission → DataObject"},
        {"type": "COMPLIES_WITH", "description": "Entity adheres to a regulation or standard", "example": "Module → Regulation"},
        {"type": "USES_METHOD", "description": "Entity applies a specific standard or method", "example": "Feature → Standard"},
        {"type": "GOVERNED_BY", "description": "Entity is controlled by a regulation or setting", "example": "DataObject → Regulation"},
        {"type": "RELATED_TO", "description": "Loose semantic relationship between entities", "example": "Feature → Feature"},
        {"type": "SEE_ALSO", "description": "Explicit documentation cross-reference", "example": "Report → Report"},
        {"type": "EQUIVALENT_TO", "description": "Same capability, concept, or requirement expressed differently, within or across products; primary signal for redundancy detection", "example": "Requirement → Requirement; Feature → Feature"},
        {"type": "CONSTRAINS", "description": "A requirement (BR, NFR, or ARC) limits, restricts, or imposes conditions on an entity", "example": "Requirement → Module | Feature | DataObject | Action"},
        {"type": "HAS_PERMISSION", "description": "An actor holds a specific access right or privilege", "example": "Actor → Permission"},
        {"type": "DERIVED_FROM", "description": "A requirement refines, elaborates, or traces back to another requirement", "example": "Requirement → Requirement"},
    ],
}

EXTRACTION_PROMPT = """
Extract entities and relationships from this requirements-specification section for a knowledge graph.

ENTITY TYPES:
{entity_types}

RELATION TYPES:
{relation_types}

REQUIREMENT PROPERTIES (only for entities of type Requirement):
{requirement_properties}

RULES:
- Only extract what is explicitly stated. Do not infer, invent, or generalise.
- Every numbered or prescriptive statement (FR-*, BR-*, NFR-*, ARC-*) must become an entity of type Requirement.
- The Requirement entity name must be its identifier as written (e.g. "FR-REG-01"). If no id exists, use a short canonical name and set req_id to null.
- Include a "properties" object only for Requirement entities. Omit it for every other type.
- req_type must be one of FR, BR, NFR, ARC. modality must be one of shall, should, may. status must be one of defined, tbd, deferred, out_of_scope. Use null when the specification does not state a value.
- Canonical names must be clean: no markup, no numbering, no surrounding punctuation.
- Relation source and target must match extracted entity names exactly.
- Assign type "Unknown" if no type fits — do not force a fit.
- Return ONLY valid JSON. No markdown fences, no reasoning, no explanation.

--- SECTION ---
Product: {product_name}
Section: {heading}
{cleaned_content}
--- END ---

{{"entities":[{{"name":"...","type":"...","mention":"..."}}],"relations":[{{"source":"...","relation":"...","target":"...","evidence":"..."}}]}}
"""


def build_schema_blocks() -> tuple[str, str, str]:
    entity_block = "\n".join(
        f"- {t['type']}: {t['description']} (e.g. {', '.join(t['examples'])})"
        for t in SCHEMA["entity_types"]
    )
    relation_block = "\n".join(
        f"- {t['type']}: {t['description']} (e.g. {t['example']})"
        for t in SCHEMA["relation_types"]
    )
    property_block = "\n".join(
        f"- {p['name']} ({p['type']}): {p['description']}"
        for p in SCHEMA["requirement_properties"]
    )
    return entity_block, relation_block, property_block


ENTITY_BLOCK, RELATION_BLOCK, REQUIREMENT_PROPERTY_BLOCK = build_schema_blocks()


def parse_markdown_sections(raw_md: str) -> list[dict]:
    """Split Markdown into sections by ATX headings.

    Returns sections with the heading path (e.g. "3. Functional Requirements >
    Registration"), so nested headings keep their module context.
    """
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
    lines = raw_md.splitlines()

    sections = []
    stack: list[tuple[int, str]] = []  # (level, heading)
    current = None

    def flush(section):
        if section is None:
            return
        section["raw_content"] = "\n".join(section.pop("lines")).strip()
        if section["raw_content"]:
            sections.append(section)

    for line in lines:
        match = heading_re.match(line)
        if match:
            flush(current)
            level = len(match.group(1))
            heading = match.group(2).strip()

            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))

            # The H1 is the document title, already provided as Product;
            # keep section paths relative to it.
            path_stack = stack[1:] if len(stack) > 1 and stack[0][0] == 1 else stack
            path = " > ".join(text for _, text in path_stack)
            slug = re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")
            current = {
                "heading": path,
                "level": level,
                "section_id": slug,
                "lines": [],
            }
        elif current is not None:
            current["lines"].append(line)

    flush(current)
    return sections


def strip_markdown_markup(text: str) -> str:
    """Light Markdown cleanup to reduce noise before sending to the LLM."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)          # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)       # links -> text
    text = re.sub(r"`([^`]+)`", r"\1", text)                   # inline code
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)            # bold
    text = re.sub(r"^[ \t]*[-*_]{3,}[ \t]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_prompt(section: dict, product_name: str) -> str:
    return EXTRACTION_PROMPT.format(
        product_name=product_name,
        heading=section["heading"],
        cleaned_content=section["cleaned_content"],
        entity_types=ENTITY_BLOCK,
        relation_types=RELATION_BLOCK,
        requirement_properties=REQUIREMENT_PROPERTY_BLOCK,
    )


def build_prompts_from_markdown(raw_md: str, product_name: str) -> list[dict]:
    prompts = []

    for section in parse_markdown_sections(raw_md):
        cleaned = strip_markdown_markup(section["raw_content"])
        if len(cleaned) < MIN_SECTION_CHARS:
            continue
        section["cleaned_content"] = cleaned
        prompts.append({
            "section_id": section["section_id"],
            "heading": section["heading"],
            "prompt": build_prompt(section, product_name),
        })

    return prompts


def parse_json_response(raw_text: str) -> dict | None:
    """Parse a JSON object, tolerating fences and trailing data."""
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw_text or "").strip())

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    start = clean.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(clean[start:])
            return obj
        except json.JSONDecodeError:
            pass

    return None


def extract_graph_from_section(prompt: str, retries: int = 3) -> dict:
    """Single-turn extraction call with high reasoning effort enabled.

    Retries when the response is not valid JSON.
    """
    raw_text = ""

    for _ in range(retries):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            extra_body={"reasoning": {"effort": REASONING_EFFORT}},
        )
        raw_text = response.choices[0].message.content or ""

        parsed = parse_json_response(raw_text)
        if parsed is not None:
            return parsed

    return {
        "error": "invalid JSON after retries",
        "raw_response": raw_text,
    }


def save_output(results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} section(s) -> {output_path}")


def product_name_for(md_path: Path, raw_md: str) -> str:
    """Prefer the product name from the matching requirement CSV."""
    prefix = md_path.stem.split("-")[0]
    chunks_csv = REPO_ROOT / "sql" / "requirement" / f"{prefix}-chunks.csv"
    if chunks_csv.exists():
        with open(chunks_csv, newline="", encoding="utf-8") as f:
            first = next(csv.DictReader(f), None)
            if first and (first.get("product") or "").strip():
                return first["product"].strip()

    title_match = re.search(r"^#\s+(.+)$", raw_md, flags=re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()
        return title.split(" — ")[0].strip()
    return md_path.stem


def process_markdown_file(
    md_content: str,
    product_name: str,
    output_path: Path,
    workers: int = 1,
) -> list[dict]:
    prompts = build_prompts_from_markdown(md_content, product_name)
    print(f"Found {len(prompts)} section(s) to process for {product_name} "
          f"with {workers} worker(s).")

    results: list[dict | None] = [None] * len(prompts)
    done = 0

    def run(item: dict) -> dict:
        try:
            return extract_graph_from_section(item["prompt"])
        except Exception as e:  # network/API errors shouldn't kill the run
            return {"error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(run, item): (index, item)
            for index, item in enumerate(prompts)
        }
        for future in as_completed(futures):
            index, item = futures[future]
            graph = future.result()
            results[index] = {
                "section_id": item["section_id"],
                "heading": item["heading"],
                "product": product_name,
                "graph": graph,
            }
            done += 1
            print(f"[{done}/{len(prompts)}] Done: {item['heading']}")

    save_output(results, output_path)
    return results


def retry_failed_sections(md_files: list[Path], workers: int) -> None:
    """Re-extract only sections whose stored graph contains an error."""
    for md_path in md_files:
        output_path = OUTPUT_DIR / f"{md_path.stem}_graph.json"
        if not output_path.exists():
            print(f"{md_path.name}: no output yet, skipping")
            continue

        results = json.loads(output_path.read_text(encoding="utf-8"))
        failed = [
            i for i, result in enumerate(results)
            if isinstance(result.get("graph"), dict) and "error" in result["graph"]
        ]
        if not failed:
            print(f"{md_path.name}: no failed sections")
            continue

        raw_md = md_path.read_text(encoding="utf-8")
        product_name = product_name_for(md_path, raw_md)
        prompts = {
            item["section_id"]: item
            for item in build_prompts_from_markdown(raw_md, product_name)
        }

        print(f"{md_path.name}: retrying {len(failed)} failed section(s)")

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {}
            for index in failed:
                section_id = results[index]["section_id"]
                item = prompts.get(section_id)
                if item is None:
                    print(f"  no prompt for {section_id}, skipping")
                    continue
                futures[executor.submit(extract_graph_from_section, item["prompt"])] = index

            for future in as_completed(futures):
                index = futures[future]
                try:
                    graph = future.result()
                except Exception as e:
                    graph = {"error": f"{type(e).__name__}: {e}"}
                results[index]["graph"] = graph
                status = "OK" if "error" not in graph else "STILL ERROR"
                print(f"  {results[index]['section_id']}: {status}")

        save_output(results, output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=None,
                        help="Process only this Markdown file (name or path)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent extraction requests (default: 1)")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Re-extract only sections that failed JSON parsing")
    args = parser.parse_args()

    if args.file:
        md_path = Path(args.file)
        if not md_path.is_absolute():
            candidate = INPUT_DIR / md_path
            md_path = candidate if candidate.exists() else Path(args.file)
        if not md_path.exists():
            raise SystemExit(f"File not found: {args.file}")
        md_files = [md_path]
    else:
        md_files = sorted(
            path for path in INPUT_DIR.glob("*.md") if path.name != "schema.md"
        )
    if not md_files:
        print(f"No Markdown files found in {INPUT_DIR}")
        return

    if args.retry_errors:
        retry_failed_sections(md_files, args.workers)
        return

    for md_path in md_files:
        raw_md = md_path.read_text(encoding="utf-8")
        product_name = product_name_for(md_path, raw_md)

        results = process_markdown_file(
            md_content=raw_md,
            product_name=product_name,
            output_path=OUTPUT_DIR / f"{md_path.stem}_graph.json",
            workers=args.workers,
        )

        for result in results:
            graph = result["graph"]
            if not isinstance(graph, dict):
                print(f"  SKIP {result['section_id']}: graph is not a JSON object")
                continue

            if "error" in graph:
                print(f"  SKIP {result['section_id']}: {graph['error']}")
                continue

            n_entities = len(graph.get("entities", []))
            n_relations = len(graph.get("relations", []))
            print(
                f"  {result['section_id']}: "
                f"{n_entities} entities, {n_relations} relations"
            )


if __name__ == "__main__":
    main()

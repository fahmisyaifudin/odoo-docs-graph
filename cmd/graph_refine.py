"""
Entity matching + graph refinement for the requirement-spec knowledge graphs.

Reads each local `output/requirement/graph/0XX-standard_graph.json` file (kept
as the structural ground truth, since the live Neo4j graph merges nodes by
(name, type) globally across products and is therefore not a safe source for
per-product structure).

For each file, writes a sibling `0XX-standard_graph_refine.json` with the same
schema (list of {section_id, heading, product, graph: {entities, relations}}),
plus a `0XX-standard_graph_entity_candidates.csv` of embedding-similarity pairs
for manual review (NOT auto-applied -- see note below).

Refinement steps, in order:
  1. Type reconciliation     - same surface name used with >1 entity type
                                across the file -> unify to the majority type.
  2. Casing canonicalization - same name differing only by case/whitespace
                                (within the same type) -> merge to one spelling.
  3. Acronym/full-name matching - deterministic: if one entity's name is the
     initials of another entity's (non-stopword) words -- e.g. "FRS V1.0" /
     "Functional Requirements Specification V1.0 (Draft)" -- add an
     EQUIVALENT_TO edge (kept as distinct entities; only added where both
     already co-occur in the same section, so no entity has to be duplicated
     across sections to host the new edge). Requirement-type entities and any
     ID-coded name (e.g. "BR-02", "ARC-21") are excluded, since those carry a
     real identifier and must never be equated by surface-form heuristics.
  4. Relation-direction normalization - a conservative, schema-grounded set of
     relation types where the target's entity type is unambiguous regardless
     of context (e.g. PERFORMED_BY's target must be an Actor); any instance
     found backwards is flipped. Ambiguous types (BELONGS_TO, CONTAINS, ...)
     are intentionally left alone -- manual evidence review showed blind
     type-pattern flipping produces false positives there.
  5. Orphan-node connection - entities absent from every relation in their
     section get linked to the section's Product entity.
  6. Missing intermediate edges - a Module entity with no BELONGS_TO edge to
     its section's Product entity gets one added.

Note on embeddings: an earlier version of this script auto-merged entities
whose Neo4j embedding vectors were highly similar. That was reverted after
testing showed it isn't safe on this corpus: short, templated identifiers
like "BR-02" and "BR-20" embed MORE similarly (cosine ~0.97) than genuine
equivalences like "CCTNS" and its spelled-out name (cosine ~0.65) -- there is
no threshold that keeps the real matches in and the false ones out, and it
silently merged distinct requirements together. Embeddings are still used
here, but only to produce a ranked CSV of candidate pairs for a human to
confirm, never to auto-apply a merge or a relation.
"""

import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

GRAPH_DIR = "output/requirement/graph"
NEO4J_DATABASE = "qwen06b"
CANDIDATE_REVIEW_THRESHOLD = 0.65

ID_PATTERN = re.compile(r"^[A-Za-z]{1,6}-\d+[A-Za-z]?$")
STOPWORDS = {"a", "an", "the", "of", "and", "or", "&", "for", "to", "in", "on", "with", "draft"}

SCHEMA_TYPE_NAMES = {
    "Product", "Module", "Sub-module", "Feature", "Workflow", "Action",
    "Integration", "Report", "DataObject", "Field", "Setting", "Permission",
    "Standard", "Concept", "Actor", "Regulation", "Requirement",
}
REQUIREMENT_PROPERTY_KEYS = {"req_id", "req_type", "modality", "priority", "status"}

# relation types where the TARGET's entity type is unambiguous regardless of
# evidence wording, so a reversed instance can be auto-corrected safely.
DIRECTION_RULES = {
    "PERFORMED_BY": {"target_types": {"Actor"}},
    "HAS_PERMISSION": {"target_types": {"Permission"}},
    "HAS_FEATURE": {"target_types": {"Feature"}},
    "HAS_REPORT": {"target_types": {"Report"}},
    "CONFIGURED_BY": {"target_types": {"Setting"}},
    "PRODUCES": {"target_types": {"DataObject"}},
}


def norm(name):
    return re.sub(r"\s+", " ", name.strip().lower())


def looks_like_id(name):
    return bool(ID_PATTERN.match(name.strip()))


def initials(text):
    words = re.findall(r"[A-Za-z]+", text)
    sig = [w for w in words if w.lower() not in STOPWORDS]
    return "".join(w[0] for w in sig).upper()


def acronym_match(a, b):
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    short_letters = re.sub(r"[^A-Za-z]", "", short).upper()
    long_letters = re.sub(r"[^A-Za-z]", "", long_).upper()
    if not (2 <= len(short_letters) <= 10) or short_letters == long_letters:
        return False
    long_initials = initials(long_)
    return short_letters == long_initials or (
        len(short_letters) >= 3 and short_letters in long_initials
    )


def fix_flattened_requirement_properties(sections):
    """Some source files put req_id/req_type/modality/priority/status directly
    on the entity instead of nesting them under `properties`, inconsistent
    with the rest of the corpus and the schema. Normalize in place."""
    fixed = 0
    for s in sections:
        for e in s["graph"]["entities"]:
            if e.get("type") == "Requirement" and "properties" not in e:
                flat = {k: e.pop(k) for k in list(e.keys()) if k in REQUIREMENT_PROPERTY_KEYS}
                if flat:
                    e["properties"] = flat
                    fixed += 1
    return fixed


def backfill_missing_entities(sections):
    """A relation may reference a source/target name with no matching entity
    in its section -- either a genuinely missing entity (add one so the graph
    is referentially complete) or, rarely, an extraction mistake where the
    endpoint is literally one of the 17 schema type names rather than an
    entity (drop that relation instead of fabricating a node named e.g.
    "DataObject")."""
    added = 0
    dropped = 0
    for s in sections:
        g = s["graph"]
        names = {e["name"] for e in g["entities"]}
        kept = []
        for r in g["relations"]:
            missing = [f for f in ("source", "target") if r[f] not in names]
            if missing and any(r[f] in SCHEMA_TYPE_NAMES for f in missing):
                dropped += 1
                continue
            for f in missing:
                nm = r[f]
                g["entities"].append({"name": nm, "type": "Unknown", "mention": r["evidence"]})
                names.add(nm)
                added += 1
            kept.append(r)
        g["relations"] = kept
    return added, dropped


def reconcile_types(sections):
    name_type_counts = defaultdict(Counter)
    for s in sections:
        for e in s["graph"]["entities"]:
            name_type_counts[e["name"]][e["type"]] += 1
    majority_type = {
        name: counter.most_common(1)[0][0]
        for name, counter in name_type_counts.items()
        if len(counter) > 1
    }
    changed = 0
    for s in sections:
        for e in s["graph"]["entities"]:
            mt = majority_type.get(e["name"])
            if mt and e["type"] != mt:
                e["type"] = mt
                changed += 1
    return changed


def canonicalize_casing(sections):
    bucket_counts = defaultdict(Counter)
    for s in sections:
        for e in s["graph"]["entities"]:
            bucket_counts[(norm(e["name"]), e["type"])][e["name"]] += 1

    rename_map = {}
    for (n, t), counter in bucket_counts.items():
        if len(counter) == 1:
            continue
        canonical = max(
            counter.items(),
            key=lambda kv: (kv[1], sum(c.isupper() for c in kv[0]), len(kv[0])),
        )[0]
        for original in counter:
            if original != canonical:
                rename_map[(original, t)] = canonical

    renamed = 0
    for s in sections:
        for e in s["graph"]["entities"]:
            key = (e["name"], e["type"])
            if key in rename_map:
                e["name"] = rename_map[key]
                renamed += 1

    # relation endpoints don't carry an explicit type, so use a file-wide
    # raw->canonical map that is unambiguous (maps to the same canonical
    # name regardless of which type bucket produced it).
    raw_to_canon = {}
    ambiguous = set()
    for (original, t), cname in rename_map.items():
        if original in raw_to_canon and raw_to_canon[original] != cname:
            ambiguous.add(original)
        else:
            raw_to_canon[original] = cname
    for a in ambiguous:
        raw_to_canon.pop(a, None)

    rel_renamed = 0
    for s in sections:
        for r in s["graph"]["relations"]:
            for field in ("source", "target"):
                if r[field] in raw_to_canon:
                    r[field] = raw_to_canon[r[field]]
                    rel_renamed += 1

    dedup = 0
    for s in sections:
        g = s["graph"]
        seen = {}
        new_entities = []
        for e in g["entities"]:
            key = (e["name"], e["type"])
            if key not in seen:
                seen[key] = e
                new_entities.append(e)
            else:
                dedup += 1
                existing = seen[key]
                if len(e.get("mention", "")) > len(existing.get("mention", "")):
                    existing["mention"] = e["mention"]
                if "properties" in e and "properties" not in existing:
                    existing["properties"] = e["properties"]
        g["entities"] = new_entities

    return renamed, rel_renamed, dedup


def add_acronym_equivalences(sections):
    added = 0
    for s in sections:
        g = s["graph"]
        by_type = defaultdict(list)
        for e in g["entities"]:
            if e["type"] == "Requirement" or looks_like_id(e["name"]):
                continue
            by_type[e["type"]].append(e["name"])

        existing_pairs = {frozenset((r["source"], r["target"])) for r in g["relations"]}

        for names in by_type.values():
            names = sorted(set(names))
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    if norm(a) == norm(b):
                        continue
                    if not acronym_match(a, b):
                        continue
                    pair = frozenset((a, b))
                    if pair in existing_pairs:
                        continue
                    g["relations"].append({
                        "source": a,
                        "relation": "EQUIVALENT_TO",
                        "target": b,
                        "evidence": f"Graph refinement: '{a}' and '{b}' co-occur in this "
                                    f"section and one is the initial-letter acronym of the "
                                    f"other's significant words.",
                    })
                    existing_pairs.add(pair)
                    added += 1
    return added


def normalize_directions(sections):
    fixed = 0
    for s in sections:
        g = s["graph"]
        etype = {e["name"]: e["type"] for e in g["entities"]}
        for r in g["relations"]:
            rule = DIRECTION_RULES.get(r["relation"])
            if not rule:
                continue
            src_t, tgt_t = etype.get(r["source"]), etype.get(r["target"])
            if tgt_t not in rule["target_types"] and src_t in rule["target_types"]:
                r["source"], r["target"] = r["target"], r["source"]
                fixed += 1
    return fixed


def connect_orphans(sections):
    connected = 0
    for s in sections:
        g = s["graph"]
        entities = g["entities"]
        if len(entities) < 2:
            continue

        used = set()
        for r in g["relations"]:
            used.add(r["source"])
            used.add(r["target"])

        product_entities = [e["name"] for e in entities if e["type"] == "Product"]
        if product_entities:
            hub = product_entities[0]
        else:
            modules = [e["name"] for e in entities if e["type"] == "Module"]
            hub = modules[0] if modules else entities[0]["name"]

        for e in entities:
            if e["name"] == hub:
                continue
            if e["name"] not in used:
                g["relations"].append({
                    "source": e["name"],
                    "relation": "RELATED_TO",
                    "target": hub,
                    "evidence": f"Graph refinement: orphan entity connected to "
                                f"section hub '{hub}'.",
                })
                used.add(e["name"])
                connected += 1

        # the hub itself may remain disconnected (e.g. a Product entity
        # declared only as section metadata, with nothing pointing to it)
        if hub not in used:
            other = next((e["name"] for e in entities if e["name"] != hub), None)
            if other:
                g["relations"].append({
                    "source": hub,
                    "relation": "RELATED_TO",
                    "target": other,
                    "evidence": f"Graph refinement: section hub '{hub}' had no "
                                f"relations; linked to '{other}'.",
                })
                connected += 1
    return connected


def add_missing_module_edges(sections):
    added = 0
    for s in sections:
        g = s["graph"]
        modules = [e["name"] for e in g["entities"] if e["type"] == "Module"]
        products = [e["name"] for e in g["entities"] if e["type"] == "Product"]
        if not modules or not products:
            continue
        hub = products[0]
        linked = {r["source"] for r in g["relations"] if r["relation"] == "BELONGS_TO"}
        for m in modules:
            if m not in linked:
                g["relations"].append({
                    "source": m,
                    "relation": "BELONGS_TO",
                    "target": hub,
                    "evidence": f"Graph refinement: restored missing Module-to-Product "
                                f"edge for '{m}'.",
                })
                added += 1
    return added


def fetch_embeddings(driver, names):
    emb_map = {}
    if not names:
        return emb_map
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(
            "UNWIND $names AS nm MATCH (n) WHERE n.name = nm AND n.embedding IS NOT NULL "
            "RETURN n.name AS name, n.type AS type, n.embedding AS embedding",
            names=list(names),
        )
        for rec in result:
            emb_map[(rec["name"], rec["type"])] = rec["embedding"]
    return emb_map


def export_review_candidates(driver, sections, out_csv):
    """Embedding-similarity candidate pairs for a human to confirm.

    Never applied automatically -- see module docstring for why.
    """
    eligible = set()
    for s in sections:
        for e in s["graph"]["entities"]:
            if e["type"] == "Requirement" or looks_like_id(e["name"]):
                continue
            eligible.add((e["name"], e["type"]))

    emb_map = fetch_embeddings(driver, {n for n, _ in eligible})

    by_type = defaultdict(list)
    for name, typ in eligible:
        if (name, typ) in emb_map:
            by_type[typ].append(name)

    rows = []
    for typ, names in by_type.items():
        names = sorted(set(names))
        if len(names) < 2:
            continue
        vecs = np.array([emb_map[(n, typ)] for n in names])
        unit = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        sim = unit @ unit.T
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if norm(names[i]) == norm(names[j]):
                    continue
                score = float(sim[i, j])
                if score >= CANDIDATE_REVIEW_THRESHOLD:
                    rows.append((typ, names[i], names[j], score))

    rows.sort(key=lambda r: -r[3])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "entity_a", "entity_b", "cosine_similarity"])
        w.writerows(rows)
    return len(rows)


def refine_file(driver, filepath):
    sections = json.load(open(filepath))

    properties_fixed = fix_flattened_requirement_properties(sections)
    type_changed = reconcile_types(sections)
    renamed, rel_renamed, dedup = canonicalize_casing(sections)
    acronym_equivalences = add_acronym_equivalences(sections)
    entities_backfilled, relations_dropped = backfill_missing_entities(sections)
    direction_fixed = normalize_directions(sections)
    orphans_connected = connect_orphans(sections)
    module_edges_added = add_missing_module_edges(sections)

    out_path = filepath.replace(".json", "_refine.json")
    with open(out_path, "w") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

    candidates_csv = filepath.replace(".json", "_entity_candidates.csv")
    candidate_count = export_review_candidates(driver, sections, candidates_csv)

    return {
        "file": os.path.basename(filepath),
        "product": sections[0]["product"] if sections else None,
        "requirement_properties_fixed": properties_fixed,
        "type_reconciled": type_changed,
        "casing_renamed": renamed,
        "casing_relation_endpoints_renamed": rel_renamed,
        "casing_dedup": dedup,
        "acronym_equivalences_added": acronym_equivalences,
        "entities_backfilled": entities_backfilled,
        "malformed_relations_dropped": relations_dropped,
        "direction_fixed": direction_fixed,
        "orphans_connected": orphans_connected,
        "module_edges_added": module_edges_added,
        "out_path": out_path,
        "review_candidates_csv": candidates_csv,
        "review_candidate_count": candidate_count,
    }


def main():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    files = sorted(
        fp for fp in glob.glob(os.path.join(GRAPH_DIR, "*.json"))
        if not fp.endswith("_refine.json")
    )

    reports = []
    for fp in files:
        report = refine_file(driver, fp)
        reports.append(report)
        print(json.dumps(report, indent=2))

    driver.close()
    return reports


if __name__ == "__main__":
    main()

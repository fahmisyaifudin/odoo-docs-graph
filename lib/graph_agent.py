"""LangChain tool-calling agent that answers questions over the Neo4j knowledge graph."""

import json
from itertools import islice
from typing import Any, Dict, Literal, Optional

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from neo4j import READ_ACCESS
from pydantic import BaseModel, Field

from lib.context_builder import format_graph_as_cypher
from lib.graph_traversal import traverse_graph_from_nodes
from lib.prompt_to_cypher import is_safe_cypher

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_CYPHER_ROWS = 50
MAX_TOOL_CALLS = 8
RECURSION_LIMIT = 2 * MAX_TOOL_CALLS + 10
MAX_FALLBACK_EVIDENCE_CHARS = 40000


class GraphAnswer(BaseModel):
    """Final answer to the user's question, grounded in the knowledge graph."""
    answer: str = Field(description=(
        "Direct answer in plain, natural language for an end user. Never mention the graph, "
        "nodes, relationships, tables, rows, chunks, queries, scores, IDs or markdown formatting."
    ))
    confidence: Literal["high", "medium", "low"]
    reasoning: str = Field(description="Step-by-step explanation citing the evidence used (node names and paths, or document headings)")


SYSTEM_PROMPT = """You are a knowledge graph analyst answering questions about software products and their requirements.

The knowledge graph is stored in Neo4j. Every node has the label :Entity plus a type label (Feature, Actor, Requirement, Module, ...) and the properties name, type, mention, product, heading, section_id. Relationships carry an `evidence` property quoting the source document.

Graph context is written as Cypher patterns:
- NODES lists each node with its properties: (:Feature {name:"...", product:"...", heading:"..."})
- PATHS lists traversed edges: (:Actor {name:"A"})-[:PERFORMS {evidence:"..."}]->(:Feature {name:"B"})

How to work:
1. Read the initial graph context in the user message first.
2. If it is missing an entity, product or relationship needed to answer, use the tools:
   - search_graph: semantic search for other concepts in the question
   - expand_node: explore the neighbourhood of a specific node by exact name
   - get_graph_schema / read_cypher: precise lookups (counts, filters, specific paths)
3. Stop calling tools once you have enough evidence, then give the final answer.

Tool budget:
- You may make at most {max_tool_calls} tool calls in total. Prefer few, targeted calls over many exploratory ones.
- Do not repeat a tool call with the same or near-identical arguments.
- If after a few calls the evidence is still incomplete, stop searching and answer with what you have,
  stating what is missing and lowering confidence. A partial answer is always better than no answer.

Rules:
- Use ONLY information found in the graph context or tool results; never invent facts.
- If the graph does not contain enough information, say so and set confidence to low.
- Cite the node names and paths that support your answer in `reasoning`.

Answer style (the `answer` field only):
- Write it as a knowledgeable person explaining the product to a user, in plain natural sentences.
- Do NOT mention how you found the information: no "graph", "node", "relationship", "PERFORMED_BY",
  "property", "query", "Cypher", "table", "row", "context" or "document says".
- Do NOT use markdown (no **bold**, bullets or headings) and do not quote requirement IDs unless asked.
- Technical evidence (node names, relationships, queries) belongs in `reasoning`, not in `answer`.
""".replace("{max_tool_calls}", str(MAX_TOOL_CALLS))


FALLBACK_INSTRUCTION = """The tool budget is exhausted; no more tools can be called.
Answer the question NOW using only the initial context and the evidence gathered above.
If the evidence is incomplete, give the best partial answer you can, say what could not be confirmed,
and set confidence to "low" (or "medium" if the evidence mostly supports the answer)."""


DOCUMENT_SYSTEM_PROMPT = """You are an analyst answering questions about software products and their requirements.

You are given the top-ranked document chunks retrieved by semantic search, each with its product, module, heading, similarity score and content.

Rules:
- Use ONLY information found in the provided chunks; never invent facts.
- If the chunks do not contain enough information, say so and set confidence to low.
- In `reasoning`, cite each supporting chunk by its heading.

Answer style (the `answer` field only):
- Write it as a knowledgeable person explaining the product to a user, in plain natural sentences.
- Do NOT mention how you found the information: no "chunk", "document", "table", "row", "similarity",
  "score", "context" or "according to the documentation".
- Do NOT use markdown (no **bold**, bullets or headings) and do not quote requirement IDs unless asked.
- Evidence and citations belong in `reasoning`, not in `answer`.
"""


def _strip_embeddings(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_embeddings(v) for k, v in value.items() if k != "embedding"}
    if isinstance(value, list):
        return [_strip_embeddings(v) for v in value]
    return value


def build_graph_tools(qa, max_traversal_depth: int = 2) -> list:
    """Build tools bound to a QuestionAnswerer (driver, database, embedder)."""
    schema_cache: Dict[str, str] = {}

    @tool
    def search_graph(query: str, product: Optional[str] = None, top_k: int = 5) -> str:
        """Semantic search over the knowledge graph. Finds nodes similar to `query`
        (optionally only within one `product`) and returns them with their neighbourhood
        as Cypher patterns."""
        seeds = qa.search_similar_nodes(qa.get_query_embedding(query), top_k, product)
        if not seeds:
            return "No matching nodes found."
        graph_data = traverse_graph_from_nodes(
            qa.driver, qa.database, [s["node_id"] for s in seeds], max_depth=max_traversal_depth
        )
        return format_graph_as_cypher(graph_data)

    @tool
    def expand_node(name: str, depth: int = 1) -> str:
        """Return the neighbourhood of nodes whose name exactly matches `name`,
        up to `depth` hops (1-3), as Cypher patterns."""
        with qa.driver.session(database=qa.database) as session:
            ids = [r["id"] for r in session.run(
                "MATCH (n:Entity {name: $name}) RETURN elementId(n) AS id LIMIT 5", name=name
            )]
        if not ids:
            return f"No node named {name!r}. Use search_graph to find the exact name."
        graph_data = traverse_graph_from_nodes(
            qa.driver, qa.database, ids, max_depth=max(1, min(depth, 3))
        )
        return format_graph_as_cypher(graph_data)

    @tool
    def get_graph_schema() -> str:
        """Return node types, relationship patterns and property keys of the graph.
        Call this before writing a read_cypher query."""
        if "schema" not in schema_cache:
            with qa.driver.session(database=qa.database) as session:
                products = [r["p"] for r in session.run(
                    "MATCH (n:Entity) RETURN DISTINCT n.product AS p ORDER BY p"
                )]
                patterns = [
                    f"(:{r['src']})-[:{r['rel']}]->(:{r['dst']})"
                    for r in session.run(
                        "MATCH (a:Entity)-[r]->(b:Entity) "
                        "RETURN DISTINCT a.type AS src, type(r) AS rel, b.type AS dst "
                        "ORDER BY src, rel, dst"
                    )
                ]
            schema_cache["schema"] = "\n".join([
                "Node labels: :Entity plus one type label per node",
                "Node properties: name, type, mention, product, heading, section_id",
                "Relationship properties: relation, evidence, product, section_id",
                f"Products: {', '.join(p for p in products if p)}",
                "Relationship patterns:",
                *patterns,
            ])
        return schema_cache["schema"]

    @tool
    def read_cypher(query: str) -> str:
        """Run a read-only Cypher query (must start with MATCH/WITH/UNWIND/RETURN;
        CALL, CREATE, MERGE, SET, DELETE are rejected). Returns up to 50 rows as JSON."""
        if not is_safe_cypher(query):
            return "Rejected: only read-only MATCH ... RETURN queries are allowed (no CALL/CREATE/MERGE/SET/DELETE)."
        try:
            with qa.driver.session(database=qa.database, default_access_mode=READ_ACCESS) as session:
                rows = [_strip_embeddings(r.data()) for r in islice(session.run(query), MAX_CYPHER_ROWS)]
        except Exception as e:
            return f"Cypher error: {e}"
        return json.dumps(rows, ensure_ascii=False, default=str) if rows else "No rows returned."

    return [search_graph, expand_node, get_graph_schema, read_cypher]


def _sum_usage(messages: list) -> Dict[str, int]:
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for msg in messages:
        meta = getattr(msg, "usage_metadata", None) or {}
        usage["prompt_tokens"] += meta.get("input_tokens", 0)
        usage["completion_tokens"] += meta.get("output_tokens", 0)
        usage["total_tokens"] += meta.get("total_tokens", 0)
    return usage


def run_agent(
    question: str,
    context: str,
    model: str,
    api_key: str,
    system_prompt: str = SYSTEM_PROMPT,
    context_label: str = "Initial graph context (from semantic search + traversal)",
    tools: Optional[list] = None,
) -> Dict[str, Any]:
    llm = ChatOpenAI(
        model=model,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        temperature=0.3,
        max_tokens=6000,
    )
    agent = create_agent(
        llm,
        tools or [],
        system_prompt=system_prompt,
        response_format=ToolStrategy(GraphAnswer),
    )
    user_message = f"## Question\n{question}\n\n## {context_label}\n{context}"

    # Stream so the last state survives a GraphRecursionError.
    state: Dict[str, Any] = {}
    hit_limit = False
    try:
        for state in agent.stream(
            {"messages": [{"role": "user", "content": user_message}]},
            config={"recursion_limit": RECURSION_LIMIT},
            stream_mode="values",
        ):
            pass
    except GraphRecursionError:
        hit_limit = True
    except Exception as e:
        return {"success": False, "error": str(e), "usage": {}, "tool_calls": []}

    messages = state.get("messages", [])
    tool_calls = [
        {"name": call["name"], "args": call["args"]}
        for msg in messages if isinstance(msg, AIMessage)
        for call in msg.tool_calls
        if call["name"] != GraphAnswer.__name__
    ]
    structured = state.get("structured_response")
    usage = _sum_usage(messages)

    used_fallback = structured is None
    fallback_error = None
    if used_fallback:
        # Recursion limit hit (or agent stopped early): answer from the evidence gathered so far.
        try:
            structured, fallback_msg = _fallback_answer(llm, system_prompt, user_message, messages)
            for key, value in _sum_usage([fallback_msg]).items():
                usage[key] += value
        except Exception as e:
            fallback_error = str(e)

    return {
        "success": structured is not None,
        "llm_response": structured.model_dump() if structured else None,
        "usage": usage,
        "tool_calls": tool_calls,
        "fallback": used_fallback,
        "hit_recursion_limit": hit_limit,
        "error": None if structured else (
            fallback_error or "Agent finished without a structured answer"
        ),
    }


def _fallback_answer(llm: ChatOpenAI, system_prompt: str, user_message: str, messages: list):
    """Force a final GraphAnswer without tools, using tool results gathered so far as evidence."""
    evidence = [
        f"### {msg.name or 'tool'} result\n{msg.content}"
        for msg in messages if isinstance(msg, ToolMessage)
    ]
    # Keep the most recent evidence when it does not all fit.
    kept, total = [], 0
    for block in reversed(evidence):
        if total + len(block) > MAX_FALLBACK_EVIDENCE_CHARS:
            break
        kept.append(block)
        total += len(block)
    evidence_text = "\n\n".join(reversed(kept)) or "(no additional evidence was gathered)"

    prompt = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"{user_message}\n\n## Evidence gathered by tools\n{evidence_text}\n\n{FALLBACK_INSTRUCTION}"
        )),
    ]
    structured_llm = llm.with_structured_output(GraphAnswer, method="function_calling", include_raw=True)
    result = structured_llm.invoke(prompt)
    if result.get("parsing_error") or result.get("parsed") is None:
        raise ValueError(f"Fallback answer could not be parsed: {result.get('parsing_error')}")
    return result["parsed"], result["raw"]

#!/usr/bin/env python3
"""Run the yes/no question set through pgvector and neo4j concurrently.

For every question in the CSV, both retrieval methods are queried in parallel
using the same LLM agent. The agent's structured JSON answer is written per
method as ``{csv_stem}_{method}.json`` with one object per question:

    {"question": ..., "product": ..., "result": {"answer": ..., "confidence": ..., "reasoning": ...},
     "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}}

Usage (from the repository root):

    uv run python run_yes_no_questions.py
    uv run python run_yes_no_questions.py --limit 5
    uv run python run_yes_no_questions.py --model openai/gpt-oss-120b --top-k 5 --depth 1
"""

import argparse
import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

from dotenv import load_dotenv

from ask_question import ask_question, get_answerer

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = REPO_ROOT / "sql" / "requirement" / "question" / "cross_platforms.csv"
DEFAULT_MODEL = "google/gemma-4-26b-a4b-it"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "requirement" / "question" / "qwen06b" / DEFAULT_MODEL
DEFAULT_TOP_K = 5
DEFAULT_DEPTH = 1
DEFAULT_TIMEOUT = 300
METHODS = ("agent", "neo4j", "pgvector")


def read_questions(csv_path: Path, limit: int = None) -> list[dict]:
    """Read the question/product columns from the CSV, preserving order."""
    questions = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            question = (row.get("question") or "").strip()
            if not question:
                continue
            questions.append({
                "question": question,
                "product": (row.get("product") or "").strip(),
            })
            if limit and len(questions) >= limit:
                break
    return questions


def run_method(question: str, method: str, model: str, top_k: int, depth: int):
    """Answer one question with one method.

    Returns a ``(result, usage)`` tuple: the agent's JSON response (or an error
    object) and the token usage reported by the LLM calls.
    """
    try:
        answer = ask_question(
            question,
            model,
            method=method,
            top_k=top_k,
            max_traversal_depth=depth,
        )
    except Exception as e:  # a failed question shouldn't kill the run
        return {"error": f"{type(e).__name__}: {e}"}, {}

    usage = answer.get("usage") or {}
    if not answer.get("success"):
        return {"error": answer.get("error") or "unknown error"}, usage
    return answer.get("llm_reasoning"), usage


def write_json(path: Path, data: list[dict]) -> None:
    """Write the accumulated results atomically."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def summarize(result) -> str:
    if not isinstance(result, dict):
        return "invalid"
    if "error" in result:
        return "ERROR"
    return result.get("confidence", "ok")


def lock_embeddings(answerer) -> None:
    """Serialize MLX embedding calls shared by the two concurrent methods."""
    original = answerer.get_query_embedding
    lock = threading.Lock()

    def locked_embedding(text: str):
        with lock:
            return original(text)

    answerer.get_query_embedding = locked_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="Question CSV (default: sql/requirement/question/yes-no-question.csv)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for the JSON results")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"OpenRouter reasoning model (default: {DEFAULT_MODEL})")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="Similar nodes/chunks retrieved per method (default: 5)")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                        help="Graph traversal depth for neo4j (default: 1)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N questions (useful for a smoke test)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"Per-question, per-method timeout in seconds; a hung call is "
                             f"skipped as an error and the run moves on (default: {DEFAULT_TIMEOUT})")
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    questions = read_questions(args.csv, args.limit)
    if not questions:
        raise SystemExit(f"No questions found in {args.csv}")

    # Load the shared MLX model once, before any worker thread uses it.
    answerer = get_answerer()
    lock_embeddings(answerer)
    print("Warming up the embedding model...")
    answerer.get_query_embedding("warm-up")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        method: args.output_dir / f"{args.csv.stem}_{method}.json"
        for method in METHODS
    }
    results = {method: [] for method in METHODS}

    print(f"\nRunning {len(questions)} question(s) with pgvector and neo4j concurrently.")
    print(f"Model: {args.model} | top-k: {args.top_k} | neo4j depth: {args.depth} | "
          f"timeout: {args.timeout}s\n")

    for index, item in enumerate(questions, 1):
        # A fresh executor per row so a hung call (past --timeout) doesn't tie up
        # a worker and block later rows; the orphaned thread keeps running but is
        # simply abandoned (shutdown(wait=False)).
        executor = ThreadPoolExecutor(max_workers=len(METHODS))
        futures = {
            executor.submit(
                run_method,
                item["question"],
                method,
                args.model,
                args.top_k,
                args.depth,
            ): method
            for method in METHODS
        }

        row_results = {}
        for future, method in futures.items():
            try:
                row_results[method] = future.result(timeout=args.timeout)
            except FutureTimeoutError:
                row_results[method] = (
                    {"error": f"timed out after {args.timeout}s"}, {}
                )
        executor.shutdown(wait=False)

        for method in METHODS:
            result, usage = row_results[method]
            results[method].append({
                "question": item["question"],
                "product": item["product"],
                "result": result,
                "usage": usage,
            })

        for method in METHODS:
            write_json(output_paths[method], results[method])

        statuses = " ".join(
            f"{method}={summarize(row_results[method][0])}"
            f" ({row_results[method][1].get('total_tokens', 0)} tok)"
            for method in METHODS
        )
        print(f"[{index}/{len(questions)}] {item['product']} — {statuses}")

    print("\nDone.")
    for method, path in output_paths.items():
        print(f"  {method}: {path} ({len(results[method])} results)")


if __name__ == "__main__":
    main()

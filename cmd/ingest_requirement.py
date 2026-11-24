"""Embed requirement chunks from CSV and insert them into pgvector.

Reads every ``sql/requirement/*.csv`` (columns: product, module, heading,
content, embedding) and writes ``product, module, heading, content, embedding``
into a pgvector table, using either a local MLX model or OpenRouter's
embeddings endpoint.

Each chunk's embedding covers all metadata columns, not just content:

    Product: <product>
    Module: <module>
    Topic: <heading>

    <content>

Usage (from the repository root):

    # Local MLX: MiniLM (384 dims) -> minilm_l12_v2
    python cmd/ingest_requirement.py --replace

    # Local MLX: Qwen3-Embedding-0.6B (1024 dims) -> qwen_embedding_0_6b
    python cmd/ingest_requirement.py --backend mlx \
        --model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ \
        --table qwen_embedding_0_6b --dim 1024 --replace

    # OpenRouter: Qwen3-Embedding-8B (4096 dims) -> qwen_embedding_8b
    python cmd/ingest_requirement.py --backend openrouter \
        --model qwen/qwen3-embedding-8b \
        --table qwen_embedding_8b --dim 4096 --replace

    # OpenRouter: Qwen3-Embedding-4B (2560 dims) -> qwen_embedding_4b
    python cmd/ingest_requirement.py --backend openrouter \
        --model qwen/qwen3-embedding-4b \
        --table qwen_embedding_4b --dim 2560 --replace

    # OpenRouter: BGE-M3 (1024 dims) -> bge_m3
    python cmd/ingest_requirement.py --backend openrouter \
        --model baai/bge-m3 --table bge_m3 --dim 1024 --replace
"""

import argparse
import csv
import glob
import os
import re
import sys
import time
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from mlx_embeddings.utils import load
from openai import OpenAI

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
DEFAULT_TABLE = "minilm_l12_v2"
DEFAULT_DIM = 384
CSV_GLOB = "sql/requirement/*.csv"
INSERT_BATCH = 64
API_BATCH = 16
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mlx", "openrouter"], default="mlx",
                        help="Embed locally with MLX or via OpenRouter (default: mlx)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"MLX repo id or OpenRouter model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--table", default=DEFAULT_TABLE,
                        help=f"Target pgvector table (default: {DEFAULT_TABLE})")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM,
                        help=f"Expected embedding dimension (default: {DEFAULT_DIM})")
    parser.add_argument("--batch-size", type=int, default=API_BATCH,
                        help=f"Texts per OpenRouter request (default: {API_BATCH})")
    parser.add_argument("--replace", action="store_true",
                        help="Clear the target table before inserting")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.table):
        parser.error(f"invalid table name: {args.table}")
    return args


def read_rows() -> list[dict]:
    """Load all CSV rows that have content."""
    rows = []
    for path in sorted(glob.glob(str(REPO_ROOT / CSV_GLOB))):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for line_no, row in enumerate(reader, start=2):
                content = (row.get("content") or "").strip()
                if not content:
                    continue
                rows.append({
                    "product": (row.get("product") or "").strip(),
                    "module": (row.get("module") or "").strip(),
                    "heading": (row.get("heading") or "").strip(),
                    "content": content,
                    "source": f"{Path(path).name}:{line_no}",
                })
    return rows


def build_embedding_text(row: dict) -> str:
    """Text that gets embedded: product, module, topic, then the content."""
    return (
        f"Product: {row['product']}\n"
        f"Module: {row['module']}\n"
        f"Topic: {row['heading']}\n\n"
        f"{row['content']}"
    )


def connect():
    conn_str = os.getenv("PG_CONNECTION_STRING")
    if conn_str:
        return psycopg2.connect(conn_str)
    return psycopg2.connect(
        host="localhost", dbname="docs", user="postgres", password="password"
    )


def embed_with_openrouter(client: OpenAI, model: str, texts: list[str],
                          retries: int = 3) -> list[list[float]]:
    for attempt in range(retries):
        try:
            response = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in
                    sorted(response.data, key=lambda item: item.index)]
        except Exception:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  request failed, retrying in {wait}s...")
            time.sleep(wait)


def iter_embeddings(rows: list[dict], args: argparse.Namespace):
    """Yield one embedding per row, in order."""
    texts = [build_embedding_text(row) for row in rows]

    if args.backend == "mlx":
        model, tokenizer = load(args.model)
        for text in texts:
            outputs = model(tokenizer.encode(text, return_tensors="mlx"))
            yield outputs.text_embeds.tolist()[0]
        return

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY is not set")
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    for start in range(0, len(texts), args.batch_size):
        chunk = texts[start:start + args.batch_size]
        embeddings = embed_with_openrouter(client, args.model, chunk)
        if len(embeddings) != len(chunk):
            raise ValueError(
                f"expected {len(chunk)} embeddings, got {len(embeddings)}"
            )
        yield from embeddings


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    rows = read_rows()
    if not rows:
        sys.exit(f"No rows found in {CSV_GLOB}")

    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {args.table}")
            existing = cur.fetchone()[0]
        if existing and not args.replace:
            sys.exit(
                f"{args.table} already has {existing} rows; refusing to insert "
                f"duplicates. Re-run with --replace to clear and re-ingest."
            )
        if existing:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE {args.table}")
            conn.commit()
            print(f"Cleared {existing} existing rows from {args.table}")

        print(f"Loading {args.model} via {args.backend}...")
        print(
            f"Embedding and inserting {len(rows)} chunks into {args.table} "
            f"({args.dim} dims)..."
        )

        insert_sql = f"""
            INSERT INTO {args.table} (product, module, heading, content, embedding)
            VALUES (%s, %s, %s, %s, %s)
        """

        batch = []
        for i, (row, embedding) in enumerate(zip(rows, iter_embeddings(rows, args)), 1):
            if len(embedding) != args.dim:
                raise ValueError(
                    f"{row['source']}: expected {args.dim} dims, got {len(embedding)}"
                )

            batch.append((
                row["product"],
                row["module"],
                row["heading"],
                row["content"],
                embedding,
            ))

            if len(batch) >= INSERT_BATCH:
                with conn.cursor() as cur:
                    cur.executemany(insert_sql, batch)
                conn.commit()
                batch.clear()

            if i % 100 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}")

        if batch:
            with conn.cursor() as cur:
                cur.executemany(insert_sql, batch)
            conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                f"SELECT count(*), count(DISTINCT product) FROM {args.table}"
            )
            total, products = cur.fetchone()
        print(f"Done: {total} rows, {products} products in {args.table}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

"""PostgreSQL pgvector utility functions for vector search."""

import os
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor


def get_pg_connection(connection_string: Optional[str] = None):
    """Get a PostgreSQL connection."""
    conn_str = connection_string or os.getenv("PG_CONNECTION_STRING")
    if not conn_str:
        raise ValueError("PostgreSQL connection string not provided")
    return psycopg2.connect(conn_str)


def search_similar_documents(
    query_embedding: List[float],
    top_k: int = 5,
    table_name: str = "documents",
    embedding_column: str = "embedding",
    connection_string: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for similar documents using pgvector cosine similarity.
    
    Args:
        query_embedding: The embedding vector to search for
        top_k: Number of results to return
        table_name: Name of the table containing documents
        embedding_column: Name of the column containing embeddings
        connection_string: PostgreSQL connection string
    
    Returns:
        List of dictionaries containing document data and similarity scores
    """
    conn = get_pg_connection(connection_string)
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Use pgvector's <=> operator for cosine distance (1 - similarity)
            # Lower distance = higher similarity
            query = f"""
                SELECT 
                    heading,
                    module,
                    content,
                    1 - ({embedding_column} <=> %s::vector) as similarity
                FROM {table_name}
                ORDER BY {embedding_column} <=> %s::vector
                LIMIT %s
            """
            
            cur.execute(query, (query_embedding, query_embedding, top_k))
            results = cur.fetchall()
            
            # Convert to list of dicts
            return [dict(row) for row in results]
            
    finally:
        conn.close()


def build_context_from_pg_results(results: List[Dict[str, Any]]) -> str:
    """
    Build a readable text context from PostgreSQL pgvector search results.
    
    Args:
        results: List of document results from search_similar_documents()
    
    Returns:
        Formatted string context for LLM consumption
    """
    if not results:
        return "No results found from vector search."
    
    context_parts = []
    context_parts.append("=" * 70)
    context_parts.append("VECTOR SEARCH RESULTS (PostgreSQL pgvector)")
    context_parts.append("=" * 70)
    context_parts.append(f"Total Results: {len(results)}\n")
    
    for i, result in enumerate(results, 1):
        content = result.get("content", "N/A")
        similarity = result.get("similarity", 0)
        heading = result.get("heading", "N/A")
        module = result.get("module", "N/A")
        
        context_parts.append(f"--- Result {i} (Similarity: {similarity:.4f}) ---")        
        # Add metadata if available
        if module:
            context_parts.append("Module:")
            context_parts.append(f"{module}")

        if heading:
            context_parts.append("Heading:")
            context_parts.append(f"{heading}")
            
        context_parts.append(f"Content: {content}")
        context_parts.append("")
    
    context_parts.append("=" * 70)
    
    return "\n".join(context_parts)


def store_document(
    content: str,
    embedding: List[float],
    metadata: Optional[Dict[str, Any]] = None,
    table_name: str = "documents",
    connection_string: Optional[str] = None
) -> int:
    """
    Store a document with its embedding in PostgreSQL.
    
    Args:
        content: The document text content
        embedding: The vector embedding of the content
        metadata: Optional JSON metadata
        table_name: Name of the table to store in
        connection_string: PostgreSQL connection string
    
    Returns:
        The ID of the inserted document
    """
    import json
    
    conn = get_pg_connection(connection_string)
    
    try:
        with conn.cursor() as cur:
            query = f"""
                INSERT INTO {table_name} (content, embedding, metadata)
                VALUES (%s, %s::vector, %s)
                RETURNING id
            """
            
            cur.execute(query, (
                content,
                embedding,
                json.dumps(metadata) if metadata else None
            ))
            
            result = cur.fetchone()
            conn.commit()
            
            return result[0]
            
    finally:
        conn.close()

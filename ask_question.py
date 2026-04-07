"""Main module for question answering with Neo4j and LLM."""

import os
from neo4j import GraphDatabase
from openai import OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv

from lib.prompt_to_cypher import prompt_to_cypher, is_safe_cypher
from lib.neo4j_utils import (
    get_node_label,
    get_relation_types,
    get_node_schema,
    execute_cypher_query
)
from lib.context_builder import (
    build_context_from_graph,
    build_context_from_cypher_result,
    build_context_from_pg_results
)
from lib.llm_utils import generate_llm_reasoning, generate_direct_llm
from lib.pgvector_utils import search_similar_documents

load_dotenv()


class QuestionAnswerer:
    """Main class for answering questions using Neo4j and LLM."""
    
    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        openrouter_api_key: str = None,
        database: str = "pos",
        pg_connection_string: str = None,
        module: str = "Human Resource"
    ):
        # Neo4j connection
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(
                neo4j_user,
                neo4j_password
            )
        )
        self.database = database
        
        # PostgreSQL connection string for pgvector
        self.pg_connection_string = pg_connection_string
        
        # OpenAI client for OpenRouter
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key
        )

        self.module = module
        
        # Model configurations
        self.embedding_model = "qwen/qwen3-embedding-8b"
        self.reasoning_model = "google/gemma-3-27b-it"
        self.cypher_model = "deepseek/deepseek-v3.2"

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using the configured model."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def search_similar_nodes(
        self,
        module: str,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search Neo4j for nodes with similar embeddings filtered by module.

        Args:
            module: Required module name to filter results (e.g., "Point of Sales")
            query_embedding: The embedding vector to search for
            top_k: Number of results to return
        """
        query = """
        CALL db.index.vector.queryNodes('embedding_index', $top_k, $query_embedding)
        YIELD node, score
        WHERE node.module = $module
        RETURN node {
            .product,
            .section_id,
            .heading,
            .confidence,
            .name,
            .type,
            .mention,
            .module
        } as properties,
        score,
        elementId(node) as node_id
        ORDER BY score DESC
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                params = {
                    "query_embedding": query_embedding,
                    "top_k": top_k,
                    "module": module
                }
                result = session.run(query, **params)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Vector index error: {e}")
            return []

    def question_to_cypher(self, question: str) -> str:
        """Convert question to Cypher query using LLM."""
        node_labels = get_node_label(self.driver, self.database)
        relation_types = get_relation_types(self.driver, self.database)
        
        prompt = prompt_to_cypher(question, node_labels, relation_types)
        
        response = self.client.chat.completions.create(
            model=self.cypher_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        query = response.choices[0].message.content.strip()
        
        if not is_safe_cypher(query):
            raise ValueError(f"Unsafe query blocked:\n{query}")
        
        return query

    def ask(
        self,
        question: str,
        llm_reasoning_model: str = None,
        top_k: int = 5,
        max_traversal_depth: int = 2,
        method: str = "neo4j"
    ) -> Dict[str, Any]:
        """
        Answer a question using either vector search or Cypher query approach.
        
        Args:
            question: The user's question
            llm_reasoning_model: The LLM model to use for reasoning (defaults to self.reasoning_model)
            top_k: Number of similar nodes to retrieve
            max_traversal_depth: Graph traversal depth
            method: Search approach to use ("neo4j" or "pgvector")
        
        Returns:
            Dictionary with answer, reasoning, and metadata
        """
        # Use provided model or fall back to default
        reasoning_model = llm_reasoning_model or self.reasoning_model
        
        print(f"\n{'='*70}")
        print(f"🔍 QUESTION: {question}")
        print(f"🤖 MODEL: {reasoning_model}")
        print(f"{'='*70}\n")

        if method == "neo4j":
            return self._ask_with_graph_search(question, top_k, max_traversal_depth, reasoning_model)
        elif method == "pgvector":
            return self._ask_with_vector_search(question, 3, reasoning_model)
        elif method == "no-context":
            return self._ask_with_direct_llm(question, reasoning_model)
        else:
            raise ValueError(f"Invalid method: {method}")
        
    def _ask_with_direct_llm(self, question, reasoning_model=None):
        """
        Answer using direct LLM reasoning.
        """
        model = reasoning_model or self.reasoning_model
        
        print("[1/4] Generating LLM reasoning...")
        reasoning_result = generate_direct_llm(
            question=question,
            client=self.client,
            model=model
        )
        print("✓ Reasoning complete\n")
        
        return {
            "success": reasoning_result.get("success", False),
            "question": question,
            "llm_reasoning": reasoning_result.get("llm_response"),
            "usage": reasoning_result.get("usage")
        }

    def _ask_with_vector_search(self, question: str, top_k: int = 1, reasoning_model: str = None) -> Dict[str, Any]:
        """
        Answer using vector search on PostgreSQL pgvector.
        
        Steps:
        1. Embed the question
        2. Search top_k similar documents in pgvector
        3. Build context from results
        4. Generate LLM reasoning
        """
        model = reasoning_model or self.reasoning_model
        
        # Step 1: Embed the question
        print("[1/4] Embedding the question...")
        query_embedding = self.get_embedding(question)
        print(f"✓ Generated {len(query_embedding)}-dim embedding\n")
        
        # Step 2: Search top_k similar documents in pgvector
        print(f"[2/4] Searching top {top_k} similar documents in pgvector...")
        similar_docs = search_similar_documents(
            query_embedding=query_embedding,
            top_k=top_k,
            connection_string=self.pg_connection_string,
            table_name="qwen_embedding_8b"
        )
        print(f"✓ Found {len(similar_docs)} similar documents\n")
        
        if not similar_docs:
            return {
                "success": False,
                "error": "No similar documents found in pgvector",
                "question": question
            }
        
        # Step 3: Build context from pgvector results
        print("[3/4] Building context from pgvector results...")
        vector_context = build_context_from_pg_results(similar_docs)
        print("✓ Context built from pgvector results\n")
        
        # Step 4: Generate LLM reasoning
        print("[4/4] Generating LLM reasoning...")
        reasoning_result = generate_llm_reasoning(
            question=question,
            graph_context=vector_context,
            seed_results=similar_docs,
            client=self.client,
            model=model
        )
        print("✓ Reasoning complete\n")
        
        return {
            "success": reasoning_result.get("success", False),
            "question": question,
            "usage": reasoning_result.get("usage", {}),
            "llm_reasoning": reasoning_result.get("llm_response"),
            "similar_docs_count": len(similar_docs),
            "error": reasoning_result.get("error")
        }

    def _ask_with_graph_search(
        self,
        question: str,
        top_k: int,
        max_traversal_depth: int,
        reasoning_model: str = None
    ) -> Dict[str, Any]:
        """Answer using vector search and graph traversal."""
        model = reasoning_model or self.reasoning_model
        
        print("[1/5] Generating embedding for question...")
        query_embedding = self.get_embedding(question)
        print(f"✓ Generated {len(query_embedding)}-dim embedding\n")
        
        print(f"[2/5] Searching for top {top_k} similar nodes...")
        seed_results = self.search_similar_nodes(self.module, query_embedding, top_k)
        print(f"✓ Found {len(seed_results)} seed nodes\n")
        
        if not seed_results:
            return {"success": False, "error": "No similar nodes found"}
        
        print(f"[3/5] Traversing graph (depth={max_traversal_depth})...")
        from lib.graph_traversal import traverse_graph_from_nodes
        seed_node_ids = [r["node_id"] for r in seed_results]
        graph_data = traverse_graph_from_nodes(
            self.driver, self.database, seed_node_ids,
            max_depth=max_traversal_depth
        )
        print(f"✓ Found {len(graph_data.get('nodes', []))} nodes\n")
        
        print("[4/5] Building context from graph...")
        graph_context = build_context_from_graph(graph_data, module=self.module)
        print("✓ Context built\n")
        
        print("[5/5] Generating LLM reasoning...")
        reasoning_result = generate_llm_reasoning(
            question, graph_context, seed_results,
            self.client, model
        )
        print("✓ Reasoning complete\n")
        
        return {
            "success": reasoning_result.get("success", False),
            "question": question,
            "usage": reasoning_result.get("usage", {}),
            "llm_reasoning": reasoning_result.get("llm_response"),
            "error": reasoning_result.get("error")
        }

    def _ask_with_cypher(self, question: str) -> Dict[str, Any]:
        """Answer using Cypher query generation."""
        print("[1/4] Converting question to Cypher query...")
        cypher_query = self.question_to_cypher(question)
        print(f"✓ Generated Cypher query\n{cypher_query}\n")
        
        print("[2/4] Executing Cypher query...")
        cypher_result = execute_cypher_query(
            self.driver, self.database, cypher_query
        )
        print(f"✓ Cypher query execution complete\n")
        
        print("[3/4] Building context from Cypher results...")
        graph_context = build_context_from_cypher_result(cypher_result)
        print("✓ Context built\n")
        
        print("[4/4] Generating LLM reasoning...")
        reasoning_result = generate_llm_reasoning(
            question, graph_context, [],
            self.client, self.reasoning_model
        )
        print("✓ Reasoning complete\n")
        
        return {
            "success": reasoning_result.get("success", False),
            "question": question,
            "cypher_query": cypher_query,
            "llm_reasoning": reasoning_result.get("llm_response"),
            "error": reasoning_result.get("error")
        }


def ask_question(question: str, llm_reasoning_model: str, method: str = "neo4j") -> Dict[str, Any]:
    """Quick function to ask a question and get results."""
    answerer = QuestionAnswerer(
       neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
       neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
       neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
       openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
       database="pos",
       module="Human Resource",
       pg_connection_string=os.getenv("PG_CONNECTION_STRING", "postgresql://postgres:postgres@localhost:5432/docs")
    )
    try:
        return answerer.ask(question, llm_reasoning_model, method=method)
    finally:
        answerer.close()

if __name__ == "__main__":
    # Example usage
    question = "Can employees check in and out using the Attendances app??"
    result = ask_question(question, llm_reasoning_model="meta-llama/llama-3.1-8b-instruct", method="neo4j")
    print(result)

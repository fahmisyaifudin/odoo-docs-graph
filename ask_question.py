# ask_question.py - Complete implementation with Graph Traversal and LLM Reasoning
import os
import json
from neo4j import GraphDatabase
from openai import OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv
from lib.prompt_to_cypher import prompt_to_cypher, is_safe_cypher

load_dotenv()

class QuestionAnswerer:
    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        openrouter_api_key: str = None,
        database: str = "pos"
    ):
        # Neo4j connection
        self.driver = GraphDatabase.driver(
            neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(neo4j_user or os.getenv("NEO4J_USER", "neo4j"), 
                  neo4j_password or os.getenv("NEO4J_PASSWORD", "password"))
        )
        self.database = database
        
        # OpenAI client for OpenRouter
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        )
        
        # IMPORTANT: Use qwen3-embedding-8b (must match your embedding.py)
        self.embedding_model = "qwen/qwen3-embedding-8b"
        
        # LLM model for reasoning
        self.reasoning_model = "meta-llama/llama-3.1-8b-instruct"

        # Cypher generation model
        self.cypher_model = "deepseek/deepseek-v3.2"

    def close(self):
        self.driver.close()

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using qwen-embedding-8b"""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def search_similar_nodes(
        self, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search Neo4j for nodes with similar embeddings"""
        
        query = """
        CALL db.index.vector.queryNodes('embedding_index', $top_k, $query_embedding)
        YIELD node, score
        RETURN node {
            .product,
            .section_id,
            .heading,
            .confidence,
            .name,
            .type,
            .mention
        } as properties, 
        score,
        elementId(node) as node_id
        ORDER BY score DESC
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query, 
                    query_embedding=query_embedding, 
                    top_k=top_k
                )
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Vector index error: {e}")
            return []

    def traverse_graph_from_nodes(
        self, 
        seed_node_ids: List[str], 
        max_depth: int = 2,
        max_nodes: int = 50
    ) -> Dict[str, Any]:
        """
        Traverse graph starting from seed nodes to find related nodes and relationships.
        
        Args:
            seed_node_ids: List of node element IDs to start traversal from
            max_depth: How many hops to traverse (default 2)
            max_nodes: Maximum nodes to return (default 50)
        
        Returns:
            Dictionary containing:
            - nodes: All discovered nodes with properties
            - relationships: All relationships between discovered nodes
            - paths: Important paths from seed nodes
        """
        
        query = """
        // Start from seed nodes
        MATCH (seed)
        WHERE elementId(seed) IN $seed_node_ids
        
        // Traverse outward up to max_depth hops
        CALL apoc.path.expandConfig(seed, {
            minLevel: 0,
            maxLevel: $max_depth,
            limit: $max_nodes,
            uniqueness: "NODE_GLOBAL"
        }) YIELD path
        
        // Collect all nodes and relationships from paths
        WITH 
            [node in nodes(path) | {
                id: elementId(node),
                properties: properties(node)
            }] as path_nodes,
            [rel in relationships(path) | {
                source: elementId(startNode(rel)),
                target: elementId(endNode(rel)),
                type: type(rel),
                properties: properties(rel)
            }] as path_rels
        
        // Aggregate results
        RETURN 
            apoc.coll.toSet(apoc.coll.flatten(collect(path_nodes))) as nodes,
            apoc.coll.toSet(apoc.coll.flatten(collect(path_rels))) as relationships
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    seed_node_ids=seed_node_ids,
                    max_depth=max_depth,
                    max_nodes=max_nodes
                )
                record = result.single()
                
                if record:
                    return {
                        "nodes": record["nodes"],
                        "relationships": record["relationships"],
                        "seed_node_count": len(seed_node_ids)
                    }
                else:
                    return {
                        "nodes": [],
                        "relationships": [],
                        "seed_node_count": len(seed_node_ids)
                    }
        except Exception as e:
            print(f"Graph traversal error: {e}")
            # Fallback: Just return seed nodes without traversal
            return self._get_seed_nodes_only(seed_node_ids)

    def _get_seed_nodes_only(self, seed_node_ids: List[str]) -> Dict[str, Any]:
        """Fallback: Get just the seed nodes without traversal"""
        query = """
        MATCH (n)
        WHERE elementId(n) IN $seed_node_ids
        RETURN {
            id: elementId(n),
            properties: properties(n)
        } as node
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, seed_node_ids=seed_node_ids)
            nodes = [record["node"] for record in result]
            
            return {
                "nodes": nodes,
                "relationships": [],
                "seed_node_count": len(seed_node_ids),
                "note": "Graph traversal failed, returned seed nodes only"
            }

    def build_context_from_graph(self, graph_data: Dict[str, Any]) -> str:
        """
        Build a readable text context from graph traversal results
        for LLM consumption.
        """
        nodes = graph_data.get("nodes", [])
        relationships = graph_data.get("relationships", [])
        
        context_parts = []
        context_parts.append("=" * 60)
        context_parts.append("KNOWLEDGE GRAPH CONTEXT")
        context_parts.append("=" * 60)
        
        # Group nodes by type
        nodes_by_type = {}
        for node in nodes:
            node_type = node.get("properties", {}).get("type", "Unknown")
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)
        
        # Add nodes to context
        context_parts.append(f"\n📊 NODES DISCOVERED: {len(nodes)} total\n")
        
        for node_type, type_nodes in nodes_by_type.items():
            context_parts.append(f"\n--- {node_type.upper()} ({len(type_nodes)}) ---")
            for node in type_nodes[:5]:  # Limit to 5 per type
                props = node.get("properties", {})
                name = props.get("name", "N/A")
                heading = props.get("heading", "N/A")
                context_parts.append(f"  • {name}")
                if heading != name:
                    context_parts.append(f"    Section: {heading}")
        
        # Add relationships to context
        if relationships:
            context_parts.append(f"\n\n🔗 RELATIONSHIPS DISCOVERED: {len(relationships)}\n")
            for rel in relationships[:10]:  # Limit to 10 relationships
                rel_type = rel.get("type", "RELATED_TO")
                context_parts.append(f"  ({rel.get('source', '?')}) -[:{rel_type}]-> ({rel.get('target', '?')})")
        
        context_parts.append("\n" + "=" * 60)
        
        return "\n".join(context_parts)

    def build_context_from_cypher_result(self, cypher_result: Dict[str, Any]) -> str:
        """
        Build a readable text context from Cypher query execution results.
        
        Args:
            cypher_result: Result from execute_cypher_query containing:
                - success: bool
                - record_count: int
                - records: list of dictionaries
        
        Returns:
            Formatted string context for LLM consumption
        """
        if not cypher_result.get("success", False):
            error_msg = cypher_result.get("error", "Unknown error")
            return f"Error executing Cypher query: {error_msg}"
        
        records = cypher_result.get("records", [])
        record_count = cypher_result.get("record_count", len(records))
        
        if not records:
            return "No results found from Cypher query."
        
        context_parts = []
        context_parts.append("=" * 70)
        context_parts.append("CYHER QUERY RESULTS")
        context_parts.append("=" * 70)
        context_parts.append(f"Total Records: {record_count}\n")
        
        # Show first 10 records in detail
        display_limit = min(10, len(records))
        context_parts.append(f"First {display_limit} Records:\n")
        
        for i, record in enumerate(records[:display_limit], 1):
            context_parts.append(f"--- Record {i} ---")
            
            # Group fields by node alias if present
            node_fields = {}
            scalar_fields = {}
            
            for key, value in record.items():
                if '.' in key:
                    # Field from a node (e.g., "a.name")
                    parts = key.split('.')
                    node_alias = parts[0]
                    prop_name = '.'.join(parts[1:])
                    
                    if node_alias not in node_fields:
                        node_fields[node_alias] = {}
                    node_fields[node_alias][prop_name] = value
                else:
                    # Scalar field or relationship type
                    scalar_fields[key] = value
            
            # Output node fields first
            for node_alias, properties in node_fields.items():
                context_parts.append(f"  Node [{node_alias}]:")
                for prop_name, prop_value in properties.items():
                    # Truncate long values
                    value_str = str(prop_value)
                    if len(value_str) > 100:
                        value_str = value_str[:100] + "..."
                    context_parts.append(f"    {prop_name}: {value_str}")
            
            # Output scalar fields
            for key, value in scalar_fields.items():
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                context_parts.append(f"  {key}: {value_str}")
            
            context_parts.append("")
        
        if len(records) > display_limit:
            context_parts.append(f"... and {len(records) - display_limit} more records\n")
        
        # Add schema info for context
        if records:
            context_parts.append("=" * 70)
            context_parts.append("RECORD SCHEMA")
            context_parts.append("=" * 70)
            first_record = records[0]
            context_parts.append("Available fields in each record:")
            for key in first_record.keys():
                value = first_record[key]
                value_type = type(value).__name__ if value is not None else "None"
                context_parts.append(f"  - {key} ({value_type})")
        
        context_parts.append("\n" + "=" * 70)
        
        return "\n".join(context_parts)

    def execute_cypher_query(self, query: str) -> Dict[str, Any]:
        """
        Execute a Cypher query on the graph database.
        Returns a dictionary with 'records' key containing all query results.
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                # result.data() returns a list of dictionaries
                records = result.data()
                
                return {
                    "success": True,
                    "record_count": len(records),
                    "records": records
                }
                
        except Exception as e:
            print(f"✗ Cypher query execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "records": []
            }

    def ask(self, question: str, top_k: int = 5, max_traversal_depth: int = 2, using_cypher: bool = False) -> Dict[str, Any]:
        """
        Complete workflow:
        1. Embed question
        2. Search top-k similar nodes
        3. Traverse graph from seed nodes
        4. Build context
        5. Generate LLM reasoning
        6. Return comprehensive answer
        """
        print(f"\n{'='*70}")
        print(f"🔍 QUESTION: {question}")
        print(f"{'='*70}\n")

        if using_cypher is False:
        # Step 1: Generate embedding for the question
            print("[1/5] Generating embedding for question...")
            query_embedding = self.get_embedding(question)
            print(f"✓ Generated {len(query_embedding)}-dim embedding\n")
            
            # Step 2: Search for similar nodes (seed nodes)
            print(f"[2/5] Searching for top {top_k} similar nodes...")
            seed_results = self.search_similar_nodes(query_embedding, top_k)
            print(f"✓ Found {len(seed_results)} seed nodes\n")
            
            if not seed_results:
                return {
                    "success": False,
                    "error": "No similar nodes found in the database",
                    "question": question
                }
            
            # Step 3: Traverse graph from seed nodes
            print(f"[3/5] Traversing graph from seed nodes (depth={max_traversal_depth})...")
            seed_node_ids = [r["node_id"] for r in seed_results]
            graph_data = self.traverse_graph_from_nodes(
                seed_node_ids=seed_node_ids,
                max_depth=max_traversal_depth,
                max_nodes=50
            )
            total_nodes = len(graph_data.get("nodes", []))
            total_rels = len(graph_data.get("relationships", []))
            print(f"✓ Traversal complete: {total_nodes} nodes, {total_rels} relationships\n")
            
            # Step 4: Build context from graph
            print("[4/5] Building context from graph data...")
            graph_context = self.build_context_from_graph(graph_data)
            
            # Step 5: Generate LLM reasoning
            print("[5/5] Generating LLM reasoning...")
            reasoning_result = self.generate_llm_reasoning(
                question=question,
                graph_context=graph_context,
                seed_results=seed_results
            )
            
            if reasoning_result.get("success"):
                print("✓ LLM reasoning complete\n")
            else:
                print(f"⚠️ LLM reasoning failed: {reasoning_result.get('error')}\n")
        
            #Compile final result
            final_result = {
                "success": reasoning_result.get("success", False),
                "question": question,
                "llm_reasoning": reasoning_result.get("llm_response") if reasoning_result.get("success") else None,
                "error": reasoning_result.get("error") if not reasoning_result.get("success") else None
            }
            
            print(f"{'='*70}")
            print("✅ PROCESS COMPLETE")
            print(f"{'='*70}\n")
            
            return final_result
        else:
            print("[1/5] Converting question to Cypher query...")
            cypher_query = self.question_to_cypher(question)
            print(f"✓ Generated Cypher query\n{cypher_query}\n")
            print("[2/5] Executing Cypher query...")
            graph_data = self.execute_cypher_query(cypher_query)
            print(f"✓ Cypher query execution complete\n{graph_data}\n")
            print("[3/5] Building context from graph data...")
            graph_context = self.build_context_from_cypher_result(graph_data)
            print(f"✓ Context built from graph data\n{graph_context}\n")
            print("[4/5] Generating LLM reasoning...")
            reasoning_result = self.generate_llm_reasoning(
                question=question,
                graph_context=graph_context,
                seed_results=[]
            )
            print(f"✓ LLM reasoning complete\n{reasoning_result}\n")
            final_result = {
                "success": reasoning_result.get("success", False),
                "question": question,
                "llm_reasoning": reasoning_result.get("llm_response") if reasoning_result.get("success") else None,
                "error": reasoning_result.get("error") if not reasoning_result.get("success") else None
            }

    def close(self):
        self.driver.close()

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using qwen-embedding-8b"""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def get_node_label(self) -> str:
        """Get node labels from Neo4j"""
        query = """
        CALL db.labels() YIELD label
        RETURN label
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return "\n".join([f"- {record['label']}" for record in result])
    
    def get_relation_types(self) -> str:
        """Get relation types from Neo4j"""
        query = """
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN relationshipType
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return "\n".join([f"- {record['relationshipType']}" for record in result])
    
    def question_to_cypher(self, question: str) -> str:
        """Convert question to Cypher query"""
        node_labels = self.get_node_label()
        relation_types = self.get_relation_types()
        prompt = prompt_to_cypher(question, node_labels, relation_types)
        response = self.client.chat.completions.create(
            model=self.cypher_model,
            messages=[{"role": "user", "content": prompt}]
        )
        query = response.choices[0].message.content.strip()
        if not is_safe_cypher(query):
            raise ValueError(f"Unsafe query blocked:\n{query}")
        return query


    def search_similar_nodes(
        self, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search Neo4j for nodes with similar embeddings"""
        
        query = """
        CALL db.index.vector.queryNodes('embedding_index', $top_k, $query_embedding)
        YIELD node, score
        RETURN node {
            .product,
            .section_id,
            .heading,
            .confidence,
            .name,
            .type,
            .mention
        } as properties, 
        score,
        elementId(node) as node_id
        ORDER BY score DESC
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query, 
                    query_embedding=query_embedding, 
                    top_k=top_k
                )
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Vector index error: {e}")
            return []

    def traverse_graph_from_nodes(
        self, 
        seed_node_ids: List[str], 
        max_depth: int = 2,
        max_nodes: int = 50
    ) -> Dict[str, Any]:
        """
        Traverse graph starting from seed nodes to find related nodes and relationships.
        """
        
        query = """
        // Start from seed nodes
        MATCH (seed)
        WHERE elementId(seed) IN $seed_node_ids
        
        // Traverse outward up to max_depth hops
        CALL apoc.path.expandConfig(seed, {
            relationshipFilter: "RELATED_TO|PART_OF|DEPENDS_ON|HAS_FEATURE|BELONGS_TO",
            minLevel: 0,
            maxLevel: $max_depth,
            limit: $max_nodes,
            uniqueness: "NODE_GLOBAL"
        }) YIELD path
        
        // Collect all nodes and relationships from paths
        WITH 
            [node in nodes(path) | {
                id: elementId(node),
                properties: properties(node)
            }] as path_nodes,
            [rel in relationships(path) | {
                source: elementId(startNode(rel)),
                target: elementId(endNode(rel)),
                type: type(rel),
                properties: properties(rel)
            }] as path_rels
        
        // Aggregate results
        RETURN 
            apoc.coll.toSet(apoc.coll.flatten(collect(path_nodes))) as nodes,
            apoc.coll.toSet(apoc.coll.flatten(collect(path_rels))) as relationships
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    seed_node_ids=seed_node_ids,
                    max_depth=max_depth,
                    max_nodes=max_nodes
                )
                record = result.single()
                
                if record:
                    return {
                        "nodes": record["nodes"],
                        "relationships": record["relationships"],
                        "seed_node_count": len(seed_node_ids)
                    }
                else:
                    return {
                        "nodes": [],
                        "relationships": [],
                        "seed_node_count": len(seed_node_ids)
                    }
        except Exception as e:
            print(f"Graph traversal error: {e}")
            return self._get_seed_nodes_only(seed_node_ids)

    def _get_seed_nodes_only(self, seed_node_ids: List[str]) -> Dict[str, Any]:
        """Fallback: Get just the seed nodes without traversal"""
        query = """
        MATCH (n)
        WHERE elementId(n) IN $seed_node_ids
        RETURN {
            id: elementId(n),
            properties: properties(n)
        } as node
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, seed_node_ids=seed_node_ids)
            nodes = [record["node"] for record in result]
            
            return {
                "nodes": nodes,
                "relationships": [],
                "seed_node_count": len(seed_node_ids),
                "note": "Graph traversal failed, returned seed nodes only"
            }

    def build_context_from_graph(self, graph_data: Dict[str, Any]) -> str:
        """
        Build a readable text context from graph traversal results
        for LLM consumption.
        
        Formats relationships with evidence property clearly:
        [NodeA] --(RELATIONSHIP: evidence)--> [NodeB]
        """
        nodes = graph_data.get("nodes", [])
        relationships = graph_data.get("relationships", [])
        
        # Build node lookup dictionary
        node_lookup = {}
        for node in nodes:
            node_id = node.get("id")
            props = node.get("properties", {})
            node_lookup[node_id] = {
                "name": props.get("name", "Unknown"),
                "type": props.get("type", "Unknown"),
                "heading": props.get("heading", "")
            }
        
        context_parts = []
        context_parts.append("=" * 70)
        context_parts.append("KNOWLEDGE GRAPH CONTEXT")
        context_parts.append("=" * 70)
        
        # Add nodes summary by type
        nodes_by_type = {}
        for node in nodes:
            node_type = node.get("properties", {}).get("type", "Unknown")
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)
        
        context_parts.append(f"\n📊 NODES DISCOVERED: {len(nodes)} total\n")
        
        for node_type, type_nodes in nodes_by_type.items():
            context_parts.append(f"\n{'─' * 50}")
            context_parts.append(f"📁 {node_type.upper()} ({len(type_nodes)} items)")
            context_parts.append(f"{'─' * 50}")
            
            for node in type_nodes[:5]:
                props = node.get("properties", {})
                name = props.get("name", "N/A")
                heading = props.get("heading", "")
                
                context_parts.append(f"\n  ▸ {name}")
                if heading and heading != name:
                    context_parts.append(f"    Section: {heading}")
        
        # Add relationships with evidence
        if relationships:
            context_parts.append(f"\n\n{'─' * 50}")
            context_parts.append(f"🔗 RELATIONSHIPS DISCOVERED: {len(relationships)}")
            context_parts.append(f"{'─' * 50}")
            context_parts.append("\nFormat: [Node A] --(RELATIONSHIP: evidence)--> [Node B]\n")
            
            for i, rel in enumerate(relationships[:20], 1):
                rel_type = rel.get("type", "RELATED_TO")
                source_id = rel.get("source", "?")
                target_id = rel.get("target", "?")
                rel_props = rel.get("properties", {})
                evidence = rel_props.get("evidence", "")
                
                # Get node names from lookup
                source_node = node_lookup.get(source_id, {"name": f"Node({source_id[:8]}...)", "type": "Unknown"})
                target_node = node_lookup.get(target_id, {"name": f"Node({target_id[:8]}...)", "type": "Unknown"})
                
                # Format with evidence
                context_parts.append(f"{i}. [{source_node['type']}] {source_node['name']}")
                if evidence:
                    context_parts.append(f"   --({rel_type}: {evidence})-->")
                else:
                    context_parts.append(f"   --({rel_type})-->")
                context_parts.append(f"   [{target_node['type']}] {target_node['name']}")
                context_parts.append("")
        
        context_parts.append("\n" + "=" * 70)
        
        return "\n".join(context_parts)

    def generate_llm_reasoning(
        self, 
        question: str, 
        graph_context: str,
        seed_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Use LLM to generate reasoning based on graph traversal results.
        """
        
        # Build the system prompt
        system_prompt = """You are an expert Odoo consultant and knowledge graph analyst. Your task is to answer user questions about Odoo Point of Sales (POS) features based on the provided knowledge graph context.

## Your Capabilities:
1. Analyze the knowledge graph structure and relationships
2. Identify relevant features, modules, and configurations
3. Provide accurate, specific answers based on the graph data
4. Explain your reasoning clearly

## Response Format:
You must respond in this JSON structure:
{
    "answer": "Your direct answer to the question",
    "confidence": "high|medium|low",
    "reasoning": "Step-by-step explanation of how you arrived at the answer",
    "sources": [
        {
            "name": "Name of the node/feature",
            "type": "Feature|Module|etc",
            "relevance": "How this supports the answer"
        }
    ],
    "related_features": ["List of related features from the graph"],
    "limitations": "Any limitations or caveats to the answer, or 'None'"
}

## Important Rules:
1. ONLY use information from the provided context
2. If the context doesn't contain enough information, say so clearly
3. Be specific about which nodes/features support your answer
4. Reference the knowledge graph structure in your reasoning
5. Do not make up information not present in the context
"""

        # Build the user prompt with context
        user_prompt = f"""## User Question:
{question}

## Knowledge Graph Context:
{graph_context}

## Top Similar Nodes (Initial Search Results):
"""
        
        # Add seed results summary
        for i, result in enumerate(seed_results[:5], 1):
            props = result.get("properties", {})
            score = result.get("score", 0)
            user_prompt += f"\n{i}. {props.get('name', 'N/A')} (Score: {score:.4f})"
            user_prompt += f"\n   Type: {props.get('type', 'N/A')}"
            user_prompt += f"\n   Section: {props.get('heading', 'N/A')}"
        
        user_prompt += """

## Task:
Based on the knowledge graph context provided above, answer the user's question.
Provide your response in the required JSON format.
"""

        print("\n🤖 Sending to LLM for reasoning...")
        
        try:
            # Call the LLM
            response = self.client.chat.completions.create(
                model=self.reasoning_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            # Parse the response
            llm_response = response.choices[0].message.content
            
            # Try to parse as JSON
            try:
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = llm_response[json_start:json_end]
                    parsed_response = json.loads(json_str)
                else:
                    parsed_response = json.loads(llm_response)
                
                print("✓ LLM reasoning complete")
                return {
                    "success": True,
                    "llm_response": parsed_response,
                    "raw_response": llm_response
                }
                
            except json.JSONDecodeError as e:
                print(f"⚠️ Could not parse LLM response as JSON: {e}")
                return {
                    "success": False,
                    "error": "JSON parse error",
                    "raw_response": llm_response
                }
                
        except Exception as e:
            print(f"✗ LLM API error: {e}")
            return {
                "success": False,
                "error": str(e)
            }


def ask_question(question: str, top_k: int = 5) -> Dict[str, Any]:
    """Quick function to ask a question and get results"""
    answerer = QuestionAnswerer()
    try:
        return answerer.ask(question, top_k, 5, False)
    finally:
        answerer.close()


if __name__ == "__main__":
    # Example usage
    question = "Can a POS session in Odoo only be opened by the user who created it?"
    result = ask_question(question, top_k=5)
    print(result)

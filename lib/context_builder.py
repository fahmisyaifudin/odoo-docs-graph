"""Context builder functions for formatting graph data and query results."""

from typing import List, Dict, Any


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

    print(context_parts)
    
    return "\n".join(context_parts)

def build_context_from_graph(graph_data: Dict[str, Any]) -> str:
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
    # context_parts.append(f"\n📊 NODES DISCOVERED: {len(nodes)} total\n")
    
    # for node_type, type_nodes in nodes_by_type.items():
    #     context_parts.append(f"\n--- {node_type.upper()} ({len(type_nodes)}) ---")
    #     for node in type_nodes:  # Limit to 5 per type
    #         props = node.get("properties", {})
    #         name = props.get("name", "N/A")
    #         heading = props.get("heading", "N/A")
    #         context_parts.append(f"  • {name}")
    #         if heading != name:
    #             context_parts.append(f"    Section: {heading}")
    
     # Create node lookup for relationship formatting
    node_lookup = {}
    for node in nodes:
        node_id = node.get("id", "")
        props = node.get("properties", {})
        node_lookup[node_id] = {
            "type": props.get("type", "Unknown"),
            "name": props.get("name", "N/A")
        }

    # Add relationships to context
    if relationships:
        context_parts.append(f"\n\n🔗 RELATIONSHIPS DISCOVERED: {len(relationships)}\n")
        for rel in relationships:  # Limit to 10 relationships
            rel_type = rel.get("type", "RELATED_TO")
            source_id = rel.get("source", "?")
            target_id = rel.get("target", "?")
            source_info = node_lookup.get(source_id, {"type": "?", "name": source_id})
            target_info = node_lookup.get(target_id, {"type": "?", "name": target_id})
            source_str = f"{source_info['type']}:{source_info['name']}"
            target_str = f"{target_info['type']}:{target_info['name']}"
            rel_props = rel.get('properties', {})
            evidence = rel_props.get('evidence', rel.get('evidence', '?'))
            context_parts.append(
                f"({source_str}) -[:{rel_type}]-> ({target_str})\n"
                f"  Evidence: {evidence}"
            )
    
    context_parts.append("\n" + "=" * 60)
    
    return "\n".join(context_parts)


def build_context_from_cypher_result(cypher_result: Dict[str, Any]) -> str:
    """
    Build a readable text context from Cypher query execution results.
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


def format_node_as_text(node: Dict[str, Any]) -> str:
    """Format a single node as readable text."""
    props = node.get("properties", {})
    lines = []
    
    name = props.get("name", "N/A")
    node_type = props.get("type", "Unknown")
    
    lines.append(f"[{node_type}] {name}")
    
    # Add other properties
    for key, value in props.items():
        if key not in ["name", "type"] and value:
            lines.append(f"  {key}: {value}")
    
    return "\n".join(lines)


def format_relationship_as_text(rel: Dict[str, Any]) -> str:
    """Format a single relationship as readable text."""
    rel_type = rel.get("type", "RELATED_TO")
    source = rel.get("source", "?")
    target = rel.get("target", "?")
    props = rel.get("properties", {})
    
    lines = [f"({source}) -[:{rel_type}]-> ({target})"]
    
    # Add relationship properties if any
    for key, value in props.items():
        lines.append(f"  {key}: {value}")
    
    return "\n".join(lines)

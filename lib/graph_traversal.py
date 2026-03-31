"""Graph traversal functions for exploring Neo4j graph."""

from typing import List, Dict, Any
from neo4j import Driver


def traverse_graph_from_nodes(
    driver: Driver,
    database: str,
    seed_node_ids: List[str],
    max_depth: int = 2,
    max_nodes: int = 50
) -> Dict[str, Any]:
    """
    Traverse graph starting from seed nodes to find related nodes and relationships.
    
    Args:
        driver: Neo4j driver instance
        database: Database name
        seed_node_ids: List of node element IDs to start traversal from
        max_depth: How many hops to traverse (default 2)
        max_nodes: Maximum nodes to return (default 50)
    
    Returns:
        Dictionary containing nodes, relationships, and metadata
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
        with driver.session(database=database) as session:
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
        return _get_seed_nodes_only(driver, database, seed_node_ids)


def _get_seed_nodes_only(
    driver: Driver,
    database: str,
    seed_node_ids: List[str]
) -> Dict[str, Any]:
    """Fallback: Get just the seed nodes without traversal."""
    query = """
    MATCH (n)
    WHERE elementId(n) IN $seed_node_ids
    RETURN {
        id: elementId(n),
        properties: properties(n)
    } as node
    """
    
    with driver.session(database=database) as session:
        result = session.run(query, seed_node_ids=seed_node_ids)
        nodes = [record["node"] for record in result]
        
        return {
            "nodes": nodes,
            "relationships": [],
            "seed_node_count": len(seed_node_ids),
            "note": "Graph traversal failed, returned seed nodes only"
        }

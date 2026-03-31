"""Neo4j database utility functions."""

from typing import List, Dict, Any


def get_node_label(driver, database: str) -> str:
    """Get node labels from Neo4j"""
    query = """
    CALL db.labels() YIELD label
    RETURN label
    """
    with driver.session(database=database) as session:
        result = session.run(query)
        return "\n".join([f"- {record['label']}" for record in result])


def get_relation_types(driver, database: str) -> str:
    """Get relation types from Neo4j"""
    query = """
    CALL db.relationshipTypes() YIELD relationshipType
    RETURN relationshipType
    """
    with driver.session(database=database) as session:
        result = session.run(query)
        return "\n".join([f"- {record['relationshipType']}" for record in result])


def get_node_schema(driver, database: str) -> str:
    """Get node schema from Neo4j - returns properties of first node type only"""
    query = """
    CALL db.schema.nodeTypeProperties()
    YIELD nodeType, propertyName
    WITH nodeType, collect(DISTINCT propertyName) AS properties
    RETURN properties
    LIMIT 1
    """
    with driver.session(database=database) as session:
        result = session.run(query)
        record = result.single()
        if record:
            properties = record["properties"]
            return ", ".join(properties) if properties else ""
        return ""


def execute_cypher_query(driver, database: str, query: str) -> Dict[str, Any]:
    """
    Execute a Cypher query on the graph database.
    Returns a dictionary with 'records' key containing all query results.
    """
    try:
        with driver.session(database=database) as session:
            result = session.run(query)
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

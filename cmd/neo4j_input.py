import json
import os
from neo4j import GraphDatabase


def connect_to_neo4j(uri="bolt://localhost:7687", username="neo4j", password="password", database="neo4j"):
    """
    Connect to Neo4j database.
    
    Args:
        uri: Neo4j connection URI (default: bolt://localhost:7687)
        username: Neo4j username (default: neo4j)
        password: Neo4j password (default: password)
        database: Database name to connect to (default: "neo4j")
    
    Returns:
        driver: Neo4j driver instance
    """
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    # Verify connection to specific database
    with driver.session(database=database) as session:
        result = session.run("RETURN 1 as test")
        record = result.single()
        if record and record["test"] == 1:
            print(f"Successfully connected to database: {database}")
    
    return driver


def read_json_file(filepath):
    """
    Read a single JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        list: List of section data dictionaries
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Loaded: {filepath}")
            # Return the list directly (each JSON file contains a list of sections)
            return data if isinstance(data, list) else [data]
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []


def sanitize_label(label):
    """
    Sanitize label name for Neo4j (remove spaces, special chars).
    
    Args:
        label: Original label string
    
    Returns:
        str: Sanitized label safe for Neo4j
    """
    safe_label = ''.join(c if c.isalnum() else '_' for c in label)
    if safe_label and safe_label[0].isdigit():
        safe_label = 'Label_' + safe_label
    return safe_label


def create_nodes_and_relations(driver, json_data, database="neo4j"):
    """
    Create nodes and relationships in Neo4j from JSON data.
    
    Structure follows the build_embedding_records pattern:
    
    Nodes:
      - name: entity name
      - type: entity type
      - mention: original mention text
      - confidence: confidence level (high/medium/low)
      - product: product name, example "Odoo"
      - section_id: section identifier
      - module: module name, example "Point of Sales"
      - heading: section heading
    
    Relationships:
      - source: source entity name
      - target: target entity name
      - relation: relationship type
      - evidence: supporting evidence text
      - product: product name
      - section_id: section identifier
    
    Args:
        driver: Neo4j driver instance
        json_data: List of section data dictionaries
    """
    with driver.session(database=database) as session:
        # Clear existing data (optional - remove if you want to keep existing data)
        # session.run("MATCH (n) DETACH DELETE n")
        
        for section in json_data:
            # Extract section metadata
            section_id = section.get("section_id", "")
            heading = section.get("heading", "")
            product = section.get("product", "")
            graph = section.get("graph", {})
            
            # Create nodes (entities)
            if "entities" in graph:
                for entity in graph["entities"]:
                    entity_type = entity.get("type", "Unknown")
                    entity_name = entity.get("name", "")
                    
                    # Sanitize label for Neo4j
                    safe_label = sanitize_label(entity_type)
                    
                    # Build node creation query with dynamic label
                    create_node_query = f"""
                    MERGE (n:`{safe_label}` {{name: $name}})
                    SET n.type = $type,
                        n.mention = $mention,
                        n.module = $module,
                        n.confidence = $confidence,
                        n.product = $product,
                        n.section_id = $section_id,
                        n.heading = $heading
                    """
                    
                    session.run(
                        create_node_query,
                        name=entity_name,
                        type=entity_type,
                        mention=entity.get("mention", ""),
                        confidence=entity.get("confidence", ""),
                        product=product,
                        module="Point of Sales",
                        section_id=section_id,
                        heading=heading
                    )
            
            # Create relationships
            if "relations" in graph:
                for rel in graph["relations"]:
                    source_name = rel.get("source", "")
                    target_name = rel.get("target", "")
                    relation_type = rel.get("relation", "RELATES")
                    
                    # Sanitize relationship type for Neo4j
                    safe_rel_type = sanitize_label(relation_type)
                    
                    # Build relationship creation query with dynamic relationship type
                    create_rel_query = f"""
                    MATCH (source {{name: $source_name}})
                    MATCH (target {{name: $target_name}})
                    MERGE (source)-[r:`{safe_rel_type}`]->(target)
                    SET r.relation = $relation_type,
                        r.evidence = $evidence,
                        r.product = $product,
                        r.section_id = $section_id
                    """
                    
                    session.run(
                        create_rel_query,
                        source_name=source_name,
                        target_name=target_name,
                        relation_type=relation_type,
                        evidence=rel.get("evidence", ""),
                        product=product,
                        section_id=section_id
                    )
        
        print("Data successfully inserted into Neo4j!")


def main():
    """
    Main function to orchestrate the Neo4j data insertion process.
    """
    # Neo4j connection parameters
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "password"  # Change this to your actual password
    NEO4J_DATABASE = "pos"
    
    # Single JSON file to test (change this to test different files)
    OUTPUT_FILE = "output/0_questions_graph.json"
    OUTPUT_DIR = "output/point_of_sales"
    
    try:
        # Connect to Neo4j
        print(f"Connecting to Neo4j at {NEO4J_URI}...")
        driver = connect_to_neo4j(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)
        
        # Read single JSON file
        print(f"\nReading JSON file: '{OUTPUT_FILE}'...")
        json_data = []
        for filename in os.listdir(OUTPUT_DIR):
            if filename.endswith(".json"):
                file_data = read_json_file(os.path.join(OUTPUT_DIR, filename))
                json_data.extend(file_data)
        
        if not json_data:
            print("No data found to insert.")
            return
        
        print(f"Found {len(json_data)} sections to process.")
        
        # Create nodes and relationships
        print(f"\nInserting data into Neo4j...")
        create_nodes_and_relations(driver, json_data, NEO4J_DATABASE)
        
        print("\nDone!")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        if 'driver' in locals():
            driver.close()


if __name__ == "__main__":
    main()

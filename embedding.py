import os
from neo4j import GraphDatabase
from openai import OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv


load_dotenv()

class Neo4jEmbeddingProcessor:
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        openrouter_api_key: str,
        module: str,
        database: str = "pos"
    ):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.database = database
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        self.module = module

    def close(self):
        self.driver.close()

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        query = """
        MATCH (n)
        RETURN n
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            nodes = []
            for record in result:
                node = record["n"]
                nodes.append({
                    "id": node.id,
                    "element_id": node.element_id,
                    "properties": dict(node)
                })
            return nodes

    def format_embedding_text(self, module: str, properties: Dict[str, Any]) -> str:
        product = properties.get("product", "")
        node_type = properties.get("type", "")
        heading = properties.get("heading", "")
        name = properties.get("name", "")

        formatted_text = f"Product: {product}\n"
        formatted_text += f"Module: {module}\n"
        formatted_text += f"Type: {node_type}\n"
        formatted_text += f"Section: {heading}\n"
        formatted_text += f"Name: {name}"

        return formatted_text

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="qwen/qwen3-embedding-8b",
            input=text
        )
        return response.data[0].embedding

    def update_node_embedding(self, element_id: str, embedding: List[float]):
        query = """
        MATCH (n)
        WHERE elementId(n) = $element_id
        SET n.embedding = $embedding
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, element_id=element_id, embedding=embedding)

    def process_all_nodes(self):
        print("Fetching all nodes from Neo4j...")
        nodes = self.get_all_nodes()
        print(f"Found {len(nodes)} nodes to process")

        for i, node in enumerate(nodes, 1):
            try:
                properties = node["properties"]
                element_id = node["element_id"]

                print(f"\nProcessing node {i}/{len(nodes)}: {properties.get('name', 'Unknown')}")

                formatted_text = self.format_embedding_text(self.module, properties)
                print(f"Text for embedding:\n{formatted_text}")

                print("Generating embedding...")
                embedding = self.get_embedding(formatted_text)
                print(f"Generated embedding with {len(embedding)} dimensions")

                print("Updating node with embedding...")
                self.update_node_embedding(element_id, embedding)
                print("✓ Node updated successfully")

            except Exception as e:
                print(f"✗ Error processing node {i}: {str(e)}")
                continue

        print(f"\n{'='*50}")
        print(f"Processing complete! Processed {len(nodes)} nodes.")
        print(f"{'='*50}")


def main():
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")

    processor = Neo4jEmbeddingProcessor(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        openrouter_api_key=openrouter_api_key,
        module="Point of Sales",
        database="pos"
    )

    try:
        processor.process_all_nodes()
    finally:
        processor.close()


if __name__ == "__main__":
    main()

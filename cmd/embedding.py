import os
import sys
from pathlib import Path
from neo4j import GraphDatabase
from openai import OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# from lib.mlx_embedding import MLXEmbedding, DEFAULT_MLX_MODEL


load_dotenv()

VECTOR_INDEX_NAME = "entity_embedding"
EMBEDDING_DIM = 4096  # qwen/qwen3-embedding-8b
OPENROUTER_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"

class Neo4jEmbeddingProcessor:
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        database: str = "qwen06b",
        # mlx_model: str = DEFAULT_MLX_MODEL,
        openrouter_api_key: str = None,
        openrouter_model: str = OPENROUTER_EMBEDDING_MODEL,
    ):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.database = database
        # self.embedder = MLXEmbedding(model=mlx_model)
        self.embedding_model = openrouter_model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key or os.getenv("OPENROUTER_API_KEY"),
        )

    def close(self):
        self.driver.close()

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        query = """
        MATCH (n)
        WHERE n.embedding is null
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

    def format_embedding_text(self, properties: Dict[str, Any]) -> str:
        heading = properties.get("heading", "")
        name = properties.get("name", "")
        product = properties.get("product", "")

        formatted_text = f"Product: {product}\n"
        formatted_text += f"Section: {heading}\n"
        formatted_text += f"Name: {name}\n"

        return formatted_text

    def get_embedding(self, text: str) -> List[float]:
        # return self.embedder.document_embedding(text)
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
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

    def ensure_vector_index(self):
        # Vector indexes are single-label, so every node also carries :Entity.
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) WHERE NOT n:Entity SET n:Entity")
            session.run(f"""
            CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
            FOR (n:Entity) ON n.embedding
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {EMBEDDING_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
            """)
        print(f"✓ Vector index '{VECTOR_INDEX_NAME}' ready on :Entity(embedding)")

    def process_all_nodes(self):
        print("Fetching all nodes from Neo4j...")
        nodes = self.get_all_nodes()
        print(f"Found {len(nodes)} nodes to process")

        for i, node in enumerate(nodes, 1):
            try:
                properties = node["properties"]
                element_id = node["element_id"]

                print(f"\nProcessing node {i}/{len(nodes)}: {properties.get('name', 'Unknown')}")

                formatted_text = self.format_embedding_text(properties)
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
    neo4j_password = os.getenv("NEO4J_PASSWORD", "1234qwer")

    processor = Neo4jEmbeddingProcessor(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        database="qwen8b",
    )

    try:
        processor.process_all_nodes()
        processor.ensure_vector_index()
    finally:
        processor.close()


if __name__ == "__main__":
    main()

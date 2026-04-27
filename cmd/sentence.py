import re
import sys
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import psycopg2
from mlx_embeddings.utils import load


class MLXEmbedding:
    def __init__(self, model: str):
        self.model = model

    def document_embedding(self, document: str) -> list[float]:
        model, tokenizer = load(self.model)

        # Tokenize and generate embedding
        input_ids = tokenizer.encode(document, return_tensors="mlx")
        outputs = model(input_ids)
        raw_embeds = outputs.last_hidden_state[:, 0, :] # CLS token
        text_embeds = outputs.text_embeds # mean pooled and normalized embeddings
        return text_embeds.tolist()[0]


def parse_rst_sections(raw_rst: str) -> list[dict]:
    """
    Split RST content into sections by heading underlines.
    Returns list of {section_id, heading, raw_rst_content}
    """
    lines = raw_rst.splitlines()
    sections = []
    i = 0

    while i < len(lines):
        # Detect heading: line followed by underline of ===, ---, ~~~, ^^^
        if i + 1 < len(lines) and re.match(r'^[=\-~^]{3,}$', lines[i + 1].strip()):
            heading = lines[i].strip()
            underline_char = lines[i + 1][0]
            level = {'=': 1, '-': 2, '~': 3, '^': 4}.get(underline_char, 5)

            # Collect content until the next heading
            content_lines = []
            j = i + 2
            while j < len(lines):
                if j + 1 < len(lines) and re.match(r'^[=\-~^]{3,}$', lines[j + 1].strip()):
                    break
                content_lines.append(lines[j])
                j += 1

            section_id = heading.lower().replace(' ', '-').replace('/', '-')
            sections.append({
                "section_id": section_id,
                "heading": heading,
                "level": level,
                "raw_rst_content": "\n".join(content_lines).strip()
            })
            i = j
        else:
            i += 1

    return sections

def strip_rst_markup(text: str) -> str:
    """
    Light RST cleanup to reduce noise before sending to LLM.
    The LLM handles the rest — don't over-engineer this.
    """
    # Remove image directives and their options
    text = re.sub(r'\.\. image::.*?(?=\n\S|\Z)', '', text, flags=re.DOTALL)
    # Remove substitution definitions
    text = re.sub(r'\.\. \|.*?\|.*?\n', '', text)
    # Remove internal anchor labels
    text = re.sub(r'\.\. _[\w/\-]+:\n', '', text)
    # Extract clean text from roles: :guilabel:`X` → X
    text = re.sub(r':\w+:`([^`]+)`', r'\1', text)
    # Remove substitution references like |clock|
    text = re.sub(r'\|\w+\|', '', text)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # remove Bold/italic markup
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    return text.strip()

def build_chunk_text(
    product:    str,
    module:     str, 
    heading:    str,
    content:    str
) -> str:
    return (
        # f"Product: {product}\n"
        # f"Module: {module}\n"
        # f"Topic: {heading}\n\n"
        f"{content}"
    )

def build_module_from_path(rst_path: Path) -> str:
    parts = rst_path.parts

    if "applications" in parts:
        start_index = parts.index("applications") + 1
    elif "content" in parts:
        start_index = parts.index("content") + 1
    else:
        start_index = 0

    module_parts = list(parts[start_index:])
    if not module_parts:
        return ""

    module_parts[-1] = Path(module_parts[-1]).stem

    normalized_parts = []
    for index, part in enumerate(module_parts):
        normalized = re.sub(r"[_-]+", " ", part).strip()
        if index == len(module_parts) - 1 and " " not in normalized and normalized.endswith("s") and len(normalized) > 3:
            normalized = normalized[:-1]
        normalized_parts.append(normalized.title())

    return " > ".join(normalized_parts)

def embed(client, text: str) -> list[float]:
    # embedding = client.embeddings.create(
    #     model="qwen/qwen3-embedding-4b",
    #     input=text,
    #     encoding_format="float"
    # )
    # return embedding.data[0].embedding
    embedding = MLXEmbedding(model="sentence-transformers/all-MiniLM-L12-v2")
    return embedding.document_embedding(text)

def insert_chunk(conn, record: dict):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO minilm_l12_v2
                (product, module, heading, content, embedding)
            VALUES
                (%(product)s, %(module)s, %(heading)s, %(content)s, %(embedding)s)
        """, {**record, "embedding": record["embedding"]})
        conn.commit()

def main():
    load_dotenv()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    conn = psycopg2.connect(
        host="localhost",
        dbname="docs",
        user="postgres",
        password="password"
    )

    with open("filtered_rst_files.txt", "r") as f:
        rst_files = [line.strip() for line in f if line.strip()]
        for rst_file in rst_files:
            rst_path = Path(rst_file)
            rst_content = Path(rst_file).read_text(encoding="utf-8")
            sections = parse_rst_sections(rst_content)
            for section in sections:
                cleaned = strip_rst_markup(section["raw_rst_content"])
                chunk_text = build_chunk_text(
                    product="Odoo",
                    module=build_module_from_path(rst_path),
                    heading=section["heading"],
                    content=cleaned
                )
                if len(cleaned) < 5:
                    continue

                embedding = embed(client, chunk_text)
                insert_chunk(conn, {
                    "product": "Odoo",
                    "module": build_module_from_path(rst_path),
                    "heading": section["heading"],
                    "content": cleaned,
                    "embedding": embedding
                })

                #test _

if __name__ == "__main__":
    main()
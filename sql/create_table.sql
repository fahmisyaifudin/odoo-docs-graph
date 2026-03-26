CREATE TABLE IF NOT EXISTS qwen_embedding_8b (
    product TEXT,
    module TEXT,
    heading TEXT,
    content TEXT,
    embedding VECTOR(4096)
);
CREATE TABLE IF NOT EXISTS qwen_embedding_8b (
    product TEXT,
    module TEXT,
    heading TEXT,
    content TEXT,
    embedding VECTOR(4096)
);

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE TABLE IF NOT EXISTS question (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    question TEXT,
    module VARCHAR(255),
    answer INTEGER
)

CREATE TABLE IF NOT EXISTS answer_result (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    question_id uuid REFERENCES question(id),
    llm_reasoning_model varchar(255),
    method varchar(255),
    embedding_model varchar(255),
    is_correct BOOLEAN,
    result TEXT
)
CREATE EXTENSION IF NOT EXISTS vector;

-- Qwen/Qwen3-Embedding-8B
CREATE TABLE IF NOT EXISTS qwen_embedding_8b (
    product TEXT,
    module TEXT,
    heading TEXT,
    content TEXT,
    embedding VECTOR(4096)
);

-- Qwen/Qwen3-Embedding-4B
CREATE TABLE IF NOT EXISTS qwen_embedding_4b (
    product TEXT,
    module TEXT,
    heading TEXT,
    content TEXT,
    embedding VECTOR(2560)
);

-- Qwen/Qwen3-Embedding-0.6B
CREATE TABLE IF NOT EXISTS qwen_embedding_0_6b (
    product TEXT,
    module TEXT,
    heading TEXT,
    content TEXT,
    embedding VECTOR(1024)
);

-- sentence-transformers/all-MiniLM-L12-v2
CREATE TABLE IF NOT EXISTS minilm_l12_v2 (
    product TEXT,
    module TEXT,
    heading TEXT,
    content TEXT,
    embedding VECTOR(384)
);

-- BAAI/bge-m3
CREATE TABLE IF NOT EXISTS bge_m3 (
    product TEXT,
    module TEXT,
    heading TEXT,
    content TEXT,
    embedding VECTOR(1024)
);

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE TABLE IF NOT EXISTS question (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    question TEXT,
    module VARCHAR(255),
    answer INTEGER
);

CREATE TABLE IF NOT EXISTS answer_result (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    question_id uuid REFERENCES question(id),
    llm_reasoning_model varchar(255),
    method varchar(255),
    embedding_model varchar(255),
    is_correct BOOLEAN,
    result TEXT
);

ALTER TABLE question
    ADD COLUMN is_hard_negative BOOLEAN DEFAULT FALSE;

ALTER TABLE answer_result
    ADD COLUMN token_usage BIGINT DEFAULT 0
    
ALTER TABLE answer_result
    ADD COLUMN confidence VARCHAR(255);

ALTER TABLE question
    ADD COLUMN answer_text TEXT;

ALTER TABLE question
    ADD COLUMN is_open_ended BOOLEAN DEFAULT FALSE;

ALTER TABLE question
    ADD COLUMN is_multi_hop BOOLEAN DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS documents (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    product VARCHAR(255),
    document TEXT    
);

CREATE TABLE IF NOT EXISTS question_documents (
    question_id uuid REFERENCES question(id),
    document_id uuid REFERENCES documents(id),
    PRIMARY KEY (question_id, document_id)
);

ALTER TABLE question
    ADD COLUMN is_relationship_discovery BOOLEAN DEFAULT FALSE;

ALTER TABLE question
    ADD COLUMN is_impact_analysis BOOLEAN DEFAULT FALSE;

ALTER TABLE question
    ADD COLUMN is_cross_platform BOOLEAN DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS ragas_evaluation (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    answer_result_id uuid REFERENCES answer_result(id),
    question_id uuid REFERENCES question(id),
    method VARCHAR(255),
    embedding_model VARCHAR(255),
    llm_reasoning_model VARCHAR(255),
    faithfulness FLOAT,
    answer_relevancy FLOAT,
    answer_correctness FLOAT,
    context_precision FLOAT,
    context_recall FLOAT,
    evaluated_at TIMESTAMP DEFAULT now()
);
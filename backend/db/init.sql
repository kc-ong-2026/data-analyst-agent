-- PostgreSQL initialization script for RAG system
-- Uses pgvector extension for semantic search

CREATE EXTENSION IF NOT EXISTS vector;

-- Dataset metadata: one row per ingested file
CREATE TABLE IF NOT EXISTS dataset_metadata (
    id SERIAL PRIMARY KEY,
    dataset_name TEXT NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    category TEXT,
    description TEXT,
    columns JSONB DEFAULT '[]'::jsonb,
    row_count INTEGER DEFAULT 0,
    year_range JSONB DEFAULT '{}'::jsonb,
    dimensions JSONB DEFAULT '{}'::jsonb,
    summary_text TEXT,
    embedding vector(1536),
    tsv tsvector,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Data chunks: group-level summaries for semantic search
CREATE TABLE IF NOT EXISTS data_chunks (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES dataset_metadata(id) ON DELETE CASCADE,
    chunk_type TEXT NOT NULL DEFAULT 'group',
    group_key JSONB DEFAULT '{}'::jsonb,
    chunk_text TEXT NOT NULL,
    row_count INTEGER DEFAULT 0,
    numeric_summary JSONB DEFAULT '{}'::jsonb,
    embedding vector(1536),
    tsv tsvector,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Raw data tables registry: tracks dynamically created tables
CREATE TABLE IF NOT EXISTS raw_data_tables (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES dataset_metadata(id) ON DELETE CASCADE,
    table_name TEXT NOT NULL UNIQUE,
    columns JSONB DEFAULT '[]'::jsonb,
    row_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for full-text search
CREATE INDEX IF NOT EXISTS idx_dataset_metadata_tsv ON dataset_metadata USING GIN (tsv);
CREATE INDEX IF NOT EXISTS idx_data_chunks_tsv ON data_chunks USING GIN (tsv);

-- Indexes for JSONB queries
CREATE INDEX IF NOT EXISTS idx_dataset_metadata_dimensions ON dataset_metadata USING GIN (dimensions);
CREATE INDEX IF NOT EXISTS idx_data_chunks_group_key ON data_chunks USING GIN (group_key);

-- Indexes for foreign keys
CREATE INDEX IF NOT EXISTS idx_data_chunks_dataset_id ON data_chunks (dataset_id);
CREATE INDEX IF NOT EXISTS idx_raw_data_tables_dataset_id ON raw_data_tables (dataset_id);

-- Category index for filtering
CREATE INDEX IF NOT EXISTS idx_dataset_metadata_category ON dataset_metadata (category);

-- Note: IVFFlat vector indexes are created after data ingestion
-- because they require existing data to build the index properly.
-- See: backend/app/services/ingestion/embedding_generator.py

-- PostgreSQL initialization script for folder-based RAG system
-- Uses pgvector extension for semantic search

CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- FOLDER METADATA TABLES (4 tables, one per category)
-- ============================================================================
-- Each table stores metadata for files within a category folder
-- Embeddings enable semantic search at folder level
-- Schema info enables LLM SQL generation for data tables

-- Employment dataset metadata
CREATE TABLE IF NOT EXISTS employment_dataset_metadata (
    id SERIAL PRIMARY KEY,
    file_name TEXT NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    description TEXT,
    table_name TEXT NOT NULL UNIQUE,  -- Links to data table (e.g., employment_resident_employment_rate_by_age_and_sex)

    -- Schema information for SQL generation
    columns JSONB NOT NULL,              -- [{name, dtype, sample_values, nullable}]
    primary_dimensions JSONB,            -- [year, sex, age, etc.]
    numeric_columns JSONB,               -- [employment_rate, income, etc.]
    categorical_columns JSONB,           -- [{column, cardinality, values}]

    row_count INTEGER DEFAULT 0,
    year_range JSONB,                    -- {min, max}

    -- Search fields
    summary_text TEXT NOT NULL,
    embedding vector(1536),
    tsv tsvector,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Hours worked dataset metadata
CREATE TABLE IF NOT EXISTS hours_worked_dataset_metadata (
    id SERIAL PRIMARY KEY,
    file_name TEXT NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    description TEXT,
    table_name TEXT NOT NULL UNIQUE,

    columns JSONB NOT NULL,
    primary_dimensions JSONB,
    numeric_columns JSONB,
    categorical_columns JSONB,

    row_count INTEGER DEFAULT 0,
    year_range JSONB,

    summary_text TEXT NOT NULL,
    embedding vector(1536),
    tsv tsvector,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Income dataset metadata
CREATE TABLE IF NOT EXISTS income_dataset_metadata (
    id SERIAL PRIMARY KEY,
    file_name TEXT NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    description TEXT,
    table_name TEXT NOT NULL UNIQUE,

    columns JSONB NOT NULL,
    primary_dimensions JSONB,
    numeric_columns JSONB,
    categorical_columns JSONB,

    row_count INTEGER DEFAULT 0,
    year_range JSONB,

    summary_text TEXT NOT NULL,
    embedding vector(1536),
    tsv tsvector,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Labour force dataset metadata
CREATE TABLE IF NOT EXISTS labour_force_dataset_metadata (
    id SERIAL PRIMARY KEY,
    file_name TEXT NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    description TEXT,
    table_name TEXT NOT NULL UNIQUE,

    columns JSONB NOT NULL,
    primary_dimensions JSONB,
    numeric_columns JSONB,
    categorical_columns JSONB,

    row_count INTEGER DEFAULT 0,
    year_range JSONB,

    summary_text TEXT NOT NULL,
    embedding vector(1536),
    tsv tsvector,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- DATA TABLE REGISTRY
-- ============================================================================
-- Tracks all dynamically created data tables (one per CSV/Excel file)
-- Used for SQL generation and validation

CREATE TABLE IF NOT EXISTS data_table_registry (
    id SERIAL PRIMARY KEY,
    category TEXT NOT NULL,              -- employment | hours_worked | income | labour_force
    metadata_table TEXT NOT NULL,        -- e.g., 'employment_dataset_metadata'
    metadata_id INTEGER NOT NULL,
    data_table TEXT NOT NULL UNIQUE,     -- e.g., 'employment_resident_employment_rate_by_age_and_sex'
    schema_info JSONB NOT NULL,          -- Full schema for SQL generation
    index_info JSONB,                    -- Created indexes
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- INDEXES FOR HYBRID SEARCH
-- ============================================================================

-- Full-text search indexes (GIN)
CREATE INDEX IF NOT EXISTS idx_employment_metadata_tsv
    ON employment_dataset_metadata USING GIN (tsv);
CREATE INDEX IF NOT EXISTS idx_hours_worked_metadata_tsv
    ON hours_worked_dataset_metadata USING GIN (tsv);
CREATE INDEX IF NOT EXISTS idx_income_metadata_tsv
    ON income_dataset_metadata USING GIN (tsv);
CREATE INDEX IF NOT EXISTS idx_labour_force_metadata_tsv
    ON labour_force_dataset_metadata USING GIN (tsv);

-- JSONB indexes for dimension filtering
CREATE INDEX IF NOT EXISTS idx_employment_metadata_dimensions
    ON employment_dataset_metadata USING GIN (primary_dimensions);
CREATE INDEX IF NOT EXISTS idx_hours_worked_metadata_dimensions
    ON hours_worked_dataset_metadata USING GIN (primary_dimensions);
CREATE INDEX IF NOT EXISTS idx_income_metadata_dimensions
    ON income_dataset_metadata USING GIN (primary_dimensions);
CREATE INDEX IF NOT EXISTS idx_labour_force_metadata_dimensions
    ON labour_force_dataset_metadata USING GIN (primary_dimensions);

-- Registry indexes
CREATE INDEX IF NOT EXISTS idx_data_table_registry_category
    ON data_table_registry (category);
CREATE INDEX IF NOT EXISTS idx_data_table_registry_metadata_table
    ON data_table_registry (metadata_table);

-- ============================================================================
-- VECTOR INDEXES (Created after ingestion)
-- ============================================================================
-- Note: IVFFlat vector indexes are created after data ingestion
-- because they require existing data to build the index properly.
-- See: backend/app/services/ingestion/embedding_generator.py
--
-- These indexes will be created programmatically:
-- - idx_employment_metadata_embedding
-- - idx_hours_worked_metadata_embedding
-- - idx_income_metadata_embedding
-- - idx_labour_force_metadata_embedding

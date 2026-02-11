# Testing Quick Start Guide

## Prerequisites

1. **Docker containers running:**
   ```bash
   make dev
   ```

2. **API keys configured in `backend/.env`:**
   ```bash
   # Verify .env file exists
   cat backend/.env | grep API_KEY

   # Should show:
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   ```

3. **Database populated with data:**
   ```bash
   # Run data ingestion
   make ingest

   # Verify data loaded
   make db-shell
   SELECT COUNT(*) FROM income_dataset_metadata;
   \q
   ```

## Running Tests

### Quick Commands

```bash
# All tests
make test

# Unit tests only (fast)
make test-unit

# Integration tests
make test-integration

# Evaluation tests (requires API keys + data)
make test-evaluation

# Specific evaluation tests
make test-evaluation-retrieval    # Retrieval metrics only
make test-evaluation-generation   # Generation metrics only
```

### Evaluation Test Status

#### Fixed Issues ✅
- RAGService interface calls corrected
- Agent import paths fixed
- Test fixtures updated
- Mock datasets created
- Makefile commands added

#### Test Database & Auto-Ingestion ✅

**How It Works**:
1. Test session starts
2. `setup_test_database` creates database schema
3. `ingest_test_data` fixture checks if data exists
4. If empty, runs full ingestion pipeline automatically
5. `sample_datasets` provides data to evaluation tests
6. All tests run with real data

**Configuration**:
- Default: Uses production database (`govtech_rag`) in Docker
- Override: Set `TEST_DATABASE_URL` env var for isolated test database
- Auto-ingestion runs once per test session (cached)

**First Run**:
```bash
# Initial setup
make dev  # Start containers

# Run evaluation tests (auto-ingests if needed)
make test-evaluation-retrieval   # ~30s first run, instant afterwards
make test-evaluation-generation  # Uses same ingested data
```

**Performance**:
- First run: ~30-60 seconds (includes ingestion)
- Subsequent runs: <5 seconds (data already exists)
- Data persists across test runs
- To force re-ingestion: Drop test database and recreate

**Manual Reset** (if needed):
```bash
# Clear test data
docker-compose exec backend python -c "
from sqlalchemy import create_engine, text
import os
engine = create_engine(os.getenv('DATABASE_URL').replace('asyncpg', 'psycopg2'))
with engine.begin() as conn:
    conn.execute(text('TRUNCATE employment_dataset_metadata CASCADE'))
    conn.execute(text('TRUNCATE income_dataset_metadata CASCADE'))
    conn.execute(text('TRUNCATE hours_worked_dataset_metadata CASCADE'))
"

# Next test run will auto-ingest fresh data
make test-evaluation
```

## Test Structure

```
backend/tests/
├── unit/                    # Unit tests (fast, no DB)
├── integration/             # Integration tests (with DB)
├── evaluation/              # Evaluation metrics tests
│   ├── test_retrieval_metrics.py    # RAG retrieval quality
│   └── test_generation_metrics.py   # LLM generation quality
├── fixtures/                # Test data
│   ├── sample_queries.json
│   ├── ground_truth_contexts.json
│   ├── ground_truth_answers.json
│   └── mock_datasets/       # Small CSV files for testing
└── utils/                   # Test utilities
    ├── ragas_evaluator.py
    └── bertscore_evaluator.py
```

## Evaluation Metrics

### Retrieval (test_retrieval_metrics.py)
- Context Precision @ 1, 3, 5, 10
- Context Recall
- Mean Reciprocal Rank (MRR)
- Hybrid vs Vector comparison
- Reranking effectiveness
- Confidence threshold validation

### Generation (test_generation_metrics.py)
- BERTScore (Precision, Recall, F1)
- Ragas Faithfulness (hallucination detection)
- Answer Relevancy (query-answer alignment)
- Answer Correctness (factual accuracy)
- Agent-specific quality metrics

## Exporting Results for Fine-Tuning

```bash
# Export evaluation results (uses mock data)
make test-evaluation-export

# Or run with real evaluation:
docker-compose exec backend python scripts/run_evaluation.py \
  --export-format openai \
  --output-dir ./fine_tuning_data \
  --min-score 0.80
```

**Output Files:**
- `evaluation_results/training_data_openai.jsonl` - OpenAI fine-tuning format
- `evaluation_results/summary_report.json` - Aggregate metrics
- `evaluation_results/evaluation_results.csv` - Detailed breakdown

## Troubleshooting

### Tests skip with "No datasets found"
```bash
# Solution: Run data ingestion
make ingest

# Verify data loaded
make db-shell
SELECT COUNT(*) FROM income_dataset_metadata;
```

### Ragas import error
```
TypeError: <class 'langchain_core.output_parsers.pydantic.PydanticOutputParser'> is not a generic class
```

**Solution:** Update ragas version in `backend/requirements.txt` to `>=0.2.0`

### "API key not found" errors
```bash
# Solution: Verify .env file exists
cat backend/.env | grep API_KEY

# If missing, add keys:
echo "OPENAI_API_KEY=sk-..." >> backend/.env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> backend/.env

# Restart containers
make down
make dev
```

### Tests are slow
- **Expected:** Generation tests make LLM API calls (5-15 minutes)
- **Cost:** Each full evaluation run may cost $1-5 in API usage
- **Optimization:** Run specific test classes instead of full suite

## Next Steps

1. **Update Ragas:** Change `requirements.txt` to use ragas 0.2.x
2. **Rebuild:** Run `make build` to update dependencies
3. **Ingest Data:** Ensure `make ingest` completes successfully
4. **Run Evaluation:** Execute `make test-evaluation`
5. **Analyze Results:** Review generated metrics and export data

For detailed information, see `backend/EVALUATION_SETUP.md`.

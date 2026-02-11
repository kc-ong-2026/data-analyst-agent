# Backend Testing Suite

Comprehensive testing infrastructure for the multi-agent RAG system with evaluation metrics.

> **📖 For usage instructions, see [TESTING.md](../TESTING.md)** — Complete guide on running tests, collecting evaluation metrics, and exporting for fine-tuning.
>
> This README focuses on the technical implementation details of the test suite itself.

## 📁 Directory Structure

```
tests/
├── README.md                          # This file
├── IMPLEMENTATION_GUIDE.md            # Guide for completing remaining tests
├── conftest.py                        # Pytest fixtures and configuration
├── pytest.ini                         # Pytest settings (in backend/)
├── test_config.yaml                   # Test-specific configuration
│
├── unit/                              # Fast, isolated unit tests
│   ├── test_rag_retrieval.py         # ✅ RAG search tests
│   ├── test_verification_agent.py     # ✅ Verification agent tests
│   ├── test_coordinator_agent.py      # TODO: Coordinator agent tests
│   ├── test_extraction_agent.py       # TODO: Extraction agent tests
│   └── test_analytics_agent.py        # TODO: Analytics agent tests
│
├── integration/                       # Multi-component integration tests
│   ├── test_orchestrator.py          # TODO: Multi-agent workflow tests
│   ├── test_rag_pipeline.py          # TODO: End-to-end RAG tests
│   ├── test_llm_resilience.py        # TODO: Fallback/circuit breaker tests
│   └── test_performance.py           # TODO: Latency and load tests
│
├── evaluation/                        # Quality evaluation with metrics
│   ├── test_retrieval_metrics.py     # TODO: Ragas retrieval evaluation
│   ├── test_generation_metrics.py    # TODO: BERTScore + Ragas generation
│   └── test_llm_as_judge.py          # TODO: LLM-based quality evaluation
│
├── fixtures/                          # Test data and ground truth
│   ├── sample_queries.json           # ✅ Test queries with metadata
│   ├── ground_truth_contexts.json    # ✅ Expected retrieval results
│   ├── ground_truth_answers.json     # ✅ Expected agent responses
│   └── mock_datasets/                # (Create as needed)
│
├── utils/                             # Testing utilities and evaluators
│   ├── ragas_evaluator.py            # ✅ Ragas metric wrapper
│   ├── bertscore_evaluator.py        # ✅ BERTScore wrapper
│   ├── llm_judge.py                  # ✅ LLM as judge implementation
│   ├── test_helpers.py               # ✅ Common test utilities
│   └── test_reporter.py              # ✅ Report generation
│
└── reports/                           # Generated test reports
    └── (auto-generated)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
# Copy .env.example and add API keys
cp .env.example .env

# Add required keys:
# - OPENAI_API_KEY
# - ANTHROPIC_API_KEY
# - DATABASE_URL (for integration tests)
```

### 3. Run Tests

```bash
# Run all tests
pytest -v

# Run specific test category
pytest -m unit              # Unit tests only (fast)
pytest -m integration       # Integration tests (requires DB)
pytest -m evaluation        # Evaluation tests (requires LLM, slow)
pytest -m performance       # Performance tests

# Run with coverage
pytest --cov=app --cov-report=html --cov-report=term

# Skip slow tests (for local dev)
pytest -m "not slow"

# Skip LLM tests (no API calls)
pytest -m "not requires_llm"
```

## 📊 Test Categories

### Unit Tests (`@pytest.mark.unit`)
- **Fast**: < 1 second per test
- **Isolated**: Mock external dependencies
- **Coverage**: Focus on individual components
- **Run frequency**: Every commit

**Examples**:
- RAG search algorithm correctness
- Agent logic validation
- Data processing functions

### Integration Tests (`@pytest.mark.integration`)
- **Slower**: 1-30 seconds per test
- **Real services**: Uses actual DB, embeddings
- **End-to-end**: Tests component interactions
- **Run frequency**: Every pull request

**Examples**:
- Multi-agent workflow
- RAG pipeline (embedding → search → rerank)
- Database queries

### Evaluation Tests (`@pytest.mark.evaluation`)
- **Slowest**: 30+ seconds per test
- **Metrics-based**: Quantitative quality assessment
- **LLM-powered**: Uses Ragas, BERTScore, LLM judge
- **Run frequency**: Nightly or weekly

**Examples**:
- Context precision/recall
- Answer faithfulness
- LLM-as-judge quality scores

### Performance Tests (`@pytest.mark.performance`)
- **Latency**: Measures response times
- **Throughput**: Tests concurrent load
- **Benchmarking**: Tracks regression
- **Run frequency**: Before releases

**Examples**:
- RAG retrieval < 2s (p50)
- End-to-end < 30s (p50)
- 10 concurrent queries without degradation

## 📈 Evaluation Metrics

### Retrieval Quality (Ragas)
| Metric | Description | Threshold |
|--------|-------------|-----------|
| **Context Precision @ 1** | Top result relevance | ≥ 0.85 |
| **Context Precision @ 3** | Top 3 results relevance | ≥ 0.75 |
| **Context Recall** | Completeness of retrieval | ≥ 0.90 |
| **MRR** | Mean Reciprocal Rank | ≥ 0.80 |
| **NDCG** | Ranking quality | ≥ 0.75 |

### Generation Quality
| Metric | Description | Threshold |
|--------|-------------|-----------|
| **BERTScore F1** | Semantic similarity | ≥ 0.75 |
| **Faithfulness** | Groundedness in context | ≥ 0.85 |
| **Answer Relevancy** | Query relevance | ≥ 0.80 |
| **Answer Correctness** | Factual accuracy | ≥ 0.75 |

### LLM as Judge (1-5 scale)
| Criterion | Threshold |
|-----------|-----------|
| **Overall** | ≥ 3.5 |
| **Accuracy** | ≥ 4.0 |
| **Completeness** | ≥ 3.5 |
| **Clarity** | ≥ 3.5 |
| **Conciseness** | ≥ 3.0 |

### Performance
| Metric | Threshold |
|--------|-----------|
| RAG Retrieval (p50) | ≤ 2s |
| RAG Retrieval (p90) | ≤ 4s |
| Code Generation (p50) | ≤ 10s |
| End-to-End (p50) | ≤ 30s |

## 🔧 Configuration

### Test Config (`test_config.yaml`)

```yaml
testing:
  use_mock_llm: false           # Set true for fast tests without LLM calls
  use_local_embeddings: true    # Use cached embeddings
  ragas:
    enabled: true
    sample_size: 50             # Number of test cases
    metrics: [context_precision, context_recall, ...]
  bertscore:
    enabled: true
    model: "microsoft/deberta-xlarge-mnli"
  llm_judge:
    enabled: true
    model: "claude-sonnet-4-5"
```

### Pytest Config (`pytest.ini`)

```ini
[pytest]
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (requires external services)
    evaluation: Evaluation tests with metrics
    performance: Performance and load tests
    slow: Tests that take > 1 second
    requires_db: Tests requiring database
    requires_llm: Tests requiring LLM API calls
```

## 📝 Writing Tests

### Example Unit Test

```python
import pytest

@pytest.mark.unit
class TestMyComponent:
    @pytest.mark.asyncio
    async def test_functionality(self, mock_service):
        # Arrange
        input_data = {"query": "test"}

        # Act
        result = await my_component.process(input_data)

        # Assert
        assert result["success"] is True
        assert "output" in result
```

### Example Evaluation Test

```python
import pytest

@pytest.mark.evaluation
@pytest.mark.slow
@pytest.mark.requires_llm
class TestRetrievalQuality:
    def test_context_precision(
        self,
        ragas_evaluator,
        sample_queries,
        ground_truth_contexts,
    ):
        # Run retrieval on test queries
        results = run_retrieval(sample_queries)

        # Evaluate with Ragas
        metrics = ragas_evaluator.evaluate_retrieval(
            queries=sample_queries["queries"],
            retrieved_contexts=results,
            ground_truth_contexts=ground_truth_contexts,
        )

        # Assert meets threshold
        assert metrics.context_precision >= 0.85
```

## 📊 Generating Reports

```python
from tests.utils.test_reporter import TestReporter, TestSummary

reporter = TestReporter(Path("tests/reports"))

report = reporter.generate_report(
    test_summary=TestSummary(...),
    retrieval_metrics=RetrievalMetrics(...),
    generation_metrics=GenerationMetrics(...),
    llm_judge_metrics=LLMJudgeMetrics(...),
    performance_metrics=PerformanceMetrics(...),
)

# Generates both markdown and JSON reports
print(f"Report: {report}")
```

## 🎯 Coverage Goals

| Component | Target |
|-----------|--------|
| **Overall** | ≥ 80% |
| **RAG Service** | ≥ 85% |
| **Agents** | ≥ 80% |
| **Orchestrator** | ≥ 90% |

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
- name: Run tests with coverage
  run: pytest --cov=app --cov-report=xml --cov-report=term

- name: Run evaluation tests
  run: pytest -m evaluation -v
  if: github.event_name == 'schedule'  # Nightly

- name: Generate test report
  run: python -m tests.utils.test_reporter
```

## 🐛 Debugging Tests

```bash
# Run single test with verbose output
pytest tests/unit/test_rag_retrieval.py::TestVectorSearch::test_search -vv

# Run with print statements visible
pytest -s

# Run with debugger on failure
pytest --pdb

# Show 10 slowest tests
pytest --durations=10
```

## 📚 Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Ragas Documentation](https://docs.ragas.io/)
- [BERTScore Paper](https://arxiv.org/abs/1904.09675)
- [Testing LLM Applications](https://www.anthropic.com/research/testing-llms)

## 🚦 Status

- ✅ **Infrastructure**: Complete (fixtures, utilities, config)
- ✅ **Unit Tests**: 2/5 complete (RAG retrieval, verification agent)
- ⏳ **Integration Tests**: 0/4 complete
- ⏳ **Evaluation Tests**: 0/3 complete
- ⏳ **Performance Tests**: 0/1 complete

See `IMPLEMENTATION_GUIDE.md` for completing remaining tests.

## 💡 Tips

1. **Start with unit tests** - fastest feedback loop
2. **Mock expensive operations** - LLM calls, embeddings
3. **Cache what you can** - embeddings, BM25 indexes
4. **Run evaluation tests nightly** - they're slow but valuable
5. **Track metrics over time** - use comparison reports
6. **Set realistic thresholds** - based on baseline runs
7. **Use fixtures liberally** - DRY principle for test setup

## 🤝 Contributing

When adding new tests:
1. Follow naming convention: `test_<component>_<functionality>`
2. Add appropriate markers (`@pytest.mark.unit`, etc.)
3. Update this README if adding new test categories
4. Keep tests independent (no shared state)
5. Use descriptive assertion messages

## 📧 Questions?

See `IMPLEMENTATION_GUIDE.md` for detailed templates and examples.

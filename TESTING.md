# Testing and Evaluation Guide

This guide explains how to run tests, collect evaluation metrics, and export scores for LLM fine-tuning in the multi-agent data analytics system.

> **📚 This document explains overall testing philosophy, metric definitions, thresholds, and hallucination detection methodology.
>
> This document focuses on **practical usage** — running tests, collecting scores, and exporting for fine-tuning.

## Table of Contents

1. [Overview](#overview)
2. [Running Tests](#running-tests)
   - [Backend Tests](#backend-tests-python-pytest)
   - [Frontend Tests](#frontend-tests-javascript-jest)
3. [Evaluation Metrics](#evaluation-metrics)
4. [Fine-Tuning Data Collection](#fine-tuning-data-collection)
5. [LangSmith Integration](#langsmith-integration)
6. [Evaluation Results](#evaluation-results)
7. [Interpreting Results](#interpreting-results)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Reference Commands](#reference-commands)

---

## Overview

### Testing Philosophy

The system uses a **four-tier testing approach**:

1. **Unit Tests** - Fast, isolated tests of individual components (< 100ms per test)
2. **Integration Tests** - Tests of multiple components working together (requires DB, < 1s per test)
3. **Security Tests** - Tests of code validation, sandboxing, and audit logging (< 1s per test)
4. **Evaluation Tests** - Quality metrics for LLM outputs (requires LLM API, 1-10s per test)

### Quick Start

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit           # Fast unit tests only
pytest -m security       # Security tests only
pytest -m evaluation     # Evaluation metrics only

# Run with coverage report
pytest --cov=app --cov-report=html

# Run evaluations and export for fine-tuning
python scripts/run_evaluation.py --export-format openai --output-dir ./fine_tuning_data
```

---

## Running Tests

This project includes comprehensive testing for both backend (Python/pytest) and frontend (TypeScript/Jest) components.

### Backend Tests (Python + pytest)

#### Basic Commands

```bash
# Run all backend tests
cd backend
pytest

# Run specific test markers
pytest -m unit                # Unit tests only (~100 tests, <10s)
pytest -m integration         # Integration tests (~50 tests, <1min)
pytest -m security            # Security tests (~30 tests, <30s)
pytest -m evaluation          # Evaluation tests (~20 tests, 2-5min)
pytest -m performance         # Performance benchmarks (~10 tests, 1-2min)

# Run specific test files
pytest tests/security/test_code_validator.py
pytest tests/security/test_sandbox_executor.py
pytest tests/security/test_security_integration.py
pytest tests/evaluation/test_retrieval_metrics.py
pytest tests/evaluation/test_generation_metrics.py
pytest tests/evaluation/test_llm_as_judge.py

# Run specific test functions
pytest tests/evaluation/test_retrieval_metrics.py::TestContextPrecision::test_precision_at_1_on_sample_queries
```

### Advanced Options

```bash
# Run with verbose output
pytest -vv

# Run slow tests (tests marked with @pytest.mark.slow)
pytest -m slow

# Skip slow tests
pytest -m "not slow"

# Run tests requiring LLM
pytest -m requires_llm

# Skip LLM tests (useful for fast CI)
pytest -m "not requires_llm"

# Parallel execution (faster on multi-core systems)
pytest -n auto

# Stop after first failure
pytest -x

# Stop after N failures
pytest --maxfail=3

# Run only tests that failed last time
pytest --lf

# Run tests that failed last time, then all others
pytest --ff

# Show local variables on failure
pytest -l

# Drop into debugger on failure
pytest --pdb
```

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=app --cov-report=html

# Open coverage report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux

# Generate terminal coverage report
pytest --cov=app --cov-report=term-missing

# Fail if coverage below threshold (80%)
pytest --cov=app --cov-fail-under=80
```

---

## Evaluation Metrics

The system uses three complementary evaluation frameworks to measure quality.

### 1. Ragas Metrics (RAG System Evaluation)

**Purpose**: Evaluate retrieval and generation quality in RAG pipeline.

**Metrics**:

| Metric | Description | Threshold | What It Measures |
|--------|-------------|-----------|------------------|
| **Context Precision** | Relevance of retrieved documents | 0.85 | Are retrieved contexts relevant to the query? |
| **Context Recall** | Completeness of retrieval | 0.90 | Are all necessary contexts retrieved? |
| **Faithfulness** | Grounding of answer in contexts | 0.85 | Is the answer based on retrieved data (no hallucinations)? |
| **Answer Relevancy** | Relevance of answer to query | 0.80 | Does the answer address what was asked? |
| **Answer Correctness** | Factual accuracy vs ground truth | 0.75 | Is the answer factually correct? |

**When to Use**:
- Testing RAG retrieval quality
- Validating that LLM doesn't hallucinate
- Comparing retrieval strategies (vector vs hybrid search)

**Example**:
```python
from tests.utils.ragas_evaluator import RagasEvaluator

evaluator = RagasEvaluator(config)
result = evaluator.evaluate_end_to_end(
    queries=["What is the average salary in 2023?"],
    answers=["The average monthly salary in Singapore in 2023 was $5,197."],
    contexts=[["Singapore average monthly salary: $5,197 (2023)"]],
    ground_truth_answers=["$5,197"]
)

print(f"Faithfulness: {result.faithfulness}")  # 0.95 = excellent
print(f"Answer Correctness: {result.answer_correctness}")  # 0.98 = excellent
```

### 2. BERTScore (Semantic Similarity)

**Purpose**: Measure semantic similarity between generated and reference text.

**Metrics**:

| Metric | Description | Threshold | What It Measures |
|--------|-------------|-----------|------------------|
| **Precision** | Relevance of generated tokens | 0.70 | How much of the generated text is relevant? |
| **Recall** | Coverage of reference tokens | 0.70 | How much of the reference is covered? |
| **F1** | Harmonic mean | 0.75 | Overall semantic similarity |

**When to Use**:
- Comparing generated text to reference outputs
- Measuring semantic equivalence (handles paraphrasing)
- Validating consistency across model updates

**Model**: Uses `microsoft/deberta-xlarge-mnli` for best accuracy (can switch to `bert-base-uncased` for speed).

### 3. LLM as Judge (Structured Quality Assessment)

**Purpose**: Expert evaluation by Claude Sonnet 4.5 across multiple quality dimensions.

**Criteria** (1-5 scale):

| Criterion | Description | Threshold | What It Measures |
|-----------|-------------|-----------|------------------|
| **Accuracy** | Factual correctness | 4.0/5.0 | Are facts, numbers, and claims correct? |
| **Completeness** | Address all aspects | 3.5/5.0 | Is anything important missing? |
| **Clarity** | Clear communication | 3.5/5.0 | Is it easy to understand? |
| **Conciseness** | Appropriate brevity | 3.0/5.0 | Is it concise without sacrificing content? |
| **Code Quality** | Analytics agent only | N/A | Is generated code clean and correct? |
| **Visualization** | Analytics agent only | N/A | Are chart recommendations appropriate? |

**When to Use**:
- Holistic quality assessment
- Identifying specific improvement areas
- Comparing multiple answers (pairwise evaluation)
- Human-interpretable feedback

**Output**: Structured JSON with scores, reasoning, strengths, weaknesses, and recommendations.

**Example**:
```python
from tests.utils.llm_judge import LLMJudge

judge = LLMJudge(config)
judgment = judge.evaluate(
    query="What is the unemployment rate trend?",
    answer="The unemployment rate decreased from 3.2% in Q1 to 2.8% in Q4 2023.",
    context="Q1: 3.2%, Q2: 3.0%, Q3: 2.9%, Q4: 2.8%"
)

print(f"Overall Score: {judgment.overall_score}/5.0")
print(f"Accuracy: {judgment.criteria_scores['accuracy'].score}/5.0")
print(f"Reasoning: {judgment.criteria_scores['accuracy'].reasoning}")
print(f"Strengths: {judgment.strengths}")
print(f"Weaknesses: {judgment.weaknesses}")
```

### 4. Security Testing (Code Execution Safety)

**Purpose**: Validate that the code execution security layer protects against malicious or dangerous code.

**Test Categories**:

| Category | Description | Test Count | What It Tests |
|----------|-------------|-----------|---------------|
| **Code Validator** | AST-based validation | 15 tests | Blocks forbidden functions, imports, attributes |
| **Sandbox Executor** | Restricted execution | 10 tests | Enforces resource limits, blocks I/O, timeout protection |
| **Security Integration** | End-to-end security | 5 tests | Defense-in-depth, audit logging, full workflow |

**When to Use**:
- Verifying code generation safety
- Testing new code validation rules
- Validating security configuration changes
- Ensuring analytics agent doesn't allow dangerous code

**Running Security Tests**:

```bash
# Run all security tests
pytest tests/security/ -v

# Run specific security test categories
pytest tests/security/test_code_validator.py -v      # Validation tests
pytest tests/security/test_sandbox_executor.py -v    # Sandbox tests
pytest tests/security/test_security_integration.py -v # Integration tests

# Run with coverage
pytest tests/security/ --cov=app.services.security --cov-report=html

# Run specific test cases
pytest tests/security/test_code_validator.py::TestCodeValidator::test_blocks_eval
pytest tests/security/test_sandbox_executor.py::TestSandboxExecutor::test_timeout_enforcement
```

**Test Coverage**:

**Code Validator Tests** (`test_code_validator.py`):
```python
✅ test_blocks_eval              # Blocks eval() function
✅ test_blocks_exec              # Blocks exec() function
✅ test_blocks_compile           # Blocks compile() function
✅ test_blocks_open              # Blocks file I/O
✅ test_blocks_import            # Blocks __import__()
✅ test_blocks_dunder_globals    # Blocks __globals__ access
✅ test_blocks_dunder_class      # Blocks __class__ access
✅ test_blocks_subprocess        # Blocks subprocess module
✅ test_blocks_os_system         # Blocks os.system()
✅ test_blocks_network           # Blocks requests, urllib
✅ test_blocks_database_ops      # Blocks pd.read_sql, to_sql
✅ test_allows_pandas            # Allows safe pandas code
✅ test_allows_numpy             # Allows safe numpy code
✅ test_allows_matplotlib        # Allows safe matplotlib code
✅ test_syntax_error_detection   # Catches syntax errors
```

**Sandbox Executor Tests** (`test_sandbox_executor.py`):
```python
✅ test_executes_safe_code       # Executes legitimate pandas code
✅ test_timeout_enforcement      # Stops infinite loops
✅ test_memory_limits            # Prevents memory exhaustion
✅ test_restricted_builtins      # Blocks dangerous builtins
✅ test_no_file_access           # Prevents file I/O
✅ test_no_network_access        # Prevents network access
✅ test_captures_exceptions      # Handles runtime errors gracefully
✅ test_result_extraction        # Extracts execution results
✅ test_dataframe_context        # Works with pandas DataFrames
✅ test_visualization_support    # Supports matplotlib figures
```

**Security Integration Tests** (`test_security_integration.py`):
```python
✅ test_validate_then_execute_legitimate_code     # Full workflow with safe code
✅ test_validate_blocks_dangerous_code            # Validation layer blocks threats
✅ test_validate_blocks_attribute_escape          # Blocks __globals__ escape
✅ test_validate_blocks_database_operations       # Blocks SQL operations
✅ test_audit_logging_captures_execution          # Audit logs all executions
✅ test_audit_logging_security_violation          # Logs security violations
✅ test_security_can_be_disabled                  # Configuration toggle works
✅ test_process_isolation_can_be_disabled         # Sandbox config works
✅ test_complex_pandas_analysis                   # Real-world analysis passes
✅ test_visualization_code                        # Matplotlib code passes
✅ test_handles_runtime_errors_gracefully         # Runtime errors handled
✅ test_handles_syntax_errors_in_validation       # Syntax errors caught
✅ test_multiple_security_layers                  # Defense in depth works
✅ test_cannot_bypass_restricted_builtins         # Bypass attempts fail
```

**Example Test Case**:

```python
def test_blocks_eval():
    """Test that eval() is blocked by code validator."""
    from app.services.security import CodeValidator

    validator = CodeValidator(config={"use_ast_visitor": True})

    # Malicious code using eval
    code = "result = eval('2+2')"

    # Validate code
    validation_result = validator.validate(code)

    # Should be invalid
    assert not validation_result.is_valid
    assert any("eval" in error.lower() for error in validation_result.errors)
    assert "eval" in validation_result.forbidden_functions


def test_timeout_enforcement():
    """Test that infinite loops are stopped by timeout."""
    from app.services.security import SandboxExecutor

    executor = SandboxExecutor(config={
        "use_process_isolation": False,
        "cpu_time_limit_seconds": 1
    })

    # Infinite loop code
    code = "while True: pass"

    # Execute with timeout
    execution_result = executor.execute_code(code)

    # Should fail with timeout
    assert not execution_result.success
    assert execution_result.error_type in ["TimeoutError", "TimeoutException"]
    assert "timeout" in execution_result.error.lower()
```

**Audit Log Verification**:

```bash
# Check audit logs after security tests
cat logs/code_execution_audit.log | jq .

# Count security violations
grep 'security_violation' logs/code_execution_audit.log | wc -l

# View specific violation types
grep 'forbidden_function' logs/code_execution_audit.log | jq .
```

**Security Test Best Practices**:

1. **Always Test Both Layers**: Test validation AND sandbox independently
2. **Use Realistic Attack Vectors**: Test real-world bypass techniques
3. **Verify Audit Logs**: Check that violations are logged correctly
4. **Test Legitimate Code**: Ensure safe code still works
5. **Test Configuration Toggles**: Verify enable/disable switches work

**Common Security Test Patterns**:

```python
# Pattern 1: Validation blocks, sandbox catches bypass
def test_defense_in_depth():
    validator = CodeValidator(config={"use_ast_visitor": True})
    executor = SandboxExecutor(config={"use_process_isolation": False})

    dangerous_code = "result = __import__('os').system('ls')"

    # Layer 1: Validation should block
    validation = validator.validate(dangerous_code)
    assert not validation.is_valid

    # Layer 2: Even if validation bypassed, sandbox blocks
    execution = executor.execute_code(dangerous_code)
    assert not execution.success


# Pattern 2: Legitimate code passes all layers
def test_legitimate_code_passes():
    validator = CodeValidator(config={"use_ast_visitor": True})
    executor = SandboxExecutor(config={"use_process_isolation": False})

    safe_code = "result = df.groupby('age')['value'].sum()"
    df = pd.DataFrame({"age": [1, 2], "value": [10, 20]})

    # Validation passes
    validation = validator.validate(safe_code, dataframe=df)
    assert validation.is_valid

    # Execution succeeds
    execution = executor.execute_code(safe_code, context={"df": df})
    assert execution.success
    assert execution.result == 30  # Expected sum
```

---

### Frontend Tests (TypeScript + Jest)

The frontend uses **Jest** with **React Testing Library** for unit and integration testing of React components, stores, and API clients.

#### Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (if not already installed)
npm install
```

#### Basic Commands

```bash
# Run all frontend tests
npm test

# Run tests in watch mode (auto-rerun on file changes)
npm run test:watch

# Run tests with coverage report
npm run test:coverage

# Run tests in CI mode (non-interactive, with coverage)
npm run test:ci
```

#### Test Structure

The frontend tests are organized as follows:

```
frontend/src/
├── api/
│   ├── __tests__/
│   │   └── client.test.ts          # API client tests (chatApi, configApi, dataApi)
│   └── client.ts
├── store/
│   ├── __tests__/
│   │   └── chatStore.test.ts       # Zustand store tests (state management)
│   └── chatStore.ts
├── types/
│   ├── __tests__/
│   │   └── index.test.ts           # TypeScript type definition tests
│   └── index.ts
└── components/                      # Component tests (future)
    └── __tests__/
```

#### Test Coverage

**Current Frontend Test Coverage**:

| Module | Description | Test Count | Coverage Target |
|--------|-------------|-----------|-----------------|
| **API Client** | HTTP requests, SSE streaming | 15 tests | 80%+ |
| **Chat Store** | State management, message handling | 12 tests | 85%+ |
| **Type Definitions** | TypeScript type validation | 8 tests | 100% |

**API Client Tests** (`api/__tests__/client.test.ts`):
```typescript
✅ test_stream_message_sse_events         # SSE streaming with events
✅ test_stream_message_visualization      # Visualization events
✅ test_stream_message_errors             # Error handling during streaming
✅ test_stream_message_http_errors        # HTTP error responses
✅ test_stream_message_network_errors     # Network failures
✅ test_stream_message_malformed_json     # Malformed JSON handling
✅ test_send_message_post                 # Traditional POST request
✅ test_get_history                       # Fetch conversation history
✅ test_clear_history                     # Clear conversation
✅ test_list_conversations                # List all conversations
✅ test_config_health_check               # Health check endpoint
✅ test_data_list_datasets                # List datasets
✅ test_data_get_dataset_info             # Get dataset metadata
✅ test_data_query_dataset_with_params    # Query with parameters
✅ test_data_query_dataset_no_params      # Query without parameters
```

**Chat Store Tests** (`store/__tests__/chatStore.test.ts`):
```typescript
✅ test_initial_state                     # Verify initial store state
✅ test_send_message_non_streaming        # Traditional message sending
✅ test_send_message_error_handling       # Error handling
✅ test_send_message_streaming            # SSE streaming message
✅ test_streaming_errors                  # Streaming error handling
✅ test_streaming_visualization           # Visualization in streaming
✅ test_clear_messages                    # Clear conversation
✅ test_set_visualization                 # Set visualization data
✅ test_clear_visualization               # Clear visualization
✅ test_toggle_streaming_on               # Enable streaming mode
✅ test_toggle_streaming_off              # Disable streaming mode
✅ test_toggle_streaming_multiple         # Toggle multiple times
```

**Type Definition Tests** (`types/__tests__/index.test.ts`):
```typescript
✅ test_message_user                      # User message type
✅ test_message_assistant_with_viz        # Assistant message with visualization
✅ test_visualization_bar_chart           # Bar chart type
✅ test_visualization_line_chart          # Line chart type
✅ test_visualization_pie_chart           # Pie chart type
✅ test_visualization_scatter_chart       # Scatter chart type
✅ test_visualization_table               # Table type
✅ test_chat_request_basic                # Basic chat request
✅ test_chat_request_with_conversation    # Request with conversation ID
✅ test_chat_response_with_metadata       # Response with metadata
```

#### Running Specific Tests

```bash
# Run specific test file
npm test -- api/__tests__/client.test.ts

# Run tests matching pattern
npm test -- --testNamePattern="streaming"

# Run tests for specific module
npm test -- store/

# Run with verbose output
npm test -- --verbose

# Run with debug output
npm test -- --no-coverage --verbose
```

#### Coverage Reports

```bash
# Generate HTML coverage report
npm run test:coverage

# Open coverage report in browser (macOS)
open coverage/lcov-report/index.html

# Open coverage report in browser (Linux)
xdg-open coverage/lcov-report/index.html

# View coverage summary in terminal
npm test -- --coverage --coverageReporters=text
```

#### Frontend Test Best Practices

1. **Mock External Dependencies**
   - Always mock `axios` for HTTP requests
   - Mock `fetch` for SSE streaming
   - Mock `ReadableStream` for browser APIs

2. **Test Zustand Stores with renderHook**
   ```typescript
   import { renderHook, act } from '@testing-library/react';
   import { useChatStore } from '../chatStore';

   const { result } = renderHook(() => useChatStore());

   act(() => {
     result.current.sendMessage('Hello');
   });
   ```

3. **Use waitFor for Asynchronous Operations**
   ```typescript
   await waitFor(() => {
     expect(result.current.messages).toHaveLength(2);
   });
   ```

4. **Test Both Success and Error Paths**
   - Always test happy path AND error scenarios
   - Test network failures, HTTP errors, malformed data

5. **Verify State Transitions**
   - Test initial state → loading → success/error
   - Verify all state fields are updated correctly

6. **Clear Mocks Between Tests**
   ```typescript
   beforeEach(() => {
     jest.clearAllMocks();
   });
   ```

#### Common Frontend Test Patterns

**Pattern 1: Testing API Calls**
```typescript
it('should send message via API', async () => {
  const mockResponse = { message: 'Hello', conversation_id: 'conv-123' };
  (chatApi.sendMessage as jest.Mock).mockResolvedValue(mockResponse);

  const { result } = renderHook(() => useChatStore());

  await act(async () => {
    await result.current.sendMessage('Test');
  });

  expect(chatApi.sendMessage).toHaveBeenCalledWith({
    message: 'Test',
    conversation_id: undefined,
    include_visualization: true,
  });
});
```

**Pattern 2: Testing SSE Streaming**
```typescript
it('should handle SSE events', async () => {
  const mockEvents = [
    'data: {"type":"start","conversation_id":"conv-123"}\n\n',
    'data: {"type":"token","content":"Hello"}\n\n',
    'data: {"type":"done"}\n\n',
  ];

  const mockReadableStream = new ReadableStream({
    start(controller) {
      mockEvents.forEach(event => {
        controller.enqueue(new TextEncoder().encode(event));
      });
      controller.close();
    },
  });

  (global.fetch as jest.Mock).mockResolvedValue({
    ok: true,
    body: mockReadableStream,
  });

  const callbacks = {
    onStart: jest.fn(),
    onToken: jest.fn(),
    onComplete: jest.fn(),
  };

  await chatApi.streamMessage({ message: 'Test' }, callbacks);

  expect(callbacks.onStart).toHaveBeenCalledWith('conv-123');
  expect(callbacks.onToken).toHaveBeenCalledWith('Hello');
});
```

**Pattern 3: Testing Error Handling**
```typescript
it('should handle errors gracefully', async () => {
  (chatApi.sendMessage as jest.Mock).mockRejectedValue(
    new Error('Network error')
  );

  const { result } = renderHook(() => useChatStore());

  await act(async () => {
    await result.current.sendMessage('Test');
  });

  expect(result.current.error).toBe('Network error');
  expect(result.current.isLoading).toBe(false);
});
```

---

## Fine-Tuning Data Collection

### Overview

The evaluation system can export high-quality query-answer pairs with scores for fine-tuning LLMs. This enables:
- Training custom models on domain-specific data
- Improving performance on Singapore government datasets
- Reducing dependency on expensive external APIs
- Creating agent-specific fine-tuned models

### Running Evaluations and Exporting

#### 1. Full Evaluation Suite

```bash
# Run all evaluation tests and export to OpenAI format
python scripts/run_evaluation.py \
  --test-suite evaluation \
  --export-format openai \
  --output-dir ./fine_tuning_data \
  --min-score 0.80 \
  --include-metadata
```

**Output**:
```
fine_tuning_data/
├── summary_report.json                    # Aggregate metrics
├── verification_agent_training.jsonl      # Verification agent examples
├── coordinator_agent_training.jsonl       # Coordinator agent examples
├── extraction_agent_training.jsonl        # Extraction agent examples
├── analytics_agent_training.jsonl         # Analytics agent examples
└── all_agents_training.jsonl             # Combined dataset
```

#### 2. Agent-Specific Evaluation

```bash
# Export only verification agent data
python scripts/run_evaluation.py \
  --agent verification \
  --export-format openai \
  --output-dir ./fine_tuning_data/verification

# Export only analytics agent data
python scripts/run_evaluation.py \
  --agent analytics \
  --export-format anthropic \
  --output-dir ./fine_tuning_data/analytics
```

#### 3. Custom Test Queries

```bash
# Use custom queries and ground truth
python scripts/run_evaluation.py \
  --queries-file tests/fixtures/custom_queries.json \
  --ground-truth tests/fixtures/custom_answers.json \
  --export-format jsonl \
  --output-dir ./custom_eval \
  --min-score 0.85
```

**custom_queries.json format**:
```json
[
  {
    "query": "What was the average salary in Singapore in 2023?",
    "agent": "analytics",
    "expected_datasets": ["singapore_salary_2023"]
  },
  {
    "query": "Show me employment trends over the last 5 years",
    "agent": "analytics",
    "expected_datasets": ["employment_statistics"]
  }
]
```

**custom_answers.json format**:
```json
{
  "What was the average salary in Singapore in 2023?": {
    "answer": "The average monthly salary in Singapore in 2023 was $5,197.",
    "contexts": ["Singapore average monthly salary: $5,197 (2023)"],
    "metadata": {"year": 2023, "topic": "salary"}
  }
}
```

### Export Formats

#### 1. OpenAI Fine-Tuning Format (JSONL)

**Use Case**: Fine-tuning GPT-3.5/GPT-4 models via OpenAI API.

**Format**:
```jsonl
{"messages": [{"role": "system", "content": "You are a data verification agent..."}, {"role": "user", "content": "Is this query about Singapore employment data?"}, {"role": "assistant", "content": "Yes, this query is relevant..."}], "metadata": {"score": 0.95, "ragas_faithfulness": 0.98, "llm_judge_accuracy": 4.5}}
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "metadata": {...}}
```

**How to Use**:
```bash
# Upload to OpenAI for fine-tuning
openai api fine_tuning.jobs.create \
  -t fine_tuning_data/verification_agent_training.jsonl \
  -m gpt-3.5-turbo
```

#### 2. Anthropic Fine-Tuning Format (JSONL)

**Use Case**: Fine-tuning Claude models (when available).

**Format**:
```jsonl
{"prompt": "You are a data coordinator agent...\n\nUser query: What datasets are needed for salary analysis?\n\nAssistant:", "completion": " To analyze salary data, I recommend using the following datasets:\n1. singapore_salary_statistics_2023\n2. employment_income_data\n\nThese datasets contain comprehensive salary information...", "metadata": {"score": 0.92, "ragas_correctness": 0.89}}
```

#### 3. Generic JSONL Format

**Use Case**: Custom training pipelines, data analysis, or other frameworks.

**Format**:
```jsonl
{"query": "What was the unemployment rate in 2023?", "answer": "The unemployment rate in Singapore in 2023 was 2.8%.", "contexts": ["..."], "scores": {"ragas_faithfulness": 0.95, "bertscore_f1": 0.88, "llm_judge_overall": 4.2}, "agent": "analytics", "timestamp": "2026-02-11T10:30:00Z"}
```

#### 4. CSV Format

**Use Case**: Spreadsheet analysis, manual review, non-ML applications.

**Format**:
```csv
query,answer,agent,ragas_faithfulness,bertscore_f1,llm_judge_overall,timestamp
"What was the unemployment rate in 2023?","The unemployment rate...",analytics,0.95,0.88,4.2,2026-02-11T10:30:00Z
```

#### 5. Parquet Format

**Use Case**: Large-scale data analysis with pandas, Spark, or data warehouses.

**Advantages**: Efficient compression, fast queries, schema preservation.

### Filtering and Quality Control

```bash
# Export only high-quality examples (score >= 0.90)
python scripts/run_evaluation.py \
  --export-format openai \
  --min-score 0.90 \
  --output-dir ./high_quality_data

# Export all examples including low scores (for negative examples)
python scripts/run_evaluation.py \
  --export-format jsonl \
  --min-score 0.0 \
  --include-low-scores \
  --output-dir ./all_examples

# Filter specific metrics
python scripts/run_evaluation.py \
  --export-format openai \
  --min-ragas-faithfulness 0.95 \
  --min-llm-judge-accuracy 4.0 \
  --output-dir ./filtered_data
```

### Batch Size and Sampling

```bash
# Evaluate on a sample (faster, for testing)
python scripts/run_evaluation.py \
  --sample-size 50 \
  --export-format openai \
  --output-dir ./sample_data

# Full evaluation (all test cases)
python scripts/run_evaluation.py \
  --test-suite full \
  --export-format openai \
  --output-dir ./full_data
```

---

## LangSmith Integration

[LangSmith](https://smith.langchain.com) provides real-time tracing and monitoring for LangChain applications.

### Setup

1. **Get API Key**:
   - Sign up at https://smith.langchain.com
   - Create an API key in settings

2. **Configure Environment**:
```bash
# Add to backend/.env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls_your_api_key_here
LANGCHAIN_PROJECT=govtech-multi-agent-system
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

3. **Configure in config.yaml**:
```yaml
langsmith:
  enabled: true
  project_name: "govtech-multi-agent-system"
  tags:
    - "production"
    - "multi-agent"
    - "singapore-data"
  sample_rate: 1.0  # Trace 100% of requests (set to 0.1 for 10% sampling)
```

### Usage

Once enabled, **all agent operations are automatically traced**. No code changes needed.

**What Gets Traced**:
- User queries
- Agent decisions and workflows
- RAG retrieval steps (search, reranking)
- LLM calls with prompts and responses
- Tool usage (database queries, file operations)
- Errors and exceptions
- Latency metrics
- Token usage

### Viewing Traces

1. Go to https://smith.langchain.com
2. Navigate to your project: "govtech-multi-agent-system"
3. View runs in real-time

**Filter Options**:
- By status (success, error)
- By agent name (verification, coordinator, extraction, analytics)
- By latency (> 5 seconds)
- By token usage (> 1000 tokens)
- By tags (production, testing, etc.)
- By date range

### Monitoring Queries

**Python Script**:
```python
from langsmith import Client

client = Client()

# Get recent runs
runs = client.list_runs(
    project_name="govtech-multi-agent-system",
    limit=100,
    filter="status = 'error'"  # Only errors
)

for run in runs:
    print(f"Run: {run.id}")
    print(f"  Name: {run.name}")
    print(f"  Status: {run.status}")
    print(f"  Latency: {run.latency_ms}ms")
    print(f"  Tokens: {run.total_tokens}")
    print(f"  Error: {run.error}")
```

**Export for Offline Analysis**:
```python
# Export runs to CSV
import pandas as pd

runs = list(client.list_runs(project_name="govtech-multi-agent-system", limit=1000))
df = pd.DataFrame([
    {
        "id": r.id,
        "name": r.name,
        "status": r.status,
        "latency_ms": r.latency_ms,
        "total_tokens": r.total_tokens,
        "start_time": r.start_time,
    }
    for r in runs
])

df.to_csv("langsmith_runs.csv", index=False)
```

### Alerts and Dashboards

**Set Up Alerts** (in LangSmith UI):
1. Go to Settings → Alerts
2. Create alert for:
   - Error rate > 5%
   - P95 latency > 30 seconds
   - Token usage > 10,000 per request
   - Specific error types

**Create Dashboards**:
1. Go to Analytics → Dashboards
2. Add widgets for:
   - Request volume over time
   - Error rate by agent
   - P50/P90/P95 latency
   - Token usage trends
   - Most common queries
   - Most common errors

---

## Evaluation Results

### Output Directory Structure

After running evaluations, results are organized as follows:

```
evaluation_results/
├── summary_report.json                    # Aggregate metrics across all agents
│
├── agent_scores/                          # Per-agent score details
│   ├── verification_scores.jsonl
│   ├── coordinator_scores.jsonl
│   ├── extraction_scores.jsonl
│   └── analytics_scores.jsonl
│
├── fine_tuning_data/                      # Ready-to-use training data
│   ├── verification_training.jsonl
│   ├── coordinator_training.jsonl
│   ├── extraction_training.jsonl
│   ├── analytics_training.jsonl
│   └── all_agents_training.jsonl
│
├── detailed_results/                      # Raw evaluation outputs
│   ├── ragas_results.json
│   ├── bertscore_results.json
│   └── llm_judge_results.json
│
├── performance_metrics/                   # Latency and throughput
│   └── latency_report.json
│
└── metadata/                              # Test configuration and timestamps
    ├── test_config.yaml
    └── run_info.json
```

### Summary Report Format

**summary_report.json**:
```json
{
  "evaluation_id": "eval_20260211_103045",
  "timestamp": "2026-02-11T10:30:45Z",
  "total_test_cases": 150,
  "agents_evaluated": ["verification", "coordinator", "extraction", "analytics"],

  "aggregate_metrics": {
    "ragas": {
      "context_precision": 0.87,
      "context_recall": 0.92,
      "faithfulness": 0.89,
      "answer_relevancy": 0.85,
      "answer_correctness": 0.81
    },
    "bertscore": {
      "precision": 0.78,
      "recall": 0.76,
      "f1": 0.77
    },
    "llm_judge": {
      "overall_score": 4.1,
      "accuracy": 4.3,
      "completeness": 3.9,
      "clarity": 4.2,
      "conciseness": 3.8
    }
  },

  "by_agent": {
    "verification": {
      "test_cases": 30,
      "pass_rate": 0.93,
      "avg_score": 4.2,
      "avg_latency_ms": 1250
    },
    "coordinator": {
      "test_cases": 40,
      "pass_rate": 0.90,
      "avg_score": 4.0,
      "avg_latency_ms": 2100
    },
    "extraction": {
      "test_cases": 40,
      "pass_rate": 0.88,
      "avg_score": 3.9,
      "avg_latency_ms": 3500
    },
    "analytics": {
      "test_cases": 40,
      "pass_rate": 0.85,
      "avg_score": 3.8,
      "avg_latency_ms": 8200
    }
  },

  "threshold_compliance": {
    "ragas_thresholds_met": true,
    "bertscore_thresholds_met": true,
    "llm_judge_thresholds_met": true
  },

  "recommendations": [
    "Analytics agent shows lower scores - review code generation quality",
    "Extraction agent latency high - optimize dataset loading",
    "Overall performance meets thresholds - system ready for production"
  ]
}
```

### Fine-Tuning Data Format

**OpenAI Format** (verification_training.jsonl):
```jsonl
{"messages": [{"role": "system", "content": "You are a query verification agent for Singapore government data. Validate if queries are relevant to employment, salary, or labour statistics."}, {"role": "user", "content": "Is this query relevant to Singapore employment data: 'What was the unemployment rate in 2023?'"}, {"role": "assistant", "content": "Yes, this query is relevant. It asks about unemployment rate, which is a core employment statistic. The query specifies a year (2023), which helps narrow the dataset search."}], "metadata": {"score": 0.95, "ragas_faithfulness": 0.98, "llm_judge_accuracy": 4.5, "llm_judge_overall": 4.3, "bertscore_f1": 0.89, "agent": "verification", "timestamp": "2026-02-11T10:30:45Z"}}
```

**Anthropic Format** (analytics_training.jsonl):
```jsonl
{"prompt": "You are a data analytics agent. Generate pandas code to analyze the following data and answer the user's question.\n\nUser Query: What is the average salary trend over the last 3 years?\n\nAvailable DataFrames:\n- df_salary: columns=['year', 'salary', 'occupation'], 5000 rows\n\nAssistant:", "completion": " Here's the pandas code to analyze the salary trend:\n\n```python\nimport pandas as pd\nimport numpy as np\n\n# Filter last 3 years\nrecent_data = df_salary[df_salary['year'] >= 2021]\n\n# Calculate average salary by year\nsalary_trend = recent_data.groupby('year')['salary'].mean()\n\nprint(salary_trend)\n```\n\nThe average salary shows an upward trend: 2021: $4,850, 2022: $5,020, 2023: $5,197, indicating a 7.2% increase over 3 years.", "metadata": {"score": 0.92, "ragas_correctness": 0.89, "llm_judge_code_quality": 4.5, "agent": "analytics"}}
```

---

## Interpreting Results

### Quality Thresholds

| Grade | Score Range | Status | Action |
|-------|-------------|--------|--------|
| **Excellent** | ≥ 90% | ✅ Production-ready | Use as training examples |
| **Good** | 80-89% | ✅ Acceptable | Minor improvements recommended |
| **Fair** | 70-79% | ⚠️ Needs improvement | Review and optimize |
| **Poor** | < 70% | ❌ Not acceptable | Major rework required |

### Metric-Specific Interpretation

#### Low Context Precision (< 0.85)

**What It Means**: RAG is retrieving irrelevant documents.

**Possible Causes**:
- Poor query embeddings
- Weak reranking model
- BM25 weights too high/low
- Too many results returned (high `top_k`)

**Solutions**:
```yaml
# In config.yaml
rag:
  top_k: 10  # Reduce to 5 to get fewer but more relevant results
  use_reranking: true  # Enable cross-encoder reranking
  confidence_threshold: 0.6  # Increase to filter low-confidence matches
```

#### Low Context Recall (< 0.90)

**What It Means**: RAG is missing necessary information.

**Possible Causes**:
- Query embedding doesn't capture all semantic meaning
- Too few results returned (low `top_k`)
- Missing data in vector database
- Chunks too small (information split across chunks)

**Solutions**:
```yaml
rag:
  top_k: 20  # Increase to retrieve more contexts
  chunk_size: 1500  # Increase chunk size
  chunk_overlap: 300  # Increase overlap to prevent splits
```

#### Low Faithfulness (< 0.85)

**What It Means**: LLM is hallucinating or making unsupported claims.

**Possible Causes**:
- Prompt doesn't emphasize grounding
- LLM temperature too high (too creative)
- Contexts don't contain answer
- LLM model too small or under-trained

**Solutions**:
```python
# In agent prompts
system_prompt = """
CRITICAL: Base your answer ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have enough information."
Never make up facts or numbers. Always cite specific data from the context.
"""

# In config.yaml
llm:
  default_provider: anthropic
  providers:
    anthropic:
      temperature: 0.0  # Reduce temperature for more factual outputs
      default_model: claude-sonnet-4-5  # Use more capable model
```

#### Low Answer Relevancy (< 0.80)

**What It Means**: Answer doesn't address the question.

**Possible Causes**:
- LLM misunderstands query
- Retrieved contexts don't match query
- Prompt doesn't focus LLM on the specific question

**Solutions**:
```python
# Improve query understanding in prompt
prompt = f"""
User Query: {query}

Task: Answer the specific question asked. Address ALL parts of the query.

Retrieved Data:
{context}

Answer:
"""
```

#### Low BERTScore (< 0.75)

**What It Means**: Generated text semantically differs from reference.

**Possible Causes**:
- Answer uses different phrasing (might still be correct)
- Answer includes extra information
- Answer is incomplete

**Solutions**:
- Check if semantic difference is acceptable (BERTScore may be overly strict)
- Update reference answers to match acceptable variations
- Use LLM Judge for more nuanced evaluation

#### Low LLM Judge Scores (< 3.5)

**What It Means**: Quality issues in specific criteria.

**Check Specific Scores**:
- **Low Accuracy** → Factual errors, wrong numbers
- **Low Completeness** → Missing information, partial answers
- **Low Clarity** → Confusing language, poor structure
- **Low Conciseness** → Too verbose, repetitive
- **Low Code Quality** → Inefficient/incorrect code (analytics agent)
- **Low Visualization** → Inappropriate chart type (analytics agent)

**Use Reasoning Field**:
```python
judgment = judge.evaluate(query, answer, context)

for criterion_name, criterion in judgment.criteria_scores.items():
    if criterion.score < 3.5:
        print(f"{criterion_name}: {criterion.score}/5.0")
        print(f"Reasoning: {criterion.reasoning}")
        print(f"Issues: {criterion.issues}")
        print(f"Recommendation: {judgment.recommendation}")
```

### Common Failure Patterns

#### Pattern 1: High Faithfulness, Low Correctness

**What It Means**: Answer is grounded in retrieved data, but the data itself is wrong or incomplete.

**Root Cause**: Poor retrieval (wrong datasets loaded).

**Solution**: Improve retrieval (better embeddings, reranking, query understanding).

#### Pattern 2: High Relevancy, Low Faithfulness

**What It Means**: Answer addresses the question but includes hallucinated information.

**Root Cause**: LLM is too creative or not following grounding instructions.

**Solution**: Reduce temperature, improve prompt, use more capable model.

#### Pattern 3: High Metrics, Poor User Experience

**What It Means**: Scores are high but users report issues.

**Root Cause**: Evaluation metrics don't capture all quality dimensions.

**Solution**: Add human evaluation, expand test cases, use LLM Judge for nuanced assessment.

---

## Best Practices

### For Fine-Tuning Data Collection

1. **Diverse Queries** (100+ samples minimum)
   - Cover all query types (simple facts, trends, comparisons, visualizations)
   - Include edge cases (ambiguous queries, out-of-scope questions)
   - Balance across all agents

2. **Include Positive and Negative Examples**
   ```bash
   # Export high-quality examples (score >= 0.85)
   python scripts/run_evaluation.py --min-score 0.85 --output-dir ./positive_examples

   # Export low-quality examples with corrections (score < 0.70)
   python scripts/run_evaluation.py --max-score 0.70 --output-dir ./negative_examples
   ```

3. **Balance Dataset Across Agents**
   - Verification: 20-30% of examples
   - Coordinator: 20-30% of examples
   - Extraction: 25-35% of examples
   - Analytics: 25-35% of examples

4. **Include Metadata for Filtering**
   ```bash
   python scripts/run_evaluation.py --include-metadata --output-dir ./data
   ```
   Metadata enables post-processing filtering and analysis.

5. **Periodically Re-Evaluate**
   - Re-run evaluations after model updates
   - Track metrics over time to detect regressions
   - Update training data as system evolves

### For Continuous Evaluation

1. **Run Evaluations Before Each Release**
   ```bash
   # In CI/CD pipeline
   pytest tests/evaluation/ --junitxml=evaluation_results.xml
   python scripts/run_evaluation.py --export-format jsonl --output-dir ./ci_eval
   ```

2. **Track Metrics Over Time**
   ```python
   # Store metrics in database or append to file
   import json
   from datetime import datetime

   metrics = {
       "timestamp": datetime.now().isoformat(),
       "git_commit": "abc123",
       "ragas_faithfulness": 0.89,
       "llm_judge_overall": 4.2,
       # ... other metrics
   }

   with open("metrics_history.jsonl", "a") as f:
       f.write(json.dumps(metrics) + "\n")
   ```

3. **Set Up LangSmith Alerts**
   - Alert on error rate > 5%
   - Alert on P95 latency > 30 seconds
   - Alert on token usage spikes

4. **Use LangSmith for Production Monitoring**
   - Enable tracing in production (with sampling if needed)
   - Review traces daily for errors
   - Identify common failure patterns

5. **Review Failed Tests for Root Cause**
   ```bash
   # Re-run failed tests with debugging
   pytest --lf -vv --tb=long
   ```

### For Debugging Low Scores

1. **Check Test Logs**
   ```bash
   pytest tests/evaluation/ -vv --log-cli-level=DEBUG
   ```

2. **Review Specific Test Case**
   ```python
   # In test file or notebook
   from tests.utils.llm_judge import LLMJudge

   judge = LLMJudge(config)
   judgment = judge.evaluate(
       query="What was the unemployment rate in 2023?",
       answer="The rate was 2.8%.",  # Test answer
       context="Q4 2023 unemployment: 2.8%",
       ground_truth="The unemployment rate in Singapore in 2023 was 2.8%."
   )

   # Print detailed judgment
   print(json.dumps(judgment.to_dict(), indent=2))
   ```

3. **Compare Against Baseline**
   ```bash
   # Run evaluation on main branch
   git checkout main
   pytest tests/evaluation/ --json-report --json-report-file=baseline.json

   # Run evaluation on feature branch
   git checkout feature-branch
   pytest tests/evaluation/ --json-report --json-report-file=feature.json

   # Compare results
   python scripts/compare_evaluations.py baseline.json feature.json
   ```

4. **Use LangSmith for Trace Inspection**
   - Find run ID from logs
   - Open trace in LangSmith UI
   - Inspect prompts, responses, and intermediate steps
   - Check token usage and latency

---

## Troubleshooting

### Tests Failing with LLM Errors

**Symptoms**:
```
Error: anthropic.RateLimitError: 429 Too Many Requests
```

**Solutions**:
1. Check API keys in `.env`:
   ```bash
   cat backend/.env | grep API_KEY
   ```
2. Verify API key is valid and has credits
3. Reduce parallelism:
   ```bash
   pytest -n 2  # Run with 2 workers instead of auto
   ```
4. Add delays between tests:
   ```python
   # In conftest.py
   import time
   pytest.fixture(autouse=True)
   def slow_down_tests():
       time.sleep(0.5)
   ```

### Slow Test Execution

**Symptoms**:
```
tests/evaluation/ taking > 10 minutes
```

**Solutions**:
1. Skip slow tests during development:
   ```bash
   pytest -m "not slow"
   ```
2. Run tests in parallel:
   ```bash
   pytest -n auto
   ```
3. Use cached embeddings:
   ```yaml
   # In test_config.yaml
   testing:
     use_local_embeddings: true
     cache:
       embeddings: true
   ```
4. Reduce sample size:
   ```yaml
   testing:
     ragas:
       sample_size: 10  # Down from 50
   ```

### Database Connection Errors

**Symptoms**:
```
Error: could not connect to server: Connection refused
```

**Solutions**:
1. Start PostgreSQL:
   ```bash
   make dev  # Docker
   # OR
   brew services start postgresql@16  # macOS
   ```
2. Check `DATABASE_URL` in `.env`:
   ```bash
   # Should match your environment
   DATABASE_URL=postgresql+asyncpg://govtech:govtech_dev@localhost:5432/govtech_rag_test
   ```
3. Reset test database:
   ```bash
   psql -U govtech -d govtech_rag_test -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
   pytest tests/integration/test_database.py::test_db_initialization
   ```

### Import Errors

**Symptoms**:
```
ModuleNotFoundError: No module named 'app'
```

**Solutions**:
1. Activate virtual environment:
   ```bash
   cd backend
   source venv/bin/activate  # macOS/Linux
   # OR
   .\venv\Scripts\activate  # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Verify PYTHONPATH:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:${PWD}"
   ```

### Embedding Errors

**Symptoms**:
```
Error: openai.error.AuthenticationError: Incorrect API key
```

**Solutions**:
1. Check `OPENAI_API_KEY`:
   ```bash
   echo $OPENAI_API_KEY
   ```
2. Use local embeddings for testing:
   ```yaml
   # In test_config.yaml
   testing:
     use_local_embeddings: true
   ```
3. Verify embeddings cache exists:
   ```bash
   ls -la backend/.embeddings_cache/
   ```

### Coverage Report Issues

**Symptoms**:
```
Coverage report empty or incomplete
```

**Solutions**:
1. Run with explicit source:
   ```bash
   pytest --cov=app --cov-report=html
   ```
2. Check `.coveragerc` configuration:
   ```ini
   [run]
   source = app
   omit = */tests/*, */venv/*
   ```
3. Clean old coverage data:
   ```bash
   rm -rf .coverage htmlcov/
   pytest --cov=app --cov-report=html
   ```

---

## Reference Commands

### Quick Reference

```bash
# ============================================
# BACKEND TESTING (Python + pytest)
# ============================================

# Run all backend tests
cd backend
pytest

# Run specific test categories
pytest -m unit                              # Fast unit tests
pytest -m integration                       # Integration tests
pytest -m security                          # Security tests
pytest -m evaluation                        # Evaluation metrics tests
pytest -m performance                       # Performance tests

# Run specific test files
pytest tests/security/test_code_validator.py
pytest tests/security/test_sandbox_executor.py
pytest tests/security/test_security_integration.py
pytest tests/evaluation/test_retrieval_metrics.py
pytest tests/evaluation/test_generation_metrics.py
pytest tests/evaluation/test_llm_as_judge.py

# Run with options
pytest -vv                                  # Verbose output
pytest -x                                   # Stop on first failure
pytest -n auto                              # Parallel execution
pytest --lf                                 # Run last failed

# ============================================
# FRONTEND TESTING (TypeScript + Jest)
# ============================================

# Run all frontend tests
cd frontend
npm test

# Run specific test categories
npm test -- api/                            # API client tests
npm test -- store/                          # Store tests
npm test -- types/                          # Type definition tests
npm test -- components/                     # Component tests

# Run with options
npm run test:watch                          # Watch mode (auto-rerun)
npm run test:coverage                       # With coverage report
npm run test:ci                             # CI mode (non-interactive)

# Run specific test file
npm test -- api/__tests__/client.test.ts

# Run tests matching pattern
npm test -- --testNamePattern="streaming"

# Verbose output
npm test -- --verbose

# Debug mode (no coverage)
npm test -- --no-coverage --verbose

# ============================================
# COVERAGE
# ============================================

# Generate HTML coverage report
pytest --cov=app --cov-report=html
open htmlcov/index.html

# Terminal coverage report
pytest --cov=app --cov-report=term-missing

# Fail if coverage < 80%
pytest --cov=app --cov-fail-under=80

# ============================================
# EVALUATION & FINE-TUNING
# ============================================

# Full evaluation with export (OpenAI format)
python scripts/run_evaluation.py \
  --test-suite evaluation \
  --export-format openai \
  --output-dir ./fine_tuning_data \
  --min-score 0.80

# Agent-specific evaluation
python scripts/run_evaluation.py \
  --agent verification \
  --export-format openai \
  --output-dir ./verification_data

# Custom queries
python scripts/run_evaluation.py \
  --queries-file tests/fixtures/custom_queries.json \
  --ground-truth tests/fixtures/custom_answers.json \
  --export-format jsonl

# Export formats
python scripts/run_evaluation.py --export-format openai      # OpenAI fine-tuning
python scripts/run_evaluation.py --export-format anthropic   # Anthropic fine-tuning
python scripts/run_evaluation.py --export-format jsonl       # Generic JSONL
python scripts/run_evaluation.py --export-format csv         # CSV for analysis
python scripts/run_evaluation.py --export-format parquet     # Parquet for big data

# ============================================
# LANGSMITH
# ============================================

# View traces (web UI)
open https://smith.langchain.com

# Export traces (Python)
python -c "from langsmith import Client; \
  runs = Client().list_runs(project_name='govtech-multi-agent-system', limit=100); \
  print([r.id for r in runs])"

# ============================================
# DEBUGGING
# ============================================

# Run with debugging
pytest --pdb                                # Drop into debugger on failure
pytest -vv --tb=long                       # Full traceback
pytest --log-cli-level=DEBUG               # Debug logs

# Re-run failed tests
pytest --lf -vv

# Show local variables on failure
pytest -l

# ============================================
# CI/CD
# ============================================

# Backend: Run tests as in CI
cd backend
pytest tests/evaluation/ --junitxml=results.xml

# Skip slow tests (for fast CI)
pytest -m "not slow"

# Skip LLM tests (for offline CI)
pytest -m "not requires_llm"

# Frontend: Run tests as in CI
cd frontend
npm run test:ci                             # Jest with coverage in CI mode

# Both: Full CI pipeline locally
cd backend && pytest -m unit && cd ../frontend && npm test
```

### LangSmith Query Examples

```python
from langsmith import Client

client = Client()

# Get recent error runs
error_runs = client.list_runs(
    project_name="govtech-multi-agent-system",
    filter="status = 'error'",
    limit=50
)

# Get slow runs (> 10 seconds)
slow_runs = client.list_runs(
    project_name="govtech-multi-agent-system",
    filter="latency > 10000",  # milliseconds
    limit=50
)

# Get runs by agent
verification_runs = client.list_runs(
    project_name="govtech-multi-agent-system",
    filter="tags CONTAINS 'verification'",
    limit=100
)

# Export to DataFrame
import pandas as pd

runs = list(client.list_runs(project_name="govtech-multi-agent-system", limit=1000))
df = pd.DataFrame([{
    "id": r.id,
    "name": r.name,
    "status": r.status,
    "latency_ms": r.latency_ms,
    "tokens": r.total_tokens,
    "start_time": r.start_time,
} for r in runs])

df.to_csv("langsmith_export.csv", index=False)
```

---

## Additional Resources

- **Ragas Documentation**: https://docs.ragas.io/
- **BERTScore Paper**: https://arxiv.org/abs/1904.09675
- **LangSmith Documentation**: https://docs.smith.langchain.com/
- **OpenAI Fine-Tuning Guide**: https://platform.openai.com/docs/guides/fine-tuning
- **Anthropic API Reference**: https://docs.anthropic.com/claude/reference

---

## Feedback and Issues

For questions or issues with testing:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review test logs with `pytest -vv --log-cli-level=DEBUG`
3. Open an issue on GitHub with test output and environment details

---

**Last Updated**: 2026-02-11

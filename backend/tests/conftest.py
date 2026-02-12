"""
Pytest configuration and fixtures for backend tests.

This module provides shared fixtures for:
- Database sessions with rollback
- Sample datasets
- Mocked services (RAG, LLM)
- Test data loading
- Evaluation tools (Ragas, BERTScore, LLM Judge)
"""

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)

# Test configuration paths
TEST_DIR = Path(__file__).parent
FIXTURES_DIR = TEST_DIR / "fixtures"
TEST_CONFIG_PATH = TEST_DIR / "test_config.yaml"

# Load test configuration
with open(TEST_CONFIG_PATH) as f:
    TEST_CONFIG = yaml.safe_load(f)["testing"]


# ============================================================================
# Session-scoped fixtures (expensive setup, reused across tests)
# ============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config() -> dict[str, Any]:
    """Load test configuration."""
    return TEST_CONFIG


@pytest.fixture(scope="session")
async def test_db_engine():
    """
    Create test database engine.

    Uses separate test database to avoid polluting production data.
    """
    db_url = os.getenv(
        "TEST_DATABASE_URL", "postgresql+asyncpg://govtech:govtech_dev@postgres:5432/govtech_rag"
    )

    engine = create_async_engine(
        db_url,
        echo=False,
        poolclass=NullPool,  # No connection pooling for tests
    )

    yield engine

    await engine.dispose()


@pytest.fixture(scope="session")
async def setup_test_database(test_db_engine, request):
    """
    Set up test database schema.

    For evaluation tests: Creates schema but preserves data (populated by ingest_test_data)
    For unit tests: Drops and recreates schema for isolation
    """
    from sqlalchemy import text

    import app.db.session as db_session
    from app.db.models import Base

    # Check if we're running evaluation tests
    is_evaluation = any("evaluation" in str(item.fspath) for item in request.session.items)

    async with test_db_engine.begin() as conn:
        if not is_evaluation:
            # Unit/integration tests: Fresh database
            await conn.run_sync(Base.metadata.drop_all)

        # Create schema (CREATE TABLE IF NOT EXISTS)
        await conn.run_sync(Base.metadata.create_all)

        # Ensure pgvector extension exists
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Create indexes if they don't exist (IVFFlat requires data to exist first)
        try:
            await conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS idx_employment_embedding
                ON employment_dataset_metadata
                USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
            """
                )
            )
            await conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS idx_income_embedding
                ON income_dataset_metadata
                USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
            """
                )
            )
            await conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS idx_hours_embedding
                ON hours_worked_dataset_metadata
                USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
            """
                )
            )
        except Exception as e:
            # Indexes might fail if no data exists yet, that's okay
            logger.warning(f"Could not create vector indexes (may not have data yet): {e}")

    # Initialize global database session factory for services (RAGService, etc.)
    db_session.engine = test_db_engine
    db_session.async_session_factory = async_sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    logger.info("Global database session factory initialized for tests")

    yield

    # Cleanup global session factory
    db_session.async_session_factory = None
    db_session.engine = None

    # Cleanup after all tests (only for non-evaluation tests)
    if not is_evaluation:
        async with test_db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)


# ============================================================================
# Function-scoped fixtures (fresh state for each test)
# ============================================================================


@pytest.fixture
async def async_db_session(
    test_db_engine, setup_test_database
) -> AsyncGenerator[AsyncSession, None]:
    """
    Provide async database session with automatic rollback.

    Each test gets a fresh database state. Changes are rolled back
    after the test completes.
    """
    async_session_factory = async_sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_factory() as session, session.begin():
        yield session
        # Rollback happens automatically when context exits


@pytest.fixture(scope="session")
async def ingest_test_data(setup_test_database, test_db_engine):
    """
    Auto-ingest data for evaluation tests.

    Runs data ingestion pipeline if database is empty.
    Uses same ingestion logic as production startup.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.services.ingestion.data_processor import DataProcessor
    from app.services.ingestion.embedding_generator import EmbeddingGenerator

    # Create a session for ingestion
    async_session_factory = async_sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_factory() as session:
        try:
            # Check if data already exists
            result = await session.execute(text("SELECT COUNT(*) FROM employment_dataset_metadata"))
            count = result.scalar()

            if count > 0:
                logger.info(
                    f"Test data already exists ({count} employment datasets), skipping ingestion"
                )
                return

            logger.info("No test data found, running ingestion...")

            # Run data ingestion
            processor = DataProcessor()
            results = await processor.process_all_datasets(session)

            logger.info(f"Ingested {results.get('metadata_entries', 0)} metadata entries")

            # Generate embeddings
            embedder = EmbeddingGenerator()
            embedding_results = await embedder.update_embeddings(session)

            logger.info(f"Generated {embedding_results.get('total_updated', 0)} embeddings")

            await session.commit()
            logger.info("Test data ingestion complete")

        except Exception as e:
            logger.error(f"Test data ingestion failed: {e}")
            await session.rollback()
            raise


@pytest.fixture
async def sample_datasets(async_db_session: AsyncSession, ingest_test_data) -> dict[str, Any]:
    """
    Provide sample datasets for evaluation tests.

    Depends on ingest_test_data fixture to ensure data is loaded.
    """
    from sqlalchemy import text

    # Query counts from all metadata tables
    result = await async_db_session.execute(text("SELECT COUNT(*) FROM income_dataset_metadata"))
    income_count = result.scalar()

    result = await async_db_session.execute(
        text("SELECT COUNT(*) FROM employment_dataset_metadata")
    )
    employment_count = result.scalar()

    result = await async_db_session.execute(
        text("SELECT COUNT(*) FROM hours_worked_dataset_metadata")
    )
    hours_count = result.scalar()

    total_count = income_count + employment_count + hours_count

    if total_count == 0:
        # This should rarely happen now that we have auto-ingestion
        pytest.skip(
            "No datasets found in database after ingestion. " "Check ingestion logs for errors."
        )

    return {
        "dataset_count": total_count,
        "income_count": income_count,
        "employment_count": employment_count,
        "hours_count": hours_count,
    }


@pytest.fixture
def mock_rag_service() -> Mock:
    """
    Provide mocked RAG service for fast unit tests.

    Returns deterministic results without database or embedding calls.
    """
    mock_service = Mock()
    mock_service.search_datasets = AsyncMock(
        return_value=[
            {
                "dataset_name": "income_from_work_2020",
                "description": "Average income from work by sex and age group in 2020",
                "score": 0.85,
                "file_path": "/path/to/income_2020.csv",
            }
        ]
    )
    mock_service.get_dataset_metadata = AsyncMock(
        return_value={
            "name": "income_from_work_2020",
            "description": "Income data",
            "columns": ["age_group", "sex", "average_income"],
            "years": [2020],
            "categories": ["income"],
        }
    )
    return mock_service


@pytest.fixture
def mock_llm() -> Mock:
    """
    Provide mocked LLM for deterministic tests.

    Returns pre-defined responses without actual API calls.
    """
    mock_llm = Mock()
    mock_llm.invoke = Mock(return_value=Mock(content="Mocked LLM response for testing purposes."))
    mock_llm.ainvoke = AsyncMock(
        return_value=Mock(content="Mocked LLM response for testing purposes.")
    )
    return mock_llm


@pytest.fixture
def mock_orchestrator():
    """
    Provide mocked orchestrator for integration tests.

    Returns realistic responses based on query patterns without LLM calls.
    """
    from tests.fixtures.mock_llm_responses import MockLLMResponses

    mock_orch = Mock()

    async def mock_execute(query: str):
        return MockLLMResponses.orchestrator_response(query)

    async def mock_process_query(query: str):
        return MockLLMResponses.orchestrator_response(query)

    mock_orch.execute = AsyncMock(side_effect=mock_execute)
    mock_orch.process_query = AsyncMock(side_effect=mock_process_query)

    return mock_orch


@pytest.fixture
def mock_agents():
    """
    Provide mocked agents for integration tests.

    Returns realistic responses based on agent type without LLM calls.
    """
    from unittest.mock import Mock

    from tests.fixtures.mock_llm_responses import get_mock_llm_response

    class MockAgent:
        def __init__(self, agent_type: str):
            self.agent_type = agent_type
            self.config = Mock()

        async def execute(self, state):
            query = state.get("current_task") or state.get("query", "")
            response_data = get_mock_llm_response(self.agent_type, query)

            # Return mock response object
            response = Mock()
            response.success = (
                response_data.get("success", True)
                if "success" in response_data
                else not response_data.get("error")
            )
            response.data = response_data
            response.message = response_data.get("message", "")
            return response

    return {
        "verification": MockAgent("verification"),
        "coordinator": MockAgent("coordinator"),
        "extraction": MockAgent("extraction"),
        "analytics": MockAgent("analytics"),
    }


@pytest.fixture
def sample_queries(test_config: dict[str, Any]) -> dict[str, list[dict]]:
    """
    Load sample test queries from JSON fixture.

    Returns queries categorized by test type.
    """
    queries_file = TEST_DIR / test_config["fixtures"]["queries_file"]
    if queries_file.exists():
        with open(queries_file) as f:
            return json.load(f)
    return {
        "retrieval_tests": [],
        "generation_tests": [],
        "agent_tests": {},
    }


@pytest.fixture
def ground_truth_contexts(test_config: dict[str, Any]) -> dict[str, Any]:
    """Load ground truth contexts for retrieval evaluation."""
    contexts_file = TEST_DIR / test_config["fixtures"]["contexts_file"]
    if contexts_file.exists():
        with open(contexts_file) as f:
            return json.load(f)
    return {}


@pytest.fixture
def ground_truth_answers(test_config: dict[str, Any]) -> dict[str, Any]:
    """Load ground truth answers for generation evaluation."""
    answers_file = TEST_DIR / test_config["fixtures"]["answers_file"]
    if answers_file.exists():
        with open(answers_file) as f:
            return json.load(f)
    return {}


# ============================================================================
# Evaluation tool fixtures
# ============================================================================


@pytest.fixture(scope="session")
def ragas_evaluator(test_config: dict[str, Any]):
    """
    Provide Ragas evaluator instance.

    Configured with test-specific settings for RAG evaluation.
    """
    if not test_config["ragas"]["enabled"]:
        pytest.skip("Ragas evaluation disabled in test config")

    from tests.utils.ragas_evaluator import RagasEvaluator

    return RagasEvaluator(test_config["ragas"])


@pytest.fixture(scope="session")
def bertscore_evaluator(test_config: dict[str, Any]):
    """
    Provide BERTScore evaluator instance.

    Configured with test-specific model and settings.
    """
    if not test_config["bertscore"]["enabled"]:
        pytest.skip("BERTScore evaluation disabled in test config")

    from tests.utils.bertscore_evaluator import BERTScoreEvaluator

    return BERTScoreEvaluator(test_config["bertscore"])


@pytest.fixture
def llm_judge(test_config: dict[str, Any]):
    """
    Provide LLM as judge evaluator.

    Uses Claude Sonnet for structured quality evaluation.
    """
    if not test_config["llm_judge"]["enabled"]:
        pytest.skip("LLM judge evaluation disabled in test config")

    from tests.utils.llm_judge import LLMJudge

    return LLMJudge(test_config["llm_judge"])


# ============================================================================
# Helper fixtures
# ============================================================================


@pytest.fixture
def temp_dataset_file(tmp_path: Path) -> Path:
    """
    Create temporary CSV file for testing data loading.

    Returns path to temporary file.
    """
    csv_content = """age_group,sex,average_income
25-34,Male,4500
25-34,Female,4200
35-44,Male,5800
35-44,Female,5200
"""
    file_path = tmp_path / "test_data.csv"
    file_path.write_text(csv_content)
    return file_path


@pytest.fixture
async def mock_graph_state() -> dict[str, Any]:
    """
    Provide mock GraphState for agent testing.

    Returns minimal valid state for agent execution.
    """
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content="What was the average income in 2020?")],
        "current_task": "data_extraction",
        "query": "What was the average income in 2020?",
        "extracted_data": {},  # Dict, not list
        "analysis_results": {},
        "workflow_plan": [],  # List, not dict
        "current_step": 0,
        "errors": [],
        "metadata": {},
        "intermediate_results": {},
        "should_continue": True,
        "retrieval_context": {},  # Dict, not list
        "query_validation": {
            "is_valid": True,
            "topic": "income",
            "specified_year": 2020,
        },
        "available_years": {},  # Dict mapping categories to year ranges
    }


@pytest.fixture
def mock_table_schemas() -> list[dict[str, Any]]:
    """
    Provide mock table schemas for extraction agent testing.

    Returns list of TableSchema-compatible dictionaries.
    """
    return [
        {
            "table_name": "income_2020",
            "description": "Income from work by age and sex in 2020",
            "columns": [
                {"name": "age_group", "type": "VARCHAR", "description": "Age group category"},
                {"name": "sex", "type": "VARCHAR", "description": "Gender"},
                {
                    "name": "average_income",
                    "type": "NUMERIC",
                    "description": "Average monthly income",
                },
            ],
            "file_path": "/path/to/income_2020.csv",
            "score": 0.87,
            "metadata": {
                "year": 2020,
                "category": "income",
                "source": "Ministry of Manpower",
            },
        }
    ]


# ============================================================================
# Pytest hooks and configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "slow: mark test as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "requires_llm: mark test as requiring LLM API calls")
    config.addinivalue_line("markers", "requires_db: mark test as requiring database")


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection based on config and markers.

    Automatically applies markers based on test path and dependencies.
    """
    for item in items:
        # Auto-apply markers based on test path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "evaluation" in str(item.fspath):
            item.add_marker(pytest.mark.evaluation)
            item.add_marker(pytest.mark.slow)

        # Add requires_llm marker to evaluation tests
        if "evaluation" in str(item.fspath) or "llm" in item.name:
            item.add_marker(pytest.mark.requires_llm)

        # Add requires_db marker to tests using db fixtures
        if "async_db_session" in item.fixturenames:
            item.add_marker(pytest.mark.requires_db)


@pytest.fixture(autouse=True)
def reset_singletons():
    """
    Reset singleton instances between tests.

    Prevents state leakage between tests.
    """
    # Note: Current config.py uses global instance without cache
    # If caching is added later, clear it here

    yield

    # Cleanup after test if needed


# ============================================================================
# Utility functions for tests
# ============================================================================


@pytest.fixture
def assert_similarity():
    """
    Provide assertion helper for semantic similarity.

    Useful for comparing LLM outputs with expected results.
    """

    def _assert_similarity(text1: str, text2: str, threshold: float = 0.8):
        """Assert two texts are semantically similar."""
        from sentence_transformers import SentenceTransformer, util

        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb1 = model.encode(text1, convert_to_tensor=True)
        emb2 = model.encode(text2, convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2).item()

        assert similarity >= threshold, (
            f"Texts not similar enough: {similarity:.3f} < {threshold}\n"
            f"Text 1: {text1}\n"
            f"Text 2: {text2}"
        )

    return _assert_similarity

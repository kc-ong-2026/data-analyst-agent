"""
Fast Integration Smoke Tests.

Quick tests to verify integration test infrastructure without heavy initialization.
These tests are fast and don't load models or make API calls.
"""

import pytest


@pytest.mark.integration
@pytest.mark.fast
@pytest.mark.asyncio
class TestDatabaseIntegration:
    """Test database connectivity (fast, no heavy loading)."""

    async def test_database_connection(self, async_db_session):
        """Test that database connection works."""
        from sqlalchemy import text

        # Simple query to verify DB connection
        result = await async_db_session.execute(text("SELECT 1 as test"))
        row = result.fetchone()
        assert row[0] == 1
        print("✅ Database connection working")

    async def test_dataset_metadata_table_exists(self, async_db_session):
        """Test that dataset_metadata table exists."""
        from sqlalchemy import text

        result = await async_db_session.execute(
            text(
                """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'dataset_metadata'
            )
            """
            )
        )
        exists = result.scalar()
        assert exists is True
        print("✅ dataset_metadata table exists")


@pytest.mark.integration
@pytest.mark.fast
class TestImportsWork:
    """Test that key modules can be imported."""

    def test_can_import_orchestrator(self):
        """Test orchestrator can be imported."""
        from app.services.agents.orchestrator import AgentOrchestrator

        assert AgentOrchestrator is not None
        print("✅ Orchestrator imports successfully")

    def test_can_import_agents(self):
        """Test agents can be imported."""
        from app.services.agents.analytics import AnalyticsAgent
        from app.services.agents.coordinator import DataCoordinatorAgent
        from app.services.agents.extraction import DataExtractionAgent
        from app.services.agents.verification import QueryVerificationAgent

        assert QueryVerificationAgent is not None
        assert DataCoordinatorAgent is not None
        assert DataExtractionAgent is not None
        assert AnalyticsAgent is not None
        print("✅ All agents import successfully")

    def test_can_import_rag_service(self):
        """Test RAG service can be imported."""
        from app.services.rag_service import RAGService

        assert RAGService is not None
        print("✅ RAG service imports successfully")

    def test_can_import_config(self):
        """Test config can be imported and loaded."""
        from app.config import get_config

        config = get_config()
        assert config is not None
        assert hasattr(config, "get_llm_config")
        assert hasattr(config, "get_rag_config")
        assert hasattr(config, "yaml_config")
        print("✅ Configuration loads successfully")


@pytest.mark.integration
@pytest.mark.fast
class TestConfigurationLoading:
    """Test configuration loading (fast, no heavy operations)."""

    def test_config_loads_successfully(self):
        """Test that configuration loads."""
        from app.config import get_config

        config = get_config()

        assert config is not None
        assert hasattr(config, "get_llm_config")
        assert hasattr(config, "get_rag_config")
        assert hasattr(config, "yaml_config")
        print("✅ Configuration loaded")

    def test_rag_config_has_required_fields(self):
        """Test RAG config has required fields."""
        from app.config import get_config

        config = get_config()
        rag_config = config.get_rag_config()

        required_fields = [
            "use_reranking",
            "use_bm25",
            "confidence_threshold",
            "min_datasets",
            "max_datasets",
        ]

        for field in required_fields:
            assert field in rag_config, f"Missing field: {field}"

        print("✅ RAG config has all required fields")

    def test_llm_config_has_required_fields(self):
        """Test LLM config has required fields."""
        from app.config import get_config

        config = get_config()
        llm_config = config.get_llm_config()

        # Check that config returns expected fields
        required_fields = ["provider", "model", "temperature", "max_tokens"]
        for field in required_fields:
            assert field in llm_config, f"Missing field: {field}"

        print("✅ LLM config has required fields")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

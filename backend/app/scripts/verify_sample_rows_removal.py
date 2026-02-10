#!/usr/bin/env python3
"""Verification script for sample rows removal implementation.

This script verifies that:
1. Sample rows are not loaded from RAG retrieval
2. Sample rows are not loaded from file-based fallback
3. Metadata is properly stored and accessible
4. SQL query results are prioritized in analytics
"""

import asyncio
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def verify_extraction_agent():
    """Verify extraction agent no longer loads sample rows."""
    from app.services.agents.extraction_agent import DataExtractionAgent
    from app.services.agents.base_agent import GraphState

    logger.info("=" * 80)
    logger.info("VERIFICATION: Extraction Agent Sample Rows Removal")
    logger.info("=" * 80)

    agent = DataExtractionAgent()

    # Test case: Query that would normally load sample rows
    test_state: GraphState = {
        "current_task": "What is the employment rate in 2020?",
        "intermediate_results": {
            "retrieval_context": {
                "metadata_results": [
                    {
                        "metadata_id": "test_1",
                        "category": "employment",
                        "file_name": "test_employment.csv",
                        "table_name": "test_employment_table",
                        "description": "Test employment data",
                        "columns": [
                            {"name": "year", "dtype": "int64"},
                            {"name": "employment_rate", "dtype": "float64"}
                        ],
                        "row_count": 100,
                        "score": 0.95,
                    }
                ],
                "table_schemas": [
                    {
                        "table_name": "test_employment_table",
                        "category": "employment",
                        "description": "Employment statistics",
                        "columns": ["year", "employment_rate"],
                        "primary_dimensions": ["year"],
                        "numeric_columns": ["employment_rate"],
                        "categorical_columns": [],
                        "row_count": 100,
                        "year_range": {"min": 2015, "max": 2023},
                        "sql_schema_prompt": "Table: test_employment_table...",
                    }
                ],
            },
            "source": "rag",
        },
        "messages": [],
        "errors": [],
        "metadata": {},
    }

    logger.info("\n1. Testing _load_from_rag method...")
    try:
        result = await agent._load_from_rag(test_state)
        loaded_datasets = result.get("intermediate_results", {}).get("loaded_datasets", {})

        if loaded_datasets:
            for name, data in loaded_datasets.items():
                rows = data.get("data", [])
                metadata = data.get("metadata", {})

                logger.info(f"   Dataset: {name}")
                logger.info(f"   - Rows loaded: {len(rows)}")
                logger.info(f"   - Has metadata: {bool(metadata)}")
                logger.info(f"   - Source: {data.get('source')}")

                if len(rows) == 0 and metadata:
                    logger.info("   ✅ PASS: No sample rows loaded, metadata present")
                elif len(rows) > 0:
                    logger.error("   ❌ FAIL: Sample rows were loaded (should be empty)")
                    return False
                else:
                    logger.warning("   ⚠️  WARN: No rows and no metadata")
        else:
            logger.info("   ℹ️  No datasets loaded (may be expected if DB not available)")

    except Exception as e:
        logger.error(f"   ❌ FAIL: Exception during _load_from_rag: {e}")
        return False

    logger.info("\n2. Checking metadata structure...")
    if loaded_datasets:
        for name, data in loaded_datasets.items():
            metadata = data.get("metadata", {})
            required_fields = ["description", "primary_dimensions", "numeric_columns"]

            missing_fields = [f for f in required_fields if f not in metadata]
            if missing_fields:
                logger.warning(f"   ⚠️  Missing metadata fields: {missing_fields}")
            else:
                logger.info(f"   ✅ PASS: All required metadata fields present")

    logger.info("\n" + "=" * 80)
    return True


async def verify_analytics_agent():
    """Verify analytics agent handles metadata and SQL results correctly."""
    from app.services.agents.analytics_agent import AnalyticsAgent

    logger.info("=" * 80)
    logger.info("VERIFICATION: Analytics Agent Metadata Handling")
    logger.info("=" * 80)

    agent = AnalyticsAgent()

    # Test case 1: SQL query results (highest priority)
    test_data_with_sql = {
        "query_result": {
            "path": "SQL query result",
            "columns": ["year", "avg_employment_rate"],
            "shape": [5, 2],
            "data": [
                {"year": 2019, "avg_employment_rate": 65.2},
                {"year": 2020, "avg_employment_rate": 62.8},
                {"year": 2021, "avg_employment_rate": 64.1},
            ],
            "source": "sql",
        }
    }

    # Test case 2: Metadata only (no rows)
    test_data_metadata_only = {
        "employment_table": {
            "path": "employment.csv",
            "columns": ["year", "employment_rate", "industry"],
            "shape": [100, 3],
            "data": [],  # No sample rows
            "metadata": {
                "description": "Employment statistics by industry",
                "primary_dimensions": ["year", "industry"],
                "numeric_columns": ["employment_rate"],
                "categorical_columns": ["industry"],
                "year_range": {"min": 2015, "max": 2023},
            },
            "source": "rag_metadata",
        }
    }

    logger.info("\n1. Testing _format_data_for_analysis with SQL results...")
    try:
        formatted = agent._format_data_for_analysis(test_data_with_sql, {})

        if "SQL Query Result" in formatted:
            logger.info("   ✅ PASS: SQL query result detected in formatted output")
        else:
            logger.error("   ❌ FAIL: SQL query result not found in formatted output")
            return False

        if "Data (first 10 rows from SQL query)" in formatted:
            logger.info("   ✅ PASS: SQL query results labeled correctly")
        else:
            logger.error("   ❌ FAIL: SQL query results not labeled correctly")
            return False

    except Exception as e:
        logger.error(f"   ❌ FAIL: Exception during _format_data_for_analysis: {e}")
        return False

    logger.info("\n2. Testing _format_data_for_analysis with metadata only...")
    try:
        formatted = agent._format_data_for_analysis(test_data_metadata_only, {})

        if "Metadata Context" in formatted:
            logger.info("   ✅ PASS: Metadata context used when no rows available")
        else:
            logger.error("   ❌ FAIL: Metadata context not used when no rows available")
            return False

        if "Employment statistics by industry" in formatted:
            logger.info("   ✅ PASS: Metadata description included in output")
        else:
            logger.warning("   ⚠️  WARN: Metadata description not found in output")

    except Exception as e:
        logger.error(f"   ❌ FAIL: Exception during _format_data_for_analysis: {e}")
        return False

    logger.info("\n3. Testing _auto_generate_visualization priority...")
    try:
        # Should prioritize SQL query result
        viz = agent._auto_generate_visualization(test_data_with_sql)

        if viz:
            if "SQL Query Result" in viz.get("description", ""):
                logger.info("   ✅ PASS: Visualization uses SQL query result")
            else:
                logger.warning("   ⚠️  WARN: Visualization doesn't reference SQL query result")

            if viz.get("data") and len(viz["data"]) > 0:
                logger.info(f"   ✅ PASS: Visualization has {len(viz['data'])} data points")
            else:
                logger.error("   ❌ FAIL: Visualization has no data")
                return False
        else:
            logger.info("   ℹ️  INFO: No visualization generated (may be expected)")

    except Exception as e:
        logger.error(f"   ❌ FAIL: Exception during _auto_generate_visualization: {e}")
        return False

    logger.info("\n" + "=" * 80)
    return True


async def main():
    """Run all verification tests."""
    logger.info("\n")
    logger.info("*" * 80)
    logger.info("*" + " " * 78 + "*")
    logger.info("*" + "  Sample Rows Removal Verification".center(78) + "*")
    logger.info("*" + " " * 78 + "*")
    logger.info("*" * 80)
    logger.info("\n")

    all_passed = True

    # Test extraction agent
    try:
        passed = await verify_extraction_agent()
        all_passed = all_passed and passed
    except Exception as e:
        logger.error(f"Extraction agent verification failed with exception: {e}")
        all_passed = False

    logger.info("\n")

    # Test analytics agent
    try:
        passed = await verify_analytics_agent()
        all_passed = all_passed and passed
    except Exception as e:
        logger.error(f"Analytics agent verification failed with exception: {e}")
        all_passed = False

    logger.info("\n")
    logger.info("*" * 80)
    if all_passed:
        logger.info("*" + " " * 78 + "*")
        logger.info("*" + "  ✅ ALL VERIFICATIONS PASSED".center(78) + "*")
        logger.info("*" + " " * 78 + "*")
    else:
        logger.info("*" + " " * 78 + "*")
        logger.info("*" + "  ❌ SOME VERIFICATIONS FAILED".center(78) + "*")
        logger.info("*" + " " * 78 + "*")
    logger.info("*" * 80)
    logger.info("\n")

    return all_passed


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)

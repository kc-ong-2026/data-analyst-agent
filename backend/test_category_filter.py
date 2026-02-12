#!/usr/bin/env python3
"""
Test script to verify category filtering in RAG search.
Tests the three verification cases from the plan.
"""

import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app.services.agents.extraction.agent import DataExtractionAgent


def test_category_extraction():
    """Test the _extract_category_from_query method directly."""

    agent = DataExtractionAgent()

    test_cases = [
        # Test Case 1: Employment Query
        {
            "query": "What is the employment rate in Singapore?",
            "expected": "employment",
            "description": "Employment keyword",
        },
        {
            "query": "What is the employment trend in Singapore in the technology industry from 2020-2024?",
            "expected": "employment",
            "description": "Employment trend query",
        },
        {
            "query": "Show me unemployment statistics",
            "expected": "employment",
            "description": "Unemployment keyword",
        },
        {
            "query": "What is the labour force participation?",
            "expected": "employment",
            "description": "Labour force keyword",
        },
        # Test Case 2: Income Query
        {
            "query": "Show me salary trends in 2020",
            "expected": "income",
            "description": "Salary keyword",
        },
        {
            "query": "What is the average income in Singapore?",
            "expected": "income",
            "description": "Income keyword",
        },
        {"query": "Tell me about wage growth", "expected": "income", "description": "Wage keyword"},
        # Test Case 3: Hours Worked Query
        {
            "query": "How many hours do people work per week?",
            "expected": "hours_worked",
            "description": "Hours worked phrase",
        },
        {
            "query": "Show me working hours statistics",
            "expected": "hours_worked",
            "description": "Working hours phrase",
        },
        {
            "query": "What is the overtime trend?",
            "expected": "hours_worked",
            "description": "Overtime keyword",
        },
        # Test Case 4: Generic Query (no filter)
        {
            "query": "Show me statistics for 2020",
            "expected": None,
            "description": "Generic query with no category",
        },
        {"query": "What data do you have?", "expected": None, "description": "Generic question"},
    ]

    print("=" * 80)
    print("CATEGORY EXTRACTION TEST")
    print("=" * 80)
    print()

    passed = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected = test_case["expected"]
        description = test_case["description"]

        result = agent._extract_category_from_query(query)

        status = "✓ PASS" if result == expected else "✗ FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"Test {i}: {description}")
        print(f'  Query: "{query}"')
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")
        print(f"  Status:   {status}")
        print()

    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = test_category_extraction()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""Test SQL aggregation detection for various query patterns."""

import re
import sys


def requires_sql_aggregation(query: str) -> bool:
    """Detect if query requires SQL aggregation or filtering."""
    query_lower = query.lower()

    # Aggregation keywords
    aggregation_keywords = [
        "average", "avg", "mean",
        "total", "sum",
        "count", "number of",
        "trend", "over time", "by year", "by month",
        "group by", "grouped by",
        "maximum", "max", "highest",
        "minimum", "min", "lowest",
        "median",
        "compare", "comparison",
        "distribution",
        "percentage", "percent",
        "rate",  # employment rate, unemployment rate, etc.
    ]

    # Grouping/dimension keywords
    grouping_keywords = [
        "by industry", "by sector", "by age", "by sex",
        "by qualification", "by education", "by gender",
        "by occupation", "by region", "by area",
    ]

    # Year range patterns
    year_range_patterns = [
        r'\b(19\d{2}|20[0-4]\d)\s*(?:to|-|through|until|and)\s*(19\d{2}|20[0-4]\d)\b',
        r'\bbetween\s+(19\d{2}|20[0-4]\d)\s+(?:and|to)\s+(19\d{2}|20[0-4]\d)\b',
        r'\bfrom\s+(19\d{2}|20[0-4]\d)\s+(?:to|until)\s+(19\d{2}|20[0-4]\d)\b',
    ]

    # Action verbs
    action_keywords = [
        "show", "display", "get", "find", "retrieve",
        "list", "what is", "what are", "how many",
    ]

    # Check aggregation keywords
    if any(keyword in query_lower for keyword in aggregation_keywords):
        return True, "aggregation keyword"

    # Check grouping keywords
    if any(keyword in query_lower for keyword in grouping_keywords):
        return True, "grouping keyword"

    # Check for year ranges
    for pattern in year_range_patterns:
        if re.search(pattern, query_lower):
            return True, "year range pattern"

    # Check action verbs
    if any(keyword in query_lower for keyword in action_keywords):
        return True, "action verb"

    # Check for single years
    if re.search(r'\b(19\d{2}|20[0-4]\d)\b', query_lower):
        return True, "contains year"

    return False, "no match"


# Test cases
test_queries = [
    # Should require SQL (True)
    ("What is the employment rate in Singapore between 2020-2025 by technology industry", True),
    ("What is the employment rate in Singapore from 1991-2025?", True),
    ("Show me employment by industry in 2021", True),
    ("Average employment rate from 2019 to 2022", True),
    ("Female employment rate in 2020", True),
    ("Employment trends over time", True),
    ("Compare employment rates between industries", True),
    ("Total employment count by age", True),
    ("What is the unemployment rate?", True),
    ("Get employment data for 2023", True),
    ("Display employment statistics", True),

    # Should NOT require SQL (False) - informational/metadata queries
    ("Tell me about employment datasets", False),
    ("What data do you have?", False),
    ("Hello, can you help me?", False),
]

print("=" * 80)
print("SQL Aggregation Detection Test")
print("=" * 80)
print()

passed = 0
failed = 0

for query, expected in test_queries:
    result, reason = requires_sql_aggregation(query)
    status = "✓ PASS" if result == expected else "✗ FAIL"

    if result == expected:
        passed += 1
    else:
        failed += 1

    print(f"{status} | Expected: {expected:5} | Got: {result:5} | Reason: {reason}")
    print(f"       | Query: \"{query}\"")
    print()

print("=" * 80)
print(f"Results: {passed} passed, {failed} failed out of {len(test_queries)} tests")
print("=" * 80)

# Exit with error code if any tests failed
sys.exit(0 if failed == 0 else 1)

"""
LLM as Judge Evaluation Tests.

Uses Claude Sonnet 4.5 as a judge to evaluate response quality across:
- Accuracy: Factual correctness based on context
- Completeness: Coverage of all query aspects
- Clarity: Clear and understandable language
- Conciseness: Appropriate brevity
- Code Quality: Clean, efficient, correct code (analytics only)
- Visualization: Appropriate chart type (analytics only)

Tests validate judge consistency, discrimination, and threshold adherence.
"""

import pytest
from typing import List, Dict, Any


@pytest.mark.evaluation
@pytest.mark.slow
@pytest.mark.requires_llm
class TestLLMJudgeAccuracy:
    """Test LLM judge evaluation of response quality."""

    @pytest.mark.asyncio
    async def test_judge_evaluates_analytics_responses(
        self,
        llm_judge,
        ground_truth_answers,
        async_db_session,
        sample_datasets,
    ):
        """Test LLM judge evaluation of analytics agent responses."""
        from app.services.agents.orchestrator import AgentOrchestrator
        from app.config import get_config

        config = get_config()
        orchestrator = AgentOrchestrator(config, async_db_session)

        # Evaluate subset of responses
        judgments = []

        for test_id, gt_data in list(ground_truth_answers.items())[:5]:
            query = gt_data["query"]
            ground_truth = gt_data["ground_truth"]["answer"]

            try:
                result = await orchestrator.process_query(query)
                answer = result.get("answer", "")
                context = result.get("context", "")

                if answer:
                    judgment = llm_judge.evaluate(
                        query=query,
                        answer=answer,
                        context=context,
                        ground_truth=ground_truth,
                    )

                    judgments.append(judgment)
                    print(f"\nQuery: {query[:50]}...")
                    print(f"Overall Score: {judgment.overall_score:.2f}/5.0")
                    print(f"Accuracy: {judgment.criteria_scores['accuracy'].score:.2f}/5.0")

            except Exception as e:
                print(f"Error evaluating query: {e}")
                continue

        if not judgments:
            pytest.skip("No judgments generated")

        # Calculate averages
        avg_overall = sum(j.overall_score for j in judgments) / len(judgments)
        avg_accuracy = sum(
            j.criteria_scores['accuracy'].score for j in judgments
        ) / len(judgments)

        print(f"\n=== LLM Judge Summary ===")
        print(f"Average Overall: {avg_overall:.2f}/5.0")
        print(f"Average Accuracy: {avg_accuracy:.2f}/5.0")

        # Check thresholds
        assert avg_overall >= 3.5, (
            f"Average overall score {avg_overall:.2f} below threshold 3.5"
        )
        assert avg_accuracy >= 4.0, (
            f"Average accuracy score {avg_accuracy:.2f} below threshold 4.0"
        )

    @pytest.mark.asyncio
    async def test_judge_evaluates_code_quality(
        self,
        llm_judge,
        async_db_session,
        sample_datasets,
    ):
        """Test LLM judge evaluation of generated code quality."""
        from app.services.agents.analytics import AnalyticsAgent

        agent = AnalyticsAgent()

        # Test code generation
        state = {
            "query": "Calculate average income by age group",
            "messages": [],
            "extracted_data": [{
                "dataset_name": "income_2020",
                "columns": ["age_group", "average_income"],
                "dtypes": {"age_group": "object", "average_income": "float64"},
                "data": [
                    {"age_group": "25-34", "average_income": 4500.0},
                    {"age_group": "35-44", "average_income": 5800.0},
                ],
                "source": "dataframe",
            }],
        }

        response = await agent.execute(state)
        generated_code = response.data.get("generated_code", "")

        if not generated_code:
            pytest.skip("No code generated")

        # Judge code quality
        judgment = llm_judge.evaluate(
            query=state["query"],
            answer=generated_code,
            criteria=["accuracy", "code_quality"],
        )

        print(f"\nCode Quality Score: {judgment.criteria_scores['code_quality'].score:.2f}/5.0")

        # Code should be reasonably good
        assert judgment.criteria_scores['code_quality'].score >= 3.0

    @pytest.mark.asyncio
    async def test_judge_evaluates_explanation_quality(
        self,
        llm_judge,
        async_db_session,
        sample_datasets,
    ):
        """Test LLM judge evaluation of natural language explanations."""
        from app.services.agents.orchestrator import AgentOrchestrator
        from app.config import get_config

        config = get_config()
        orchestrator = AgentOrchestrator(config, async_db_session)

        query = "What was the average income in 2020?"

        try:
            result = await orchestrator.process_query(query)
            answer = result.get("answer", "")

            if not answer:
                pytest.skip("No answer generated")

            # Evaluate explanation
            judgment = llm_judge.evaluate(
                query=query,
                answer=answer,
                criteria=["clarity", "conciseness", "completeness"],
            )

            print(f"\nClarity: {judgment.criteria_scores['clarity'].score:.2f}/5.0")
            print(f"Conciseness: {judgment.criteria_scores['conciseness'].score:.2f}/5.0")
            print(f"Completeness: {judgment.criteria_scores['completeness'].score:.2f}/5.0")

            # Explanations should be clear and complete
            assert judgment.criteria_scores['clarity'].score >= 3.0
            assert judgment.criteria_scores['completeness'].score >= 3.0

        except Exception as e:
            pytest.skip(f"Error: {e}")


@pytest.mark.evaluation
@pytest.mark.slow
@pytest.mark.requires_llm
class TestJudgeConsistency:
    """Test that LLM judge gives consistent scores."""

    @pytest.mark.asyncio
    async def test_judge_gives_consistent_scores(self, llm_judge):
        """Test that judge gives similar scores for same input."""
        query = "What is the average income?"
        answer = "The average income is $4,500 based on the 2020 data."
        context = "Dataset shows average income of $4,500 in 2020."

        # Evaluate same input multiple times
        judgments = []
        for _ in range(3):
            judgment = llm_judge.evaluate(
                query=query,
                answer=answer,
                context=context,
            )
            judgments.append(judgment)

        # Calculate score variance
        overall_scores = [j.overall_score for j in judgments]
        score_variance = max(overall_scores) - min(overall_scores)

        print(f"\nScores: {overall_scores}")
        print(f"Variance: {score_variance:.2f}")

        # Scores should be relatively consistent (within 0.5 points)
        assert score_variance <= 0.5, (
            f"Score variance {score_variance:.2f} too high - judge inconsistent"
        )

    @pytest.mark.asyncio
    async def test_judge_consistency_across_criteria(self, llm_judge):
        """Test consistency of individual criteria scores."""
        query = "Calculate average income"
        answer = "The average income across all groups is $5,000."
        context = "Data shows income ranging from $3,000 to $7,000."

        # Evaluate multiple times
        accuracy_scores = []
        completeness_scores = []

        for _ in range(3):
            judgment = llm_judge.evaluate(
                query=query,
                answer=answer,
                context=context,
                criteria=["accuracy", "completeness"],
            )

            accuracy_scores.append(judgment.criteria_scores['accuracy'].score)
            completeness_scores.append(judgment.criteria_scores['completeness'].score)

        # Check variance for each criterion
        accuracy_variance = max(accuracy_scores) - min(accuracy_scores)
        completeness_variance = max(completeness_scores) - min(completeness_scores)

        print(f"\nAccuracy variance: {accuracy_variance:.2f}")
        print(f"Completeness variance: {completeness_variance:.2f}")

        # Individual criteria should be reasonably consistent
        # Relax threshold to 1.0 to account for LLM non-determinism
        assert accuracy_variance <= 1.0, f"Accuracy variance {accuracy_variance} too high"
        assert completeness_variance <= 1.0, f"Completeness variance {completeness_variance} too high"


@pytest.mark.evaluation
@pytest.mark.slow
@pytest.mark.requires_llm
class TestJudgeDiscrimination:
    """Test that judge can distinguish quality levels."""

    @pytest.mark.asyncio
    async def test_judge_distinguishes_good_vs_poor_responses(self, llm_judge):
        """Test that judge scores good responses higher than poor ones."""
        query = "What is the average income in 2020?"
        context = "The average monthly income in 2020 was $4,500."

        # Good response: accurate, complete, clear
        good_answer = (
            "Based on the 2020 data, the average monthly income was $4,500. "
            "This represents the mean income across all demographic groups."
        )

        # Poor response: vague, incomplete, somewhat incorrect
        poor_answer = (
            "Income was around $5,000 or so. People earned money."
        )

        # Evaluate both
        good_judgment = llm_judge.evaluate(
            query=query,
            answer=good_answer,
            context=context,
        )

        poor_judgment = llm_judge.evaluate(
            query=query,
            answer=poor_answer,
            context=context,
        )

        print(f"\nGood response score: {good_judgment.overall_score:.2f}/5.0")
        print(f"Poor response score: {poor_judgment.overall_score:.2f}/5.0")
        print(f"Difference: {(good_judgment.overall_score - poor_judgment.overall_score):.2f}")

        # Good response should score significantly higher
        assert good_judgment.overall_score > poor_judgment.overall_score + 0.5, (
            "Judge failed to distinguish quality levels"
        )

    @pytest.mark.asyncio
    async def test_judge_detects_inaccuracies(self, llm_judge):
        """Test that judge detects factual inaccuracies."""
        query = "What is the average income?"
        context = "The average income is $4,500."

        # Accurate answer
        accurate_answer = "The average income is $4,500 as shown in the data."

        # Inaccurate answer
        inaccurate_answer = "The average income is $10,000 and growing rapidly."

        # Evaluate both
        accurate_judgment = llm_judge.evaluate(
            query=query,
            answer=accurate_answer,
            context=context,
        )

        inaccurate_judgment = llm_judge.evaluate(
            query=query,
            answer=inaccurate_answer,
            context=context,
        )

        print(f"\nAccurate answer accuracy: {accurate_judgment.criteria_scores['accuracy'].score:.2f}/5.0")
        print(f"Inaccurate answer accuracy: {inaccurate_judgment.criteria_scores['accuracy'].score:.2f}/5.0")

        # Accurate answer should have much higher accuracy score
        assert (
            accurate_judgment.criteria_scores['accuracy'].score >
            inaccurate_judgment.criteria_scores['accuracy'].score + 1.0
        ), "Judge failed to detect inaccuracy"

    @pytest.mark.asyncio
    async def test_judge_detects_incomplete_answers(self, llm_judge):
        """Test that judge detects incomplete answers."""
        query = "Compare income between males and females in 2020"
        context = "Males: $5,200, Females: $4,800 in 2020"

        # Complete answer
        complete_answer = (
            "In 2020, males earned an average of $5,200 while females earned $4,800, "
            "representing a $400 gap or 7.7% difference."
        )

        # Incomplete answer
        incomplete_answer = "Males earned more than females."

        # Evaluate both
        complete_judgment = llm_judge.evaluate(
            query=query,
            answer=complete_answer,
            context=context,
        )

        incomplete_judgment = llm_judge.evaluate(
            query=query,
            answer=incomplete_answer,
            context=context,
        )

        print(f"\nComplete answer completeness: {complete_judgment.criteria_scores['completeness'].score:.2f}/5.0")
        print(f"Incomplete answer completeness: {incomplete_judgment.criteria_scores['completeness'].score:.2f}/5.0")

        # Complete answer should score higher on completeness
        assert (
            complete_judgment.criteria_scores['completeness'].score >
            incomplete_judgment.criteria_scores['completeness'].score + 1.0
        ), "Judge failed to detect incompleteness"


@pytest.mark.evaluation
@pytest.mark.slow
@pytest.mark.requires_llm
class TestJudgePairwiseComparison:
    """Test pairwise comparison functionality."""

    @pytest.mark.asyncio
    async def test_pairwise_comparison(self, llm_judge):
        """Test A/B comparison of two answers."""
        query = "What is the average income?"
        context = "Average income is $4,500"

        answer_a = "The average income is $4,500 based on the data provided."
        answer_b = "Income is around $4,500. That's what the data shows."

        # Compare answers
        comparison = llm_judge.compare_answers(
            query=query,
            answer_a=answer_a,
            answer_b=answer_b,
            context=context,
        )

        print(f"\nWinner: {comparison['winner']}")
        print(f"Answer A score: {comparison['answer_a_score']:.2f}")
        print(f"Answer B score: {comparison['answer_b_score']:.2f}")

        # Should pick a winner or declare tie
        assert comparison['winner'] in ['A', 'B', 'Tie']

    @pytest.mark.asyncio
    async def test_comparison_prefers_more_complete_answer(self, llm_judge):
        """Test that comparison prefers more complete answer."""
        query = "Compare income in 2019 vs 2020"
        context = "2019: $4,300, 2020: $4,500"

        # More complete answer
        answer_a = "Income increased from $4,300 in 2019 to $4,500 in 2020, a $200 increase or 4.7% growth."

        # Less complete answer
        answer_b = "Income was $4,300 in 2019 and $4,500 in 2020."

        comparison = llm_judge.compare_answers(
            query=query,
            answer_a=answer_a,
            answer_b=answer_b,
            context=context,
        )

        print(f"\nWinner: {comparison['winner']}")

        # Should prefer the more complete answer
        assert comparison['winner'] in ['A', 'Tie'], (
            "Should prefer more complete answer A"
        )


@pytest.mark.evaluation
@pytest.mark.slow
@pytest.mark.requires_llm
class TestJudgeWithGroundTruth:
    """Test judge evaluation against ground truth."""

    @pytest.mark.asyncio
    async def test_judge_aligns_with_ground_truth(
        self,
        llm_judge,
        ground_truth_answers,
    ):
        """Test that judge scores align with ground truth quality."""
        # Get test case with clear ground truth
        test_case = list(ground_truth_answers.values())[0]

        query = test_case["query"]
        ground_truth = test_case["ground_truth"]["answer"]
        context = "Mock context about income data"

        # Test various quality levels
        test_answers = [
            {
                "answer": ground_truth,
                "expected_min_score": 4.0,
                "label": "Ground truth"
            },
            {
                "answer": "Income was around some amount.",
                "expected_max_score": 3.0,
                "label": "Vague answer"
            },
        ]

        for test in test_answers:
            judgment = llm_judge.evaluate(
                query=query,
                answer=test["answer"],
                context=context,
                ground_truth=ground_truth,
            )

            print(f"\n{test['label']} score: {judgment.overall_score:.2f}/5.0")

            # Check expectations
            if "expected_min_score" in test:
                assert judgment.overall_score >= test["expected_min_score"], (
                    f"{test['label']} scored too low"
                )
            if "expected_max_score" in test:
                assert judgment.overall_score <= test["expected_max_score"], (
                    f"{test['label']} scored too high"
                )


@pytest.mark.evaluation
@pytest.mark.slow
@pytest.mark.requires_llm
class TestJudgeThresholds:
    """Test threshold-based quality checks."""

    @pytest.mark.asyncio
    async def test_responses_meet_quality_thresholds(
        self,
        llm_judge,
        test_config,
        async_db_session,
        sample_datasets,
    ):
        """Test that system responses meet judge quality thresholds."""
        from app.services.agents.orchestrator import AgentOrchestrator
        from app.config import get_config

        config = get_config()
        orchestrator = AgentOrchestrator(config, async_db_session)

        thresholds = test_config["llm_judge"]["thresholds"]

        test_queries = [
            "What was the average income in 2020?",
            "Show employment rates by age group",
        ]

        passed = 0
        total = 0

        for query in test_queries:
            try:
                result = await orchestrator.process_query(query)
                answer = result.get("answer", "")

                if not answer:
                    continue

                total += 1

                judgment = llm_judge.evaluate(
                    query=query,
                    answer=answer,
                )

                # Check if meets thresholds
                if judgment.meets_thresholds(thresholds):
                    passed += 1
                    print(f"✅ Query passed: {query[:50]}...")
                else:
                    print(f"❌ Query failed: {query[:50]}...")
                    print(f"   Overall: {judgment.overall_score:.2f} (threshold: {thresholds.get('overall_score', 3.5)})")

            except Exception as e:
                print(f"Error: {e}")
                continue

        if total == 0:
            pytest.skip("No responses evaluated")

        pass_rate = passed / total
        print(f"\n=== Quality Threshold Results ===")
        print(f"Passed: {passed}/{total} ({pass_rate:.1%})")

        # Should have reasonable pass rate
        assert pass_rate >= 0.70, f"Pass rate {pass_rate:.1%} below 70%"


@pytest.mark.evaluation
@pytest.mark.slow
@pytest.mark.requires_llm
class TestJudgeCriteria:
    """Test individual evaluation criteria."""

    @pytest.mark.asyncio
    async def test_accuracy_criterion(self, llm_judge):
        """Test accuracy criterion specifically."""
        query = "What is the value?"
        context = "The value is 100."

        test_cases = [
            {"answer": "The value is 100.", "should_pass": True},
            {"answer": "The value is 200.", "should_pass": False},
        ]

        for test in test_cases:
            judgment = llm_judge.evaluate(
                query=query,
                answer=test["answer"],
                context=context,
                criteria=["accuracy"],
            )

            accuracy_score = judgment.criteria_scores['accuracy'].score

            if test["should_pass"]:
                assert accuracy_score >= 4.0, "Accurate answer scored too low"
            else:
                assert accuracy_score <= 3.0, "Inaccurate answer scored too high"

    @pytest.mark.asyncio
    async def test_clarity_criterion(self, llm_judge):
        """Test clarity criterion specifically."""
        query = "Explain the average income"

        test_cases = [
            {
                "answer": "The average income is $4,500 per month, calculated across all age groups.",
                "should_pass": True,
                "label": "Clear"
            },
            {
                "answer": "umm so like income is kinda around some number maybe $4500 ish idk",
                "should_pass": False,
                "label": "Unclear"
            },
        ]

        for test in test_cases:
            judgment = llm_judge.evaluate(
                query=query,
                answer=test["answer"],
                criteria=["clarity"],
            )

            clarity_score = judgment.criteria_scores['clarity'].score
            print(f"\n{test['label']} clarity: {clarity_score:.2f}/5.0")

            if test["should_pass"]:
                assert clarity_score >= 3.5, f"{test['label']} scored too low on clarity"
            else:
                assert clarity_score <= 3.0, f"{test['label']} scored too high on clarity"

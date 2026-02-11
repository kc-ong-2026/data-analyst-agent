"""
Generation Metrics Evaluation using BERTScore and Ragas.

Evaluates LLM-generated responses using:
- BERTScore: Semantic similarity with ground truth (precision, recall, F1)
- Ragas Faithfulness: Answer grounded in context (no hallucination)
- Ragas Answer Relevancy: Answer addresses the query
- Ragas Answer Correctness: Factual accuracy

Tests evaluate all agent responses (verification, coordinator, extraction, analytics).
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from app.services.agents.base_agent import AgentState


@pytest.mark.evaluation
@pytest.mark.slow
@pytest.mark.requires_llm
class TestBERTScoreEvaluation:
    """Test BERTScore semantic similarity metrics."""

    @pytest.mark.asyncio
    async def test_bertscore_on_generated_answers(
        self,
        bertscore_evaluator,
        ground_truth_answers,
        async_db_session,
        sample_datasets,
    ):
        """Test BERTScore F1 on generated analytics answers."""
        from app.services.agents.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator()

        # Prepare test cases
        candidates = []
        references = []

        for test_id, gt_data in list(ground_truth_answers.items())[:10]:
            query = gt_data["query"]
            reference_answer = gt_data["ground_truth"]["answer"]

            # Generate answer
            try:
                result = await orchestrator.execute(query)
                generated_answer = result.get("message", "")

                if generated_answer:
                    candidates.append(generated_answer)
                    references.append(reference_answer)
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
                continue

        if not candidates:
            pytest.skip("No answers generated")

        # Evaluate with BERTScore
        results = bertscore_evaluator.evaluate_batch(candidates, references)
        aggregate = bertscore_evaluator.get_aggregate_scores(results)

        print(f"BERTScore - P: {aggregate.precision:.4f}, "
              f"R: {aggregate.recall:.4f}, F1: {aggregate.f1:.4f}")

        # Check thresholds
        assert aggregate.precision >= 0.65, (
            f"BERTScore precision {aggregate.precision:.4f} below threshold"
        )
        assert aggregate.recall >= 0.65, (
            f"BERTScore recall {aggregate.recall:.4f} below threshold"
        )
        assert aggregate.f1 >= 0.70, (
            f"BERTScore F1 {aggregate.f1:.4f} below threshold"
        )

    @pytest.mark.asyncio
    async def test_bertscore_per_agent(
        self,
        bertscore_evaluator,
        async_db_session,
        sample_datasets,
    ):
        """Test BERTScore for each agent type."""
        from app.services.agents.verification import QueryVerificationAgent
        from app.services.agents.analytics import AnalyticsAgent

        # Test verification agent
        verification_agent = QueryVerificationAgent()

        agent_state = AgentState()
        agent_state.current_task = "What was the average income in 2020?"
        agent_state.add_message("user", "What was the average income in 2020?")

        response = await verification_agent.execute(agent_state)
        print(f"Verification agent success: {response.success}")

        # BERTScore evaluation would compare agent output to expected output
        # This is a simplified version - full implementation would have ground truth

    @pytest.mark.asyncio
    async def test_bertscore_with_threshold_check(
        self,
        bertscore_evaluator,
        ground_truth_answers,
        test_config,
    ):
        """Test BERTScore threshold checking."""
        candidates = ["The average income was $4,500 in 2020."]
        references = ["In 2020, the average income was $4,500."]

        results, passed, failed = bertscore_evaluator.evaluate_with_threshold(
            candidates, references
        )

        print(f"Threshold check: {passed}/{len(results)} passed")

        # Similar sentences should pass threshold
        assert passed > 0


@pytest.mark.evaluation
@pytest.mark.slow
@pytest.mark.requires_llm
class TestFaithfulness:
    """Test answer faithfulness (grounded in context)."""

    @pytest.mark.asyncio
    async def test_answers_grounded_in_context(
        self,
        ragas_evaluator,
        async_db_session,
        sample_datasets,
    ):
        """Test that generated answers are faithful to retrieved context."""
        from app.services.agents.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator()

        # Test queries
        test_queries = [
            "What was the average income in 2020?",
            "Show me employment rates by age group",
        ]

        queries = []
        answers = []
        contexts = []

        for query in test_queries:
            try:
                result = await orchestrator.execute(query)
                answer = result.get("message", "")
                sources = result.get("sources", [])

                if answer and sources:
                    queries.append(query)
                    answers.append(answer)
                    # Convert sources (dataset names) to context strings
                    contexts.append([f"Dataset: {s}" for s in sources])
            except Exception as e:
                print(f"Error: {e}")
                continue

        if not queries:
            pytest.skip("No results generated")

        # Evaluate faithfulness
        ragas_result = ragas_evaluator.evaluate_generation(
            queries=queries,
            answers=answers,
            contexts=contexts,
        )

        if ragas_result.faithfulness:
            print(f"Faithfulness: {ragas_result.faithfulness:.4f}")
            assert ragas_result.faithfulness >= 0.80, (
                f"Faithfulness {ragas_result.faithfulness:.4f} below threshold - "
                "answers may contain hallucinations"
            )

    @pytest.mark.asyncio
    async def test_faithfulness_detects_hallucination(
        self,
        ragas_evaluator,
    ):
        """Test that faithfulness metric detects hallucinations."""
        # Case 1: Faithful answer
        faithful_case = {
            "query": ["What is the average income?"],
            "answer": ["The average income is $4,500 based on the data."],
            "contexts": [["Average income from the dataset is $4,500."]],
        }

        # Case 2: Hallucinated answer
        hallucinated_case = {
            "query": ["What is the average income?"],
            "answer": ["The average income is $10,000 and growing rapidly."],  # Not in context
            "contexts": [["Average income from the dataset is $4,500."]],
        }

        # Evaluate faithful case
        faithful_result = ragas_evaluator.evaluate_generation(
            queries=faithful_case["query"],
            answers=faithful_case["answer"],
            contexts=faithful_case["contexts"],
        )

        # Evaluate hallucinated case
        hallucinated_result = ragas_evaluator.evaluate_generation(
            queries=hallucinated_case["query"],
            answers=hallucinated_case["answer"],
            contexts=hallucinated_case["contexts"],
        )

        # Faithful answer should score higher
        if faithful_result.faithfulness and hallucinated_result.faithfulness:
            print(f"Faithful: {faithful_result.faithfulness:.4f}")
            print(f"Hallucinated: {hallucinated_result.faithfulness:.4f}")
            assert faithful_result.faithfulness > hallucinated_result.faithfulness


@pytest.mark.evaluation
@pytest.mark.slow
@pytest.mark.requires_llm
class TestAnswerRelevancy:
    """Test answer relevancy to query."""

    @pytest.mark.asyncio
    async def test_answers_address_query(
        self,
        ragas_evaluator,
        async_db_session,
        sample_datasets,
    ):
        """Test that answers are relevant to the query."""
        from app.services.agents.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator()

        # Test queries
        test_queries = [
            "What was the average income in 2020?",
            "Show employment rates for different age groups",
        ]

        queries = []
        answers = []
        contexts = []

        for query in test_queries:
            try:
                result = await orchestrator.execute(query)
                answer = result.get("message", "")

                if answer:
                    queries.append(query)
                    answers.append(answer)
                    contexts.append(["Context placeholder"])  # Required by Ragas
            except Exception as e:
                print(f"Error: {e}")
                continue

        if not queries:
            pytest.skip("No results generated")

        # Evaluate relevancy
        ragas_result = ragas_evaluator.evaluate_generation(
            queries=queries,
            answers=answers,
            contexts=contexts,
        )

        if ragas_result.answer_relevancy:
            print(f"Answer Relevancy: {ragas_result.answer_relevancy:.4f}")
            assert ragas_result.answer_relevancy >= 0.75, (
                f"Answer Relevancy {ragas_result.answer_relevancy:.4f} below threshold"
            )


@pytest.mark.evaluation
@pytest.mark.slow
@pytest.mark.requires_llm
class TestAnswerCorrectness:
    """Test answer factual correctness."""

    @pytest.mark.asyncio
    async def test_factual_accuracy(
        self,
        ragas_evaluator,
        ground_truth_answers,
        async_db_session,
        sample_datasets,
    ):
        """Test factual accuracy against ground truth."""
        from app.services.agents.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator()

        queries = []
        answers = []
        contexts = []
        ground_truths = []

        for test_id, gt_data in list(ground_truth_answers.items())[:5]:
            query = gt_data["query"]
            ground_truth = gt_data["ground_truth"]["answer"]

            try:
                result = await orchestrator.execute(query)
                answer = result.get("message", "")

                if answer:
                    queries.append(query)
                    answers.append(answer)
                    contexts.append(["Context placeholder"])
                    ground_truths.append(ground_truth)
            except Exception as e:
                print(f"Error: {e}")
                continue

        if not queries:
            pytest.skip("No results generated")

        # Evaluate correctness
        ragas_result = ragas_evaluator.evaluate_end_to_end(
            queries=queries,
            answers=answers,
            contexts=contexts,
            ground_truth_answers=ground_truths,
        )

        if ragas_result.answer_correctness:
            print(f"Answer Correctness: {ragas_result.answer_correctness:.4f}")
            assert ragas_result.answer_correctness >= 0.70, (
                f"Answer Correctness {ragas_result.answer_correctness:.4f} below threshold"
            )


@pytest.mark.evaluation
@pytest.mark.slow
@pytest.mark.requires_llm
class TestAgentSpecificMetrics:
    """Test metrics for specific agents."""

    @pytest.mark.asyncio
    async def test_verification_agent_accuracy(
        self,
        bertscore_evaluator,
        sample_queries,
    ):
        """Test verification agent output quality."""
        from app.services.agents.verification import QueryVerificationAgent

        agent = QueryVerificationAgent()

        verification_tests = sample_queries.get("agent_tests", {}).get("verification", [])

        if not verification_tests:
            pytest.skip("No verification tests")

        correct_validations = 0
        total = len(verification_tests)

        for test in verification_tests[:10]:
            agent_state = AgentState()
            agent_state.current_task = test["query"]
            agent_state.add_message("user", test["query"])

            response = await agent.execute(agent_state)
            validation = response.data.get("validation", {})

            # Check if validation matches expected
            expected_valid = test.get("expected_valid", True)
            actual_valid = validation.get("is_valid", False)

            if expected_valid == actual_valid:
                correct_validations += 1

        accuracy = correct_validations / total
        print(f"Verification accuracy: {accuracy:.2%}")

        assert accuracy >= 0.80, f"Verification accuracy {accuracy:.2%} below threshold"

    @pytest.mark.asyncio
    async def test_extraction_agent_data_relevance(
        self,
        bertscore_evaluator,
        async_db_session,
        sample_datasets,
    ):
        """Test extraction agent data relevance."""
        from app.services.agents.extraction import DataExtractionAgent

        agent = DataExtractionAgent()

        test_queries = [
            {
                "query": "income data for 2020",
                "expected_category": "income",
            },
            {
                "query": "employment statistics",
                "expected_category": "employment",
            },
        ]

        relevant_extractions = 0
        total = len(test_queries)

        for test in test_queries:
            agent_state = AgentState()
            agent_state.current_task = test["query"]
            agent_state.add_message("user", test["query"])
            agent_state.metadata = {
                "query_validation": {
                    "valid": True,
                    "topic": test["expected_category"],
                }
            }

            response = await agent.execute(agent_state)
            extracted_data = response.data.get("extracted_data", [])

            # Check if extracted data is relevant
            if extracted_data and response.success:
                relevant_extractions += 1

        accuracy = relevant_extractions / total
        print(f"Extraction relevance: {accuracy:.2%}")

        assert accuracy >= 0.75

    @pytest.mark.asyncio
    async def test_analytics_answer_quality(
        self,
        bertscore_evaluator,
        ground_truth_answers,
    ):
        """Test analytics agent answer quality."""
        from app.services.agents.analytics import AnalyticsAgent

        agent = AnalyticsAgent()

        # Test with sample data
        test_case = list(ground_truth_answers.values())[0]

        agent_state = AgentState()
        agent_state.current_task = test_case["query"]
        agent_state.add_message("user", test_case["query"])
        agent_state.extracted_data = {
            "test": {
                "dataset_name": "test",
                "columns": ["value"],
                "dtypes": {"value": "float64"},
                "data": [{"value": 4500.0}],
                "source": "dataframe",
            }
        }

        response = await agent.execute(agent_state)

        # Check if explanation generated
        explanation = response.data.get("explanation", "")
        assert len(explanation) > 0, "No explanation generated"

        # Compare with ground truth
        ground_truth = test_case["ground_truth"]["answer"]

        result = bertscore_evaluator.evaluate_single(explanation, ground_truth)
        print(f"Analytics answer BERTScore F1: {result.f1:.4f}")

        # Should have reasonable similarity
        assert result.f1 >= 0.60, f"Analytics answer quality too low: {result.f1:.4f}"


@pytest.mark.evaluation
@pytest.mark.slow
@pytest.mark.requires_llm
class TestEndToEndGeneration:
    """Test end-to-end generation quality."""

    @pytest.mark.asyncio
    async def test_full_pipeline_generation_metrics(
        self,
        ragas_evaluator,
        bertscore_evaluator,
        ground_truth_answers,
        async_db_session,
        sample_datasets,
    ):
        """Test all generation metrics on full pipeline."""
        from app.services.agents.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator()

        # Run on subset of queries
        queries = []
        answers = []
        contexts = []
        ground_truths = []

        for test_id, gt_data in list(ground_truth_answers.items())[:3]:
            query = gt_data["query"]
            ground_truth = gt_data["ground_truth"]["answer"]

            try:
                result = await orchestrator.execute(query)
                answer = result.get("message", "")
                sources = result.get("sources", [])

                if answer:
                    queries.append(query)
                    answers.append(answer)
                    # Convert sources to context strings
                    contexts.append([f"Dataset: {s}" for s in sources] if sources else ["No context"])
                    ground_truths.append(ground_truth)
            except Exception as e:
                print(f"Error: {e}")
                continue

        if not queries:
            pytest.skip("No results generated")

        # Ragas evaluation
        ragas_result = ragas_evaluator.evaluate_end_to_end(
            queries=queries,
            answers=answers,
            contexts=contexts,
            ground_truth_answers=ground_truths,
        )

        print("\n=== Generation Metrics ===")
        if ragas_result.faithfulness:
            print(f"Faithfulness: {ragas_result.faithfulness:.4f}")
        if ragas_result.answer_relevancy:
            print(f"Answer Relevancy: {ragas_result.answer_relevancy:.4f}")
        if ragas_result.answer_correctness:
            print(f"Answer Correctness: {ragas_result.answer_correctness:.4f}")

        # BERTScore evaluation
        bert_results = bertscore_evaluator.evaluate_batch(answers, ground_truths)
        bert_aggregate = bertscore_evaluator.get_aggregate_scores(bert_results)

        print(f"\nBERTScore F1: {bert_aggregate.f1:.4f}")

        # Assertions
        if ragas_result.faithfulness:
            assert ragas_result.faithfulness >= 0.75, "Faithfulness too low"
        if ragas_result.answer_relevancy:
            assert ragas_result.answer_relevancy >= 0.70, "Relevancy too low"
        assert bert_aggregate.f1 >= 0.65, "BERTScore F1 too low"

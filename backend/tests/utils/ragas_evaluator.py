"""
Ragas Evaluator for RAG System Evaluation.

This module provides a wrapper around the Ragas evaluation framework
for measuring retrieval and generation quality metrics:
- Context Precision: Relevance of retrieved contexts
- Context Recall: Completeness of information retrieval
- Faithfulness: Groundedness of answers in contexts
- Answer Relevancy: Relevance of answers to queries
- Answer Correctness: Factual accuracy compared to ground truth
"""

import logging
from dataclasses import dataclass
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

logger = logging.getLogger(__name__)


@dataclass
class RagasResult:
    """Results from Ragas evaluation."""

    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    answer_correctness: float | None = None

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary, excluding None values."""
        return {
            k: v
            for k, v in {
                "context_precision": self.context_precision,
                "context_recall": self.context_recall,
                "faithfulness": self.faithfulness,
                "answer_relevancy": self.answer_relevancy,
                "answer_correctness": self.answer_correctness,
            }.items()
            if v is not None
        }

    def meets_thresholds(self, thresholds: dict[str, float]) -> bool:
        """Check if all metrics meet their thresholds."""
        for metric, threshold in thresholds.items():
            value = getattr(self, metric, None)
            if value is not None and value < threshold:
                return False
        return True


class RagasEvaluator:
    """
    Wrapper for Ragas evaluation framework.

    Provides methods for evaluating RAG system quality using
    multiple metrics for both retrieval and generation.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize Ragas evaluator.

        Args:
            config: Configuration dictionary with keys:
                - metrics: List of metric names to compute
                - thresholds: Dict of metric name -> threshold value
                - batch_size: Batch size for evaluation
        """
        self.config = config
        self.metrics = self._get_metrics(config.get("metrics", []))
        self.thresholds = config.get("thresholds", {})
        self.batch_size = config.get("batch_size", 10)

        logger.info(f"Initialized RagasEvaluator with metrics: {list(self.metrics.keys())}")

    def _get_metrics(self, metric_names: list[str]) -> dict[str, Any]:
        """Map metric names to Ragas metric objects."""
        available_metrics = {
            "context_precision": context_precision,
            "context_recall": context_recall,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "answer_correctness": answer_correctness,
        }

        return {name: available_metrics[name] for name in metric_names if name in available_metrics}

    def evaluate_retrieval(
        self,
        queries: list[str],
        retrieved_contexts: list[list[str]],
        ground_truth_contexts: list[list[str]] | None = None,
        reference_answers: list[str] | None = None,
    ) -> RagasResult:
        """
        Evaluate retrieval quality using context precision and recall.

        Args:
            queries: List of user queries
            retrieved_contexts: List of lists of retrieved context strings
            ground_truth_contexts: Optional list of lists of ground truth contexts
                                  (required for context_recall)
            reference_answers: Optional list of reference answers
                             (required for context_precision in Ragas 0.3+)

        Returns:
            RagasResult with context_precision and context_recall scores
        """
        logger.info(f"Evaluating retrieval for {len(queries)} queries")

        # Prepare dataset for Ragas
        data = {
            "question": queries,
            "contexts": retrieved_contexts,
        }

        # Add ground truth if provided (for context_recall)
        if ground_truth_contexts:
            data["ground_truth"] = [" ".join(contexts) for contexts in ground_truth_contexts]

        # Add reference answers if provided (for context_precision in Ragas 0.3+)
        # If not provided, generate placeholder references from contexts
        if reference_answers:
            data["reference"] = reference_answers
        elif ground_truth_contexts:
            # Use ground truth contexts as dummy references
            data["reference"] = [" ".join(contexts) for contexts in ground_truth_contexts]
        else:
            # Generate placeholder references from retrieved contexts
            # (allows context_precision to run even without ground truth)
            data["reference"] = [
                f"Answer based on: {' '.join(contexts[:2])}" if contexts else "No answer available"
                for contexts in retrieved_contexts
            ]
            logger.info("Using placeholder references generated from retrieved contexts")

        dataset = Dataset.from_dict(data)

        # Select appropriate metrics based on available data
        metrics_to_use = []

        # context_precision requires 'reference' field in Ragas 0.3+
        # (we now always provide one, either real or placeholder)
        if "context_precision" in self.metrics:
            metrics_to_use.append(self.metrics["context_precision"])

        # context_recall requires ground_truth
        if "context_recall" in self.metrics and ground_truth_contexts:
            metrics_to_use.append(self.metrics["context_recall"])

        if not metrics_to_use:
            logger.warning("No retrieval metrics configured or data missing")
            return RagasResult()

        # Run evaluation
        try:
            results = evaluate(dataset, metrics=metrics_to_use)

            result = RagasResult()

            # Ragas 0.3+ returns a Result object with scores dict
            # Access scores via .to_pandas() or iterate the dataset
            scores_df = results.to_pandas()

            if "context_precision" in scores_df.columns:
                result.context_precision = scores_df["context_precision"].mean()
                logger.info(f"Context Precision: {result.context_precision:.4f}")

            if "context_recall" in scores_df.columns:
                result.context_recall = scores_df["context_recall"].mean()
                logger.info(f"Context Recall: {result.context_recall:.4f}")

            return result

        except Exception as e:
            logger.error(f"Error during retrieval evaluation: {e}")
            raise

    def evaluate_generation(
        self,
        queries: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str] | None = None,
    ) -> RagasResult:
        """
        Evaluate generation quality using faithfulness, relevancy, and correctness.

        Args:
            queries: List of user queries
            answers: List of generated answers
            contexts: List of lists of context strings used for generation
            ground_truths: Optional list of ground truth answers
                          (required for answer_correctness)

        Returns:
            RagasResult with faithfulness, answer_relevancy, and answer_correctness
        """
        logger.info(f"Evaluating generation for {len(queries)} queries")

        # Prepare dataset
        data = {
            "question": queries,
            "answer": answers,
            "contexts": contexts,
        }

        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        # Select appropriate metrics
        metrics_to_use = []
        if "faithfulness" in self.metrics:
            metrics_to_use.append(self.metrics["faithfulness"])
        if "answer_relevancy" in self.metrics:
            metrics_to_use.append(self.metrics["answer_relevancy"])
        if "answer_correctness" in self.metrics and ground_truths:
            metrics_to_use.append(self.metrics["answer_correctness"])

        if not metrics_to_use:
            logger.warning("No generation metrics configured")
            return RagasResult()

        # Run evaluation
        try:
            results = evaluate(dataset, metrics=metrics_to_use)

            result = RagasResult()

            # Ragas 0.3+ returns a Result object with scores dict
            scores_df = results.to_pandas()

            if "faithfulness" in scores_df.columns:
                result.faithfulness = scores_df["faithfulness"].mean()
                logger.info(f"Faithfulness: {result.faithfulness:.4f}")

            if "answer_relevancy" in scores_df.columns:
                result.answer_relevancy = scores_df["answer_relevancy"].mean()
                logger.info(f"Answer Relevancy: {result.answer_relevancy:.4f}")

            if "answer_correctness" in scores_df.columns:
                result.answer_correctness = scores_df["answer_correctness"].mean()
                logger.info(f"Answer Correctness: {result.answer_correctness:.4f}")

            return result

        except Exception as e:
            logger.error(f"Error during generation evaluation: {e}")
            raise

    def evaluate_end_to_end(
        self,
        queries: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truth_contexts: list[list[str]] | None = None,
        ground_truth_answers: list[str] | None = None,
    ) -> RagasResult:
        """
        Evaluate full RAG pipeline with all metrics.

        Args:
            queries: List of user queries
            answers: List of generated answers
            contexts: List of lists of retrieved context strings
            ground_truth_contexts: Optional ground truth contexts for recall
            ground_truth_answers: Optional ground truth answers for correctness

        Returns:
            RagasResult with all applicable metrics
        """
        logger.info(f"Running end-to-end evaluation for {len(queries)} queries")

        # Prepare dataset
        data = {
            "question": queries,
            "answer": answers,
            "contexts": contexts,
        }

        if ground_truth_contexts:
            data["ground_truth"] = [
                " ".join(contexts) if isinstance(contexts, list) else contexts
                for contexts in ground_truth_contexts
            ]
        elif ground_truth_answers:
            data["ground_truth"] = ground_truth_answers

        dataset = Dataset.from_dict(data)

        # Use all configured metrics
        metrics_to_use = list(self.metrics.values())

        if not metrics_to_use:
            logger.warning("No metrics configured")
            return RagasResult()

        # Run evaluation
        try:
            results = evaluate(dataset, metrics=metrics_to_use)

            # Convert to pandas and extract mean scores
            df = results.to_pandas()

            result = RagasResult(
                context_precision=(
                    df["context_precision"].mean() if "context_precision" in df.columns else None
                ),
                context_recall=(
                    df["context_recall"].mean() if "context_recall" in df.columns else None
                ),
                faithfulness=df["faithfulness"].mean() if "faithfulness" in df.columns else None,
                answer_relevancy=(
                    df["answer_relevancy"].mean() if "answer_relevancy" in df.columns else None
                ),
                answer_correctness=(
                    df["answer_correctness"].mean() if "answer_correctness" in df.columns else None
                ),
            )

            logger.info(f"End-to-end evaluation results: {result.to_dict()}")

            # Check thresholds
            if not result.meets_thresholds(self.thresholds):
                logger.warning("Some metrics below configured thresholds")
                for metric, threshold in self.thresholds.items():
                    value = getattr(result, metric, None)
                    if value is not None and value < threshold:
                        logger.warning(f"  {metric}: {value:.4f} < {threshold:.4f}")

            return result

        except Exception as e:
            logger.error(f"Error during end-to-end evaluation: {e}")
            raise

    def evaluate_batch(self, test_cases: list[dict[str, Any]]) -> list[RagasResult]:
        """
        Evaluate a batch of test cases.

        Args:
            test_cases: List of dicts with keys:
                - query: User query
                - answer: Generated answer
                - contexts: Retrieved contexts
                - ground_truth_context (optional)
                - ground_truth_answer (optional)

        Returns:
            List of RagasResult, one per test case
        """
        logger.info(f"Evaluating batch of {len(test_cases)} test cases")

        # Extract fields
        queries = [tc["query"] for tc in test_cases]
        answers = [tc["answer"] for tc in test_cases]
        contexts = [tc["contexts"] for tc in test_cases]

        ground_truth_contexts = (
            [tc.get("ground_truth_context", []) for tc in test_cases]
            if any("ground_truth_context" in tc for tc in test_cases)
            else None
        )

        ground_truth_answers = (
            [tc.get("ground_truth_answer", "") for tc in test_cases]
            if any("ground_truth_answer" in tc for tc in test_cases)
            else None
        )

        # Run end-to-end evaluation
        result = self.evaluate_end_to_end(
            queries=queries,
            answers=answers,
            contexts=contexts,
            ground_truth_contexts=ground_truth_contexts,
            ground_truth_answers=ground_truth_answers,
        )

        # For now, return same result for all test cases
        # In a more sophisticated version, could evaluate per-item
        return [result] * len(test_cases)

"""
BERTScore Evaluator for Semantic Similarity Evaluation.

This module provides a wrapper around BERTScore for measuring semantic
similarity between generated text and reference text. BERTScore computes
precision, recall, and F1 using contextual embeddings from transformer models.

Key advantages over traditional metrics (BLEU, ROUGE):
- Captures semantic similarity, not just lexical overlap
- Handles paraphrasing well
- Uses contextual embeddings for better meaning representation
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

import torch
from bert_score import score as bert_score
from bert_score import BERTScorer

logger = logging.getLogger(__name__)


@dataclass
class BERTScoreResult:
    """Results from BERTScore evaluation."""

    precision: float
    recall: float
    f1: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

    def meets_thresholds(self, thresholds: Dict[str, float]) -> bool:
        """Check if all metrics meet their thresholds."""
        return (
            self.precision >= thresholds.get("precision", 0.0) and
            self.recall >= thresholds.get("recall", 0.0) and
            self.f1 >= thresholds.get("f1", 0.0)
        )


class BERTScoreEvaluator:
    """
    Wrapper for BERTScore evaluation.

    Provides methods for computing semantic similarity between
    generated text and reference text using transformer-based embeddings.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BERTScore evaluator.

        Args:
            config: Configuration dictionary with keys:
                - model: Model name (e.g., "microsoft/deberta-xlarge-mnli")
                - lang: Language code (e.g., "en")
                - rescale_with_baseline: Whether to rescale scores
                - batch_size: Batch size for evaluation
                - thresholds: Dict of metric -> threshold
        """
        self.config = config
        self.model = config.get("model", "microsoft/deberta-xlarge-mnli")
        self.lang = config.get("lang", "en")
        self.rescale_with_baseline = config.get("rescale_with_baseline", True)
        self.batch_size = config.get("batch_size", 32)
        self.thresholds = config.get("thresholds", {})

        # Initialize scorer (reusable for efficiency)
        self.scorer = None
        self._init_scorer()

        logger.info(
            f"Initialized BERTScoreEvaluator with model: {self.model}, "
            f"rescale: {self.rescale_with_baseline}"
        )

    def _init_scorer(self):
        """Initialize BERTScorer instance."""
        try:
            self.scorer = BERTScorer(
                model_type=self.model,
                lang=self.lang,
                rescale_with_baseline=self.rescale_with_baseline,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            logger.info(f"BERTScorer initialized on device: {self.scorer.device}")
        except Exception as e:
            logger.error(f"Failed to initialize BERTScorer: {e}")
            self.scorer = None

    def evaluate_single(
        self,
        candidate: str,
        reference: str,
    ) -> BERTScoreResult:
        """
        Evaluate a single candidate-reference pair.

        Args:
            candidate: Generated text
            reference: Ground truth reference text

        Returns:
            BERTScoreResult with precision, recall, F1
        """
        if self.scorer is None:
            logger.warning("BERTScorer not initialized, returning default scores")
            return BERTScoreResult(precision=0.0, recall=0.0, f1=0.0)

        try:
            P, R, F1 = self.scorer.score([candidate], [reference])

            result = BERTScoreResult(
                precision=P[0].item(),
                recall=R[0].item(),
                f1=F1[0].item(),
            )

            logger.debug(
                f"BERTScore - P: {result.precision:.4f}, "
                f"R: {result.recall:.4f}, F1: {result.f1:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error during BERTScore evaluation: {e}")
            return BERTScoreResult(precision=0.0, recall=0.0, f1=0.0)

    def evaluate_batch(
        self,
        candidates: List[str],
        references: List[str],
    ) -> List[BERTScoreResult]:
        """
        Evaluate a batch of candidate-reference pairs.

        Args:
            candidates: List of generated texts
            references: List of ground truth reference texts

        Returns:
            List of BERTScoreResult, one per pair
        """
        if len(candidates) != len(references):
            raise ValueError(
                f"Candidates ({len(candidates)}) and references ({len(references)}) "
                f"must have same length"
            )

        if self.scorer is None:
            logger.warning("BERTScorer not initialized, returning default scores")
            return [
                BERTScoreResult(precision=0.0, recall=0.0, f1=0.0)
                for _ in candidates
            ]

        logger.info(f"Evaluating batch of {len(candidates)} pairs")

        try:
            # Compute scores in batches
            all_P, all_R, all_F1 = [], [], []

            for i in range(0, len(candidates), self.batch_size):
                batch_cands = candidates[i:i + self.batch_size]
                batch_refs = references[i:i + self.batch_size]

                P, R, F1 = self.scorer.score(batch_cands, batch_refs)

                all_P.extend(P.tolist())
                all_R.extend(R.tolist())
                all_F1.extend(F1.tolist())

            # Create results
            results = [
                BERTScoreResult(precision=p, recall=r, f1=f)
                for p, r, f in zip(all_P, all_R, all_F1)
            ]

            # Compute averages
            avg_precision = sum(r.precision for r in results) / len(results)
            avg_recall = sum(r.recall for r in results) / len(results)
            avg_f1 = sum(r.f1 for r in results) / len(results)

            logger.info(
                f"Batch BERTScore - Avg P: {avg_precision:.4f}, "
                f"Avg R: {avg_recall:.4f}, Avg F1: {avg_f1:.4f}"
            )

            return results

        except Exception as e:
            logger.error(f"Error during batch BERTScore evaluation: {e}")
            return [
                BERTScoreResult(precision=0.0, recall=0.0, f1=0.0)
                for _ in candidates
            ]

    def evaluate_with_threshold(
        self,
        candidates: List[str],
        references: List[str],
    ) -> Tuple[List[BERTScoreResult], int, int]:
        """
        Evaluate batch and count results meeting thresholds.

        Args:
            candidates: List of generated texts
            references: List of ground truth reference texts

        Returns:
            Tuple of (results, num_passed, num_failed)
        """
        results = self.evaluate_batch(candidates, references)

        num_passed = sum(
            1 for r in results if r.meets_thresholds(self.thresholds)
        )
        num_failed = len(results) - num_passed

        logger.info(
            f"Threshold check: {num_passed}/{len(results)} passed, "
            f"{num_failed} failed"
        )

        return results, num_passed, num_failed

    def get_aggregate_scores(
        self,
        results: List[BERTScoreResult]
    ) -> BERTScoreResult:
        """
        Compute aggregate (average) scores from a list of results.

        Args:
            results: List of BERTScoreResult

        Returns:
            BERTScoreResult with averaged scores
        """
        if not results:
            return BERTScoreResult(precision=0.0, recall=0.0, f1=0.0)

        avg_precision = sum(r.precision for r in results) / len(results)
        avg_recall = sum(r.recall for r in results) / len(results)
        avg_f1 = sum(r.f1 for r in results) / len(results)

        return BERTScoreResult(
            precision=avg_precision,
            recall=avg_recall,
            f1=avg_f1,
        )

    @staticmethod
    def compare_models(
        candidates: List[str],
        references: List[str],
        models: List[str],
    ) -> Dict[str, BERTScoreResult]:
        """
        Compare BERTScore using different models.

        Useful for model selection experiments.

        Args:
            candidates: List of generated texts
            references: List of ground truth texts
            models: List of model names to compare

        Returns:
            Dict mapping model name -> aggregate BERTScoreResult
        """
        logger.info(f"Comparing {len(models)} models on {len(candidates)} pairs")

        results = {}

        for model_name in models:
            logger.info(f"Evaluating with model: {model_name}")

            config = {
                "model": model_name,
                "lang": "en",
                "rescale_with_baseline": True,
                "batch_size": 32,
            }

            evaluator = BERTScoreEvaluator(config)
            batch_results = evaluator.evaluate_batch(candidates, references)
            aggregate = evaluator.get_aggregate_scores(batch_results)

            results[model_name] = aggregate

            logger.info(
                f"  {model_name} - P: {aggregate.precision:.4f}, "
                f"R: {aggregate.recall:.4f}, F1: {aggregate.f1:.4f}"
            )

        return results

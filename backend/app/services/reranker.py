"""Cross-encoder reranker for improving retrieval relevance."""

import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Rerank search results using cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize cross-encoder model.

        Args:
            model_name: HuggingFace model name. Default is lightweight (80MB) and fast.
        """
        self.model = CrossEncoder(model_name)
        logger.info(f"Loaded cross-encoder model: {model_name}")

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Rerank documents by semantic relevance to query.

        Args:
            query: User query
            documents: List of document texts to rank
            top_k: Number of top results to return

        Returns:
            List of (doc_index, score) tuples sorted by score descending
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score with cross-encoder
        scores = self.model.predict(pairs)

        # Sort by score descending
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [(idx, float(scores[idx])) for idx in ranked_indices]

    def rerank_with_threshold(
        self,
        query: str,
        documents: list[str],
        threshold: float = 0.5,
        min_results: int = 1,
        max_results: int = 5,
    ) -> list[tuple[int, float]]:
        """Rerank and filter by confidence threshold.

        Args:
            query: User query
            documents: List of document texts
            threshold: Minimum score to include (0-1)
            min_results: Guarantee at least this many results (even if below threshold)
            max_results: Maximum results to return

        Returns:
            List of (doc_index, score) tuples filtered by threshold
        """
        all_ranked = self.rerank(query, documents, top_k=len(documents))

        # Filter by threshold
        filtered = [(idx, score) for idx, score in all_ranked if score >= threshold]

        # Guarantee min_results
        if len(filtered) < min_results:
            filtered = all_ranked[:min_results]

        # Enforce max_results
        return filtered[:max_results]


# Global instance (lazy loaded)
_reranker_instance = None


def get_reranker() -> CrossEncoderReranker:
    """Get global reranker instance (lazy initialization)."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance

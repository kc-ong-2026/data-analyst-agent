"""
Evaluation Export Utility.

Exports evaluation scores and results in various formats for fine-tuning
and analysis purposes. Supports:
- JSONL format (generic)
- OpenAI fine-tuning format
- Anthropic fine-tuning format
- CSV format (for spreadsheet analysis)
- Parquet format (for large-scale analysis)

Usage:
    from tests.utils.evaluation_export import EvaluationExporter

    exporter = EvaluationExporter()
    exporter.export_to_openai_format(
        results=evaluation_results,
        output_path=Path("./fine_tuning_data/training.jsonl"),
        min_score=0.80
    )
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Single evaluation result.

    Contains query, answer, contexts, scores, and metadata.
    """
    query: str
    answer: str
    agent: str
    contexts: Optional[List[str]] = None
    ground_truth: Optional[str] = None

    # Scores
    ragas_context_precision: Optional[float] = None
    ragas_context_recall: Optional[float] = None
    ragas_faithfulness: Optional[float] = None
    ragas_answer_relevancy: Optional[float] = None
    ragas_answer_correctness: Optional[float] = None

    bertscore_precision: Optional[float] = None
    bertscore_recall: Optional[float] = None
    bertscore_f1: Optional[float] = None

    llm_judge_overall: Optional[float] = None
    llm_judge_accuracy: Optional[float] = None
    llm_judge_completeness: Optional[float] = None
    llm_judge_clarity: Optional[float] = None
    llm_judge_conciseness: Optional[float] = None
    llm_judge_code_quality: Optional[float] = None
    llm_judge_visualization: Optional[float] = None

    # Metadata
    timestamp: Optional[str] = None
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    model_name: Optional[str] = None

    def get_overall_score(self) -> float:
        """Calculate aggregate score across all metrics."""
        scores = []

        # Ragas metrics (weight: 0.4)
        ragas_scores = [
            self.ragas_faithfulness,
            self.ragas_answer_relevancy,
            self.ragas_answer_correctness,
        ]
        ragas_avg = sum(s for s in ragas_scores if s is not None) / len(
            [s for s in ragas_scores if s is not None]
        ) if any(s is not None for s in ragas_scores) else None
        if ragas_avg:
            scores.append(ragas_avg * 0.4)

        # BERTScore (weight: 0.2)
        if self.bertscore_f1:
            scores.append(self.bertscore_f1 * 0.2)

        # LLM Judge (weight: 0.4, scale from 5.0 to 1.0)
        if self.llm_judge_overall:
            scores.append((self.llm_judge_overall / 5.0) * 0.4)

        return sum(scores) if scores else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class EvaluationExporter:
    """
    Utility for exporting evaluation results in various formats.

    Supports filtering by score, agent, and other criteria.
    """

    def __init__(self):
        """Initialize exporter."""
        pass

    def export_to_jsonl(
        self,
        results: List[Dict[str, Any]],
        output_path: Path,
        min_score: float = 0.0,
        include_metadata: bool = True,
    ) -> int:
        """
        Export results to generic JSONL format.

        Args:
            results: List of evaluation result dictionaries
            output_path: Output file path
            min_score: Minimum overall score to include
            include_metadata: Include metadata fields

        Returns:
            Number of results exported
        """
        logger.info(f"Exporting to JSONL: {output_path}")

        # Filter by score
        filtered = self.filter_by_threshold(results, min_score)

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSONL
        count = 0
        with open(output_path, "w") as f:
            for result in filtered:
                # Build record
                record = {
                    "query": result.get("query"),
                    "answer": result.get("answer"),
                    "agent": result.get("agent"),
                }

                if result.get("contexts"):
                    record["contexts"] = result["contexts"]

                if result.get("ground_truth"):
                    record["ground_truth"] = result["ground_truth"]

                # Add scores
                record["scores"] = self._extract_scores(result)

                # Add metadata
                if include_metadata:
                    record["metadata"] = self._extract_metadata(result)

                f.write(json.dumps(record) + "\n")
                count += 1

        logger.info(f"Exported {count} results to {output_path}")
        return count

    def export_to_openai_format(
        self,
        results: List[Dict[str, Any]],
        output_path: Path,
        min_score: float = 0.0,
        system_prompts: Optional[Dict[str, str]] = None,
        include_metadata: bool = True,
    ) -> int:
        """
        Export results to OpenAI fine-tuning format.

        Format:
        {"messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ], "metadata": {...}}

        Args:
            results: List of evaluation result dictionaries
            output_path: Output file path
            min_score: Minimum overall score to include
            system_prompts: Dict of agent name -> system prompt
            include_metadata: Include metadata with scores

        Returns:
            Number of results exported
        """
        logger.info(f"Exporting to OpenAI format: {output_path}")

        # Default system prompts
        if system_prompts is None:
            system_prompts = {
                "verification": "You are a query verification agent for Singapore government data. Validate if queries are relevant to employment, salary, or labour statistics and check if years are specified.",
                "coordinator": "You are a data coordinator agent. Analyze user queries, identify required datasets, and create execution plans for the data pipeline.",
                "extraction": "You are a data extraction agent. Match user queries to Singapore government datasets and load relevant data from CSV/Excel files.",
                "analytics": "You are a data analytics agent. Generate pandas code to analyze data and create visualizations to answer user queries.",
            }

        # Filter by score
        filtered = self.filter_by_threshold(results, min_score)

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSONL
        count = 0
        with open(output_path, "w") as f:
            for result in filtered:
                agent = result.get("agent", "analytics")
                system_prompt = system_prompts.get(agent, "You are a helpful data assistant.")

                # Build messages
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": result.get("query")},
                    {"role": "assistant", "content": result.get("answer")},
                ]

                # Build record
                record = {"messages": messages}

                # Add metadata with scores
                if include_metadata:
                    record["metadata"] = self._extract_scores(result)
                    record["metadata"].update(self._extract_metadata(result))

                f.write(json.dumps(record) + "\n")
                count += 1

        logger.info(f"Exported {count} results to {output_path}")
        return count

    def export_to_anthropic_format(
        self,
        results: List[Dict[str, Any]],
        output_path: Path,
        min_score: float = 0.0,
        system_prompts: Optional[Dict[str, str]] = None,
        include_metadata: bool = True,
    ) -> int:
        """
        Export results to Anthropic fine-tuning format.

        Format:
        {"prompt": "System: ...\n\nHuman: ...\n\nAssistant:",
         "completion": " ...",
         "metadata": {...}}

        Args:
            results: List of evaluation result dictionaries
            output_path: Output file path
            min_score: Minimum overall score to include
            system_prompts: Dict of agent name -> system prompt
            include_metadata: Include metadata with scores

        Returns:
            Number of results exported
        """
        logger.info(f"Exporting to Anthropic format: {output_path}")

        # Default system prompts
        if system_prompts is None:
            system_prompts = {
                "verification": "You are a query verification agent for Singapore government data.",
                "coordinator": "You are a data coordinator agent.",
                "extraction": "You are a data extraction agent.",
                "analytics": "You are a data analytics agent.",
            }

        # Filter by score
        filtered = self.filter_by_threshold(results, min_score)

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSONL
        count = 0
        with open(output_path, "w") as f:
            for result in filtered:
                agent = result.get("agent", "analytics")
                system_prompt = system_prompts.get(agent, "You are a helpful assistant.")

                # Build prompt (without completion)
                prompt = f"{system_prompt}\n\nHuman: {result.get('query')}\n\nAssistant:"

                # Completion starts with space (Anthropic convention)
                completion = f" {result.get('answer')}"

                # Build record
                record = {
                    "prompt": prompt,
                    "completion": completion,
                }

                # Add metadata
                if include_metadata:
                    record["metadata"] = self._extract_scores(result)
                    record["metadata"].update(self._extract_metadata(result))

                f.write(json.dumps(record) + "\n")
                count += 1

        logger.info(f"Exported {count} results to {output_path}")
        return count

    def export_to_csv(
        self,
        results: List[Dict[str, Any]],
        output_path: Path,
        min_score: float = 0.0,
    ) -> int:
        """
        Export results to CSV format.

        Args:
            results: List of evaluation result dictionaries
            output_path: Output file path
            min_score: Minimum overall score to include

        Returns:
            Number of results exported
        """
        logger.info(f"Exporting to CSV: {output_path}")

        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas not installed, cannot export to CSV")
            return 0

        # Filter by score
        filtered = self.filter_by_threshold(results, min_score)

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build DataFrame
        rows = []
        for result in filtered:
            row = {
                "query": result.get("query"),
                "answer": result.get("answer"),
                "agent": result.get("agent"),
                "overall_score": self._calculate_overall_score(result),
            }

            # Add scores
            row.update(self._extract_scores(result))

            # Add metadata
            row.update(self._extract_metadata(result))

            rows.append(row)

        df = pd.DataFrame(rows)

        # Write CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} results to {output_path}")

        return len(df)

    def export_to_parquet(
        self,
        results: List[Dict[str, Any]],
        output_path: Path,
        min_score: float = 0.0,
    ) -> int:
        """
        Export results to Parquet format.

        Args:
            results: List of evaluation result dictionaries
            output_path: Output file path
            min_score: Minimum overall score to include

        Returns:
            Number of results exported
        """
        logger.info(f"Exporting to Parquet: {output_path}")

        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas not installed, cannot export to Parquet")
            return 0

        # Filter by score
        filtered = self.filter_by_threshold(results, min_score)

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build DataFrame
        rows = []
        for result in filtered:
            row = {
                "query": result.get("query"),
                "answer": result.get("answer"),
                "agent": result.get("agent"),
                "overall_score": self._calculate_overall_score(result),
            }

            # Add scores
            row.update(self._extract_scores(result))

            # Add metadata
            row.update(self._extract_metadata(result))

            # Convert lists to strings (Parquet doesn't support nested lists well)
            if result.get("contexts"):
                row["contexts"] = json.dumps(result["contexts"])

            rows.append(row)

        df = pd.DataFrame(rows)

        # Write Parquet
        df.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info(f"Exported {len(df)} results to {output_path}")

        return len(df)

    def filter_by_threshold(
        self,
        results: List[Dict[str, Any]],
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Filter results by minimum overall score.

        Args:
            results: List of evaluation result dictionaries
            min_score: Minimum overall score (0.0 to 1.0)

        Returns:
            Filtered list of results
        """
        if min_score <= 0.0:
            return results

        filtered = []
        for result in results:
            score = self._calculate_overall_score(result)
            if score >= min_score:
                filtered.append(result)

        logger.info(
            f"Filtered {len(results)} results to {len(filtered)} "
            f"with min_score={min_score}"
        )

        return filtered

    def generate_summary_report(
        self,
        results: List[Dict[str, Any]],
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate summary report with aggregate metrics.

        Args:
            results: List of evaluation result dictionaries
            output_path: Optional path to save report JSON

        Returns:
            Summary report dictionary
        """
        logger.info("Generating summary report")

        # Calculate aggregate metrics
        total = len(results)
        by_agent = {}

        for result in results:
            agent = result.get("agent", "unknown")
            if agent not in by_agent:
                by_agent[agent] = {
                    "count": 0,
                    "scores": [],
                    "latencies": [],
                }

            by_agent[agent]["count"] += 1
            by_agent[agent]["scores"].append(self._calculate_overall_score(result))

            if result.get("latency_ms"):
                by_agent[agent]["latencies"].append(result["latency_ms"])

        # Build report
        report = {
            "evaluation_id": f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "total_test_cases": total,
            "agents_evaluated": list(by_agent.keys()),
            "aggregate_metrics": self._aggregate_metrics(results),
            "by_agent": {},
        }

        for agent, data in by_agent.items():
            scores = data["scores"]
            latencies = data["latencies"]

            report["by_agent"][agent] = {
                "test_cases": data["count"],
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "pass_rate": sum(1 for s in scores if s >= 0.80) / len(scores)
                if scores
                else 0.0,
            }

            if latencies:
                report["by_agent"][agent]["avg_latency_ms"] = sum(latencies) / len(
                    latencies
                )

        # Save to file if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved summary report to {output_path}")

        return report

    def _extract_scores(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Extract score fields from result."""
        scores = {}

        score_fields = [
            "ragas_context_precision",
            "ragas_context_recall",
            "ragas_faithfulness",
            "ragas_answer_relevancy",
            "ragas_answer_correctness",
            "bertscore_precision",
            "bertscore_recall",
            "bertscore_f1",
            "llm_judge_overall",
            "llm_judge_accuracy",
            "llm_judge_completeness",
            "llm_judge_clarity",
            "llm_judge_conciseness",
            "llm_judge_code_quality",
            "llm_judge_visualization",
        ]

        for field in score_fields:
            if field in result and result[field] is not None:
                scores[field] = result[field]

        return scores

    def _extract_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata fields from result."""
        metadata = {}

        metadata_fields = [
            "timestamp",
            "latency_ms",
            "tokens_used",
            "model_name",
            "agent",
        ]

        for field in metadata_fields:
            if field in result and result[field] is not None:
                metadata[field] = result[field]

        return metadata

    def _calculate_overall_score(self, result: Dict[str, Any]) -> float:
        """
        Calculate aggregate overall score.

        Weights:
        - Ragas metrics: 40%
        - BERTScore: 20%
        - LLM Judge: 40%
        """
        scores = []

        # Ragas metrics (weight: 0.4)
        ragas_scores = [
            result.get("ragas_faithfulness"),
            result.get("ragas_answer_relevancy"),
            result.get("ragas_answer_correctness"),
        ]
        ragas_avg = sum(s for s in ragas_scores if s is not None) / len(
            [s for s in ragas_scores if s is not None]
        ) if any(s is not None for s in ragas_scores) else None
        if ragas_avg:
            scores.append(ragas_avg * 0.4)

        # BERTScore (weight: 0.2)
        if result.get("bertscore_f1"):
            scores.append(result["bertscore_f1"] * 0.2)

        # LLM Judge (weight: 0.4, scale from 5.0 to 1.0)
        if result.get("llm_judge_overall"):
            scores.append((result["llm_judge_overall"] / 5.0) * 0.4)

        return sum(scores) if scores else 0.0

    def _aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all results."""
        metrics = {
            "ragas": {},
            "bertscore": {},
            "llm_judge": {},
        }

        # Ragas
        ragas_fields = [
            "ragas_context_precision",
            "ragas_context_recall",
            "ragas_faithfulness",
            "ragas_answer_relevancy",
            "ragas_answer_correctness",
        ]
        for field in ragas_fields:
            values = [r.get(field) for r in results if r.get(field) is not None]
            if values:
                key = field.replace("ragas_", "")
                metrics["ragas"][key] = sum(values) / len(values)

        # BERTScore
        bertscore_fields = ["bertscore_precision", "bertscore_recall", "bertscore_f1"]
        for field in bertscore_fields:
            values = [r.get(field) for r in results if r.get(field) is not None]
            if values:
                key = field.replace("bertscore_", "")
                metrics["bertscore"][key] = sum(values) / len(values)

        # LLM Judge
        llm_judge_fields = [
            "llm_judge_overall",
            "llm_judge_accuracy",
            "llm_judge_completeness",
            "llm_judge_clarity",
            "llm_judge_conciseness",
        ]
        for field in llm_judge_fields:
            values = [r.get(field) for r in results if r.get(field) is not None]
            if values:
                key = field.replace("llm_judge_", "")
                metrics["llm_judge"][key] = sum(values) / len(values)

        return metrics

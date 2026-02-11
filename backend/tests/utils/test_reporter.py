"""
Test Reporter for Generating Comprehensive Test Reports.

This module generates markdown reports summarizing:
- Test execution results (pass/fail counts, coverage)
- Retrieval metrics (Ragas: precision, recall, MRR, NDCG)
- Generation metrics (BERTScore, faithfulness, relevancy)
- LLM as judge scores
- Performance metrics (latency percentiles)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TestSummary:
    """Summary of test execution."""
    total_tests: int
    passed: int
    failed: int
    skipped: int
    coverage_percent: float
    duration_seconds: float


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics."""
    context_precision_at_1: float
    context_precision_at_3: float
    context_precision_at_5: float
    context_recall: float
    mrr: float
    ndcg: float


@dataclass
class GenerationMetrics:
    """Generation evaluation metrics."""
    bertscore_precision: float
    bertscore_recall: float
    bertscore_f1: float
    ragas_faithfulness: float
    ragas_answer_relevancy: float
    ragas_answer_correctness: float


@dataclass
class LLMJudgeMetrics:
    """LLM as judge evaluation metrics."""
    overall_score: float
    accuracy_score: float
    completeness_score: float
    clarity_score: float
    conciseness_score: float


@dataclass
class PerformanceMetrics:
    """Performance and latency metrics."""
    rag_retrieval_p50_ms: float
    rag_retrieval_p90_ms: float
    code_generation_p50_ms: float
    code_generation_p90_ms: float
    end_to_end_p50_ms: float
    end_to_end_p90_ms: float


class TestReporter:
    """
    Generate comprehensive test reports.

    Aggregates results from pytest, ragas, BERTScore, LLM judge,
    and performance tests into formatted markdown reports.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize test reporter.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized TestReporter, output dir: {self.output_dir}")

    def generate_report(
        self,
        test_summary: TestSummary,
        retrieval_metrics: Optional[RetrievalMetrics] = None,
        generation_metrics: Optional[GenerationMetrics] = None,
        llm_judge_metrics: Optional[LLMJudgeMetrics] = None,
        performance_metrics: Optional[PerformanceMetrics] = None,
        additional_notes: Optional[str] = None,
    ) -> Path:
        """
        Generate comprehensive test report.

        Args:
            test_summary: Test execution summary
            retrieval_metrics: Optional retrieval metrics
            generation_metrics: Optional generation metrics
            llm_judge_metrics: Optional LLM judge metrics
            performance_metrics: Optional performance metrics
            additional_notes: Optional additional notes

        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = self.output_dir / f"test_report_{timestamp}.md"

        logger.info(f"Generating test report: {report_path}")

        with open(report_path, "w") as f:
            # Header
            f.write(f"# Test Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Test execution summary
            f.write(self._format_test_summary(test_summary))

            # Retrieval metrics
            if retrieval_metrics:
                f.write(self._format_retrieval_metrics(retrieval_metrics))

            # Generation metrics
            if generation_metrics:
                f.write(self._format_generation_metrics(generation_metrics))

            # LLM judge metrics
            if llm_judge_metrics:
                f.write(self._format_llm_judge_metrics(llm_judge_metrics))

            # Performance metrics
            if performance_metrics:
                f.write(self._format_performance_metrics(performance_metrics))

            # Additional notes
            if additional_notes:
                f.write("\n## Additional Notes\n\n")
                f.write(f"{additional_notes}\n\n")

            # Footer
            f.write(self._format_footer())

        logger.info(f"Report generated successfully: {report_path}")

        # Also save JSON version
        self._save_json_report(report_path, test_summary, retrieval_metrics,
                              generation_metrics, llm_judge_metrics, performance_metrics)

        return report_path

    def _format_test_summary(self, summary: TestSummary) -> str:
        """Format test execution summary section."""
        pass_rate = (summary.passed / summary.total_tests * 100) if summary.total_tests > 0 else 0

        return f"""## Test Execution Summary

| Metric | Value |
|--------|-------|
| **Total Tests** | {summary.total_tests} |
| **Passed** | {summary.passed} ✅ |
| **Failed** | {summary.failed} ❌ |
| **Skipped** | {summary.skipped} ⚠️ |
| **Pass Rate** | {pass_rate:.1f}% |
| **Coverage** | {summary.coverage_percent:.1f}% |
| **Duration** | {summary.duration_seconds:.1f}s |

"""

    def _format_retrieval_metrics(self, metrics: RetrievalMetrics) -> str:
        """Format retrieval metrics section."""
        return f"""## Retrieval Metrics (Ragas)

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| **Context Precision @ 1** | {metrics.context_precision_at_1:.4f} | 0.85 | {self._status_icon(metrics.context_precision_at_1, 0.85)} |
| **Context Precision @ 3** | {metrics.context_precision_at_3:.4f} | 0.75 | {self._status_icon(metrics.context_precision_at_3, 0.75)} |
| **Context Precision @ 5** | {metrics.context_precision_at_5:.4f} | 0.70 | {self._status_icon(metrics.context_precision_at_5, 0.70)} |
| **Context Recall** | {metrics.context_recall:.4f} | 0.90 | {self._status_icon(metrics.context_recall, 0.90)} |
| **MRR** | {metrics.mrr:.4f} | 0.80 | {self._status_icon(metrics.mrr, 0.80)} |
| **NDCG** | {metrics.ndcg:.4f} | 0.75 | {self._status_icon(metrics.ndcg, 0.75)} |

### Interpretation
- **Context Precision**: Measures relevance of retrieved contexts (higher = less noise)
- **Context Recall**: Measures completeness of information retrieval (higher = more complete)
- **MRR**: Mean Reciprocal Rank - how quickly the right answer is found
- **NDCG**: Normalized Discounted Cumulative Gain - overall ranking quality

"""

    def _format_generation_metrics(self, metrics: GenerationMetrics) -> str:
        """Format generation metrics section."""
        return f"""## Generation Metrics

### BERTScore (Semantic Similarity)

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| **Precision** | {metrics.bertscore_precision:.4f} | 0.70 | {self._status_icon(metrics.bertscore_precision, 0.70)} |
| **Recall** | {metrics.bertscore_recall:.4f} | 0.70 | {self._status_icon(metrics.bertscore_recall, 0.70)} |
| **F1** | {metrics.bertscore_f1:.4f} | 0.75 | {self._status_icon(metrics.bertscore_f1, 0.75)} |

### Ragas (Faithfulness & Relevancy)

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| **Faithfulness** | {metrics.ragas_faithfulness:.4f} | 0.85 | {self._status_icon(metrics.ragas_faithfulness, 0.85)} |
| **Answer Relevancy** | {metrics.ragas_answer_relevancy:.4f} | 0.80 | {self._status_icon(metrics.ragas_answer_relevancy, 0.80)} |
| **Answer Correctness** | {metrics.ragas_answer_correctness:.4f} | 0.75 | {self._status_icon(metrics.ragas_answer_correctness, 0.75)} |

### Interpretation
- **BERTScore**: Semantic similarity with ground truth (transformer-based)
- **Faithfulness**: Answer is grounded in retrieved context (no hallucination)
- **Answer Relevancy**: Answer addresses the user query
- **Answer Correctness**: Factual accuracy compared to ground truth

"""

    def _format_llm_judge_metrics(self, metrics: LLMJudgeMetrics) -> str:
        """Format LLM judge metrics section."""
        return f"""## LLM as Judge Evaluation

| Criterion | Score (1-5) | Threshold | Status |
|-----------|-------------|-----------|--------|
| **Overall** | {metrics.overall_score:.2f} | 3.5 | {self._status_icon(metrics.overall_score, 3.5, scale_5=True)} |
| **Accuracy** | {metrics.accuracy_score:.2f} | 4.0 | {self._status_icon(metrics.accuracy_score, 4.0, scale_5=True)} |
| **Completeness** | {metrics.completeness_score:.2f} | 3.5 | {self._status_icon(metrics.completeness_score, 3.5, scale_5=True)} |
| **Clarity** | {metrics.clarity_score:.2f} | 3.5 | {self._status_icon(metrics.clarity_score, 3.5, scale_5=True)} |
| **Conciseness** | {metrics.conciseness_score:.2f} | 3.0 | {self._status_icon(metrics.conciseness_score, 3.0, scale_5=True)} |

### Interpretation
- **1-2**: Poor quality, significant issues
- **3**: Acceptable quality, some improvements needed
- **4**: Very good quality, minor improvements possible
- **5**: Excellent quality, no issues

"""

    def _format_performance_metrics(self, metrics: PerformanceMetrics) -> str:
        """Format performance metrics section."""
        return f"""## Performance Metrics

### RAG Retrieval Latency

| Percentile | Latency | Threshold | Status |
|------------|---------|-----------|--------|
| **p50** | {metrics.rag_retrieval_p50_ms:.0f}ms | 2000ms | {self._latency_status(metrics.rag_retrieval_p50_ms, 2000)} |
| **p90** | {metrics.rag_retrieval_p90_ms:.0f}ms | 4000ms | {self._latency_status(metrics.rag_retrieval_p90_ms, 4000)} |

### Code Generation Latency

| Percentile | Latency | Threshold | Status |
|------------|---------|-----------|--------|
| **p50** | {metrics.code_generation_p50_ms:.0f}ms | 10000ms | {self._latency_status(metrics.code_generation_p50_ms, 10000)} |
| **p90** | {metrics.code_generation_p90_ms:.0f}ms | 15000ms | {self._latency_status(metrics.code_generation_p90_ms, 15000)} |

### End-to-End Pipeline Latency

| Percentile | Latency | Threshold | Status |
|------------|---------|-----------|--------|
| **p50** | {metrics.end_to_end_p50_ms:.0f}ms | 30000ms | {self._latency_status(metrics.end_to_end_p50_ms, 30000)} |
| **p90** | {metrics.end_to_end_p90_ms:.0f}ms | 45000ms | {self._latency_status(metrics.end_to_end_p90_ms, 45000)} |

"""

    def _format_footer(self) -> str:
        """Format report footer."""
        return f"""---

**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Legend:**
- ✅ Meets threshold
- ⚠️ Below threshold (needs improvement)
- ❌ Significantly below threshold (critical)
"""

    def _status_icon(self, value: float, threshold: float, scale_5: bool = False) -> str:
        """Get status icon based on value vs threshold."""
        if scale_5:
            # For 1-5 scale
            if value >= threshold:
                return "✅"
            elif value >= threshold - 0.5:
                return "⚠️"
            else:
                return "❌"
        else:
            # For 0-1 scale
            if value >= threshold:
                return "✅"
            elif value >= threshold - 0.10:
                return "⚠️"
            else:
                return "❌"

    def _latency_status(self, latency_ms: float, threshold_ms: float) -> str:
        """Get status icon for latency metrics (lower is better)."""
        if latency_ms <= threshold_ms:
            return "✅"
        elif latency_ms <= threshold_ms * 1.2:
            return "⚠️"
        else:
            return "❌"

    def _save_json_report(
        self,
        markdown_path: Path,
        test_summary: TestSummary,
        retrieval_metrics: Optional[RetrievalMetrics],
        generation_metrics: Optional[GenerationMetrics],
        llm_judge_metrics: Optional[LLMJudgeMetrics],
        performance_metrics: Optional[PerformanceMetrics],
    ) -> None:
        """Save JSON version of report for programmatic access."""
        json_path = markdown_path.with_suffix(".json")

        data = {
            "timestamp": datetime.now().isoformat(),
            "test_summary": asdict(test_summary),
        }

        if retrieval_metrics:
            data["retrieval_metrics"] = asdict(retrieval_metrics)
        if generation_metrics:
            data["generation_metrics"] = asdict(generation_metrics)
        if llm_judge_metrics:
            data["llm_judge_metrics"] = asdict(llm_judge_metrics)
        if performance_metrics:
            data["performance_metrics"] = asdict(performance_metrics)

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"JSON report saved: {json_path}")

    def generate_comparison_report(
        self,
        reports: List[Path],
        output_name: str = "comparison_report.md"
    ) -> Path:
        """
        Generate comparison report from multiple test reports.

        Useful for tracking metrics over time or comparing different configurations.

        Args:
            reports: List of paths to JSON report files
            output_name: Name for comparison report

        Returns:
            Path to generated comparison report
        """
        logger.info(f"Generating comparison report from {len(reports)} reports")

        comparison_data = []
        for report_path in reports:
            json_path = report_path.with_suffix(".json")
            if json_path.exists():
                with open(json_path) as f:
                    comparison_data.append(json.load(f))

        if not comparison_data:
            logger.warning("No JSON reports found for comparison")
            return None

        output_path = self.output_dir / output_name

        with open(output_path, "w") as f:
            f.write("# Test Metrics Comparison Report\n\n")
            f.write(f"Comparing {len(comparison_data)} test runs\n\n")

            # Test summary comparison
            f.write("## Test Execution Trends\n\n")
            f.write("| Timestamp | Total | Passed | Failed | Coverage |\n")
            f.write("|-----------|-------|--------|--------|----------|\n")

            for data in comparison_data:
                ts = data["timestamp"][:19]  # Trim to readable format
                summary = data["test_summary"]
                f.write(
                    f"| {ts} | {summary['total_tests']} | "
                    f"{summary['passed']} | {summary['failed']} | "
                    f"{summary['coverage_percent']:.1f}% |\n"
                )

            # Retrieval metrics trends
            if all("retrieval_metrics" in d for d in comparison_data):
                f.write("\n## Retrieval Metrics Trends\n\n")
                f.write("| Timestamp | Precision@1 | Recall | MRR |\n")
                f.write("|-----------|-------------|--------|-----|\n")

                for data in comparison_data:
                    ts = data["timestamp"][:19]
                    metrics = data["retrieval_metrics"]
                    f.write(
                        f"| {ts} | {metrics['context_precision_at_1']:.4f} | "
                        f"{metrics['context_recall']:.4f} | {metrics['mrr']:.4f} |\n"
                    )

            # Generation metrics trends
            if all("generation_metrics" in d for d in comparison_data):
                f.write("\n## Generation Metrics Trends\n\n")
                f.write("| Timestamp | BERTScore F1 | Faithfulness | Relevancy |\n")
                f.write("|-----------|--------------|--------------|----------|\n")

                for data in comparison_data:
                    ts = data["timestamp"][:19]
                    metrics = data["generation_metrics"]
                    f.write(
                        f"| {ts} | {metrics['bertscore_f1']:.4f} | "
                        f"{metrics['ragas_faithfulness']:.4f} | "
                        f"{metrics['ragas_answer_relevancy']:.4f} |\n"
                    )

        logger.info(f"Comparison report generated: {output_path}")
        return output_path

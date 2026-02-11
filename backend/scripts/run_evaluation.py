#!/usr/bin/env python3
"""
Evaluation Runner Script.

CLI tool to run evaluation tests and export results for fine-tuning.

Usage:
    # Full evaluation with export
    python scripts/run_evaluation.py \\
      --test-suite evaluation \\
      --export-format openai \\
      --output-dir ./fine_tuning_data \\
      --min-score 0.80

    # Agent-specific evaluation
    python scripts/run_evaluation.py \\
      --agent verification \\
      --export-format openai \\
      --output-dir ./verification_data

    # Custom queries
    python scripts/run_evaluation.py \\
      --queries-file tests/fixtures/custom_queries.json \\
      --ground-truth tests/fixtures/custom_answers.json \\
      --export-format jsonl

Export formats:
    - jsonl: Generic JSONL format
    - openai: OpenAI fine-tuning format
    - anthropic: Anthropic fine-tuning format
    - csv: CSV format for spreadsheet analysis
    - parquet: Parquet format for big data analysis
"""

import argparse
import json
import logging
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utils.evaluation_export import EvaluationExporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class EvaluationRunner:
    """
    CLI tool to run evaluations and export results.

    Features:
    - Run pytest with specific markers
    - Collect evaluation results
    - Export in multiple formats
    - Generate summary reports
    """

    def __init__(self, args: argparse.Namespace):
        """Initialize runner with CLI arguments."""
        self.args = args
        self.exporter = EvaluationExporter()
        self.results: List[Dict[str, Any]] = []

    def run(self) -> int:
        """
        Run evaluation pipeline.

        Returns:
            Exit code (0 = success, 1 = failure)
        """
        logger.info("=" * 80)
        logger.info("Starting Evaluation Pipeline")
        logger.info("=" * 80)

        try:
            # Step 1: Run pytest tests
            if not self.args.skip_tests:
                logger.info("\n[1/4] Running pytest tests...")
                if not self._run_pytest():
                    logger.error("Tests failed!")
                    return 1
            else:
                logger.info("\n[1/4] Skipping tests (--skip-tests)")

            # Step 2: Collect results
            logger.info("\n[2/4] Collecting evaluation results...")
            self._collect_results()

            if not self.results:
                logger.warning("No results collected!")
                return 1

            logger.info(f"Collected {len(self.results)} evaluation results")

            # Step 3: Export results
            logger.info(f"\n[3/4] Exporting to {self.args.export_format} format...")
            self._export_results()

            # Step 4: Generate summary report
            logger.info("\n[4/4] Generating summary report...")
            self._generate_summary()

            logger.info("\n" + "=" * 80)
            logger.info("Evaluation Pipeline Completed Successfully!")
            logger.info("=" * 80)
            logger.info(f"Output directory: {self.args.output_dir}")

            return 0

        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
            return 1

    def _run_pytest(self) -> bool:
        """
        Run pytest with appropriate markers.

        Returns:
            True if tests passed, False otherwise
        """
        # Build pytest command
        cmd = ["pytest"]

        # Add test suite marker
        if self.args.test_suite:
            cmd.extend(["-m", self.args.test_suite])

        # Add specific test file/path if provided
        if self.args.test_path:
            cmd.append(self.args.test_path)
        else:
            # Default to evaluation tests
            cmd.append("tests/evaluation/")

        # Add verbosity
        if self.args.verbose:
            cmd.append("-vv")
        else:
            cmd.append("-v")

        # Add JSON report for result collection
        json_report_path = Path(self.args.output_dir) / "pytest_report.json"
        json_report_path.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--json-report", f"--json-report-file={json_report_path}"])

        # Add other options
        if self.args.parallel:
            cmd.extend(["-n", "auto"])

        if self.args.maxfail:
            cmd.extend(["--maxfail", str(self.args.maxfail)])

        logger.info(f"Running: {' '.join(cmd)}")

        # Run pytest
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

        return result.returncode == 0

    def _collect_results(self) -> None:
        """
        Collect evaluation results from various sources.

        Sources:
        - pytest JSON report
        - Custom queries file
        - Mock/sample results (for testing)
        """
        # Try to load pytest JSON report
        json_report_path = Path(self.args.output_dir) / "pytest_report.json"
        if json_report_path.exists():
            self._collect_from_pytest_report(json_report_path)

        # Load custom queries if provided
        if self.args.queries_file:
            self._collect_from_custom_queries()

        # If no results, generate mock results for testing
        if not self.results and self.args.generate_mock:
            logger.warning("No results found, generating mock data for testing")
            self._generate_mock_results()

    def _collect_from_pytest_report(self, report_path: Path) -> None:
        """Collect results from pytest JSON report."""
        try:
            with open(report_path, "r") as f:
                report = json.load(f)

            # Extract test results
            # Note: This is a simplified parser - adjust based on actual report structure
            for test in report.get("tests", []):
                if test.get("outcome") == "passed":
                    # Extract metadata from test
                    # This would need to be adapted based on actual test structure
                    result = {
                        "query": test.get("nodeid", "unknown"),
                        "answer": "Test passed",
                        "agent": self._extract_agent_from_test(test),
                        "timestamp": datetime.now().isoformat(),
                    }
                    self.results.append(result)

        except Exception as e:
            logger.warning(f"Could not parse pytest report: {e}")

    def _collect_from_custom_queries(self) -> None:
        """Collect results from custom queries file."""
        try:
            queries_path = Path(self.args.queries_file)
            with open(queries_path, "r") as f:
                queries = json.load(f)

            # Load ground truth if provided
            ground_truth = {}
            if self.args.ground_truth:
                with open(self.args.ground_truth, "r") as f:
                    ground_truth = json.load(f)

            # For each query, create a result entry
            # In a real scenario, you'd run the query through the system
            for query_data in queries:
                query = query_data.get("query")
                agent = query_data.get("agent", "analytics")

                result = {
                    "query": query,
                    "answer": ground_truth.get(query, {}).get("answer", ""),
                    "agent": agent,
                    "contexts": ground_truth.get(query, {}).get("contexts", []),
                    "ground_truth": ground_truth.get(query, {}).get("answer", ""),
                    "timestamp": datetime.now().isoformat(),
                }

                self.results.append(result)

        except Exception as e:
            logger.error(f"Error loading custom queries: {e}")

    def _generate_mock_results(self) -> None:
        """
        Generate mock evaluation results for testing.

        This is useful for testing the export functionality without
        running actual evaluations.
        """
        logger.info("Generating mock evaluation results")

        agents = ["verification", "coordinator", "extraction", "analytics"]
        queries = [
            "What was the unemployment rate in 2023?",
            "Show me salary trends over the last 5 years",
            "Compare employment across different sectors",
            "What is the average income in Singapore?",
        ]

        for i, query in enumerate(queries):
            agent = agents[i % len(agents)]

            result = {
                "query": query,
                "answer": f"Mock answer for: {query}",
                "agent": agent,
                "contexts": [
                    f"Context 1 for {query}",
                    f"Context 2 for {query}",
                ],
                "ground_truth": f"Ground truth answer for: {query}",
                # Ragas metrics
                "ragas_context_precision": 0.85 + (i * 0.02),
                "ragas_context_recall": 0.90 + (i * 0.01),
                "ragas_faithfulness": 0.88 + (i * 0.02),
                "ragas_answer_relevancy": 0.82 + (i * 0.03),
                "ragas_answer_correctness": 0.80 + (i * 0.02),
                # BERTScore metrics
                "bertscore_precision": 0.75 + (i * 0.02),
                "bertscore_recall": 0.73 + (i * 0.02),
                "bertscore_f1": 0.74 + (i * 0.02),
                # LLM Judge metrics
                "llm_judge_overall": 4.0 + (i * 0.1),
                "llm_judge_accuracy": 4.2 + (i * 0.1),
                "llm_judge_completeness": 3.8 + (i * 0.1),
                "llm_judge_clarity": 4.0 + (i * 0.1),
                "llm_judge_conciseness": 3.5 + (i * 0.1),
                # Metadata
                "timestamp": datetime.now().isoformat(),
                "latency_ms": 2000 + (i * 500),
                "tokens_used": 500 + (i * 100),
                "model_name": "claude-sonnet-4-5",
            }

            self.results.append(result)

        # Add some analytics-specific results with code quality
        for i in range(2):
            result = {
                "query": f"Generate code to analyze dataset {i+1}",
                "answer": f"Here's the code to analyze dataset {i+1}:\n\n```python\nimport pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.describe())\n```",
                "agent": "analytics",
                "contexts": [f"Dataset {i+1} schema", f"Dataset {i+1} samples"],
                "ragas_faithfulness": 0.92,
                "ragas_answer_relevancy": 0.88,
                "bertscore_f1": 0.80,
                "llm_judge_overall": 4.3,
                "llm_judge_accuracy": 4.5,
                "llm_judge_code_quality": 4.0,
                "llm_judge_visualization": 3.8,
                "timestamp": datetime.now().isoformat(),
                "latency_ms": 5000,
                "model_name": "claude-sonnet-4-5",
            }
            self.results.append(result)

    def _export_results(self) -> None:
        """Export results in specified format."""
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        export_format = self.args.export_format
        min_score = self.args.min_score

        # Filter by agent if specified
        results_to_export = self.results
        if self.args.agent:
            results_to_export = [
                r for r in self.results if r.get("agent") == self.args.agent
            ]
            logger.info(
                f"Filtered to {len(results_to_export)} results for agent: {self.args.agent}"
            )

        # Export based on format
        if export_format == "jsonl":
            output_path = output_dir / "evaluation_results.jsonl"
            self.exporter.export_to_jsonl(
                results_to_export,
                output_path,
                min_score=min_score,
                include_metadata=self.args.include_metadata,
            )

        elif export_format == "openai":
            output_path = output_dir / "training_data_openai.jsonl"
            self.exporter.export_to_openai_format(
                results_to_export,
                output_path,
                min_score=min_score,
                include_metadata=self.args.include_metadata,
            )

        elif export_format == "anthropic":
            output_path = output_dir / "training_data_anthropic.jsonl"
            self.exporter.export_to_anthropic_format(
                results_to_export,
                output_path,
                min_score=min_score,
                include_metadata=self.args.include_metadata,
            )

        elif export_format == "csv":
            output_path = output_dir / "evaluation_results.csv"
            self.exporter.export_to_csv(
                results_to_export, output_path, min_score=min_score
            )

        elif export_format == "parquet":
            output_path = output_dir / "evaluation_results.parquet"
            self.exporter.export_to_parquet(
                results_to_export, output_path, min_score=min_score
            )

        else:
            logger.error(f"Unknown export format: {export_format}")
            return

        logger.info(f"Exported results to: {output_path}")

    def _generate_summary(self) -> None:
        """Generate summary report."""
        output_path = Path(self.args.output_dir) / "summary_report.json"

        summary = self.exporter.generate_summary_report(
            self.results, output_path=output_path
        )

        # Print summary to console
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY REPORT")
        logger.info("=" * 80)
        logger.info(f"Total Test Cases: {summary['total_test_cases']}")
        logger.info(f"Agents Evaluated: {', '.join(summary['agents_evaluated'])}")

        logger.info("\nAggregate Metrics:")
        for category, metrics in summary.get("aggregate_metrics", {}).items():
            logger.info(f"  {category.upper()}:")
            for metric, value in metrics.items():
                logger.info(f"    {metric}: {value:.4f}")

        logger.info("\nBy Agent:")
        for agent, data in summary.get("by_agent", {}).items():
            logger.info(f"  {agent.upper()}:")
            logger.info(f"    Test Cases: {data['test_cases']}")
            logger.info(f"    Avg Score: {data['avg_score']:.4f}")
            logger.info(f"    Pass Rate: {data['pass_rate']:.2%}")
            if "avg_latency_ms" in data:
                logger.info(f"    Avg Latency: {data['avg_latency_ms']:.0f}ms")

    def _extract_agent_from_test(self, test: Dict[str, Any]) -> str:
        """Extract agent name from test nodeid."""
        nodeid = test.get("nodeid", "")

        if "verification" in nodeid.lower():
            return "verification"
        elif "coordinator" in nodeid.lower():
            return "coordinator"
        elif "extraction" in nodeid.lower():
            return "extraction"
        elif "analytics" in nodeid.lower():
            return "analytics"
        else:
            return "unknown"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run evaluation tests and export results for fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation with OpenAI export
  python scripts/run_evaluation.py --export-format openai --output-dir ./fine_tuning_data

  # Agent-specific evaluation
  python scripts/run_evaluation.py --agent analytics --export-format jsonl

  # Custom queries
  python scripts/run_evaluation.py --queries-file queries.json --export-format csv

  # Generate mock data (for testing)
  python scripts/run_evaluation.py --generate-mock --export-format openai

Export formats: jsonl, openai, anthropic, csv, parquet
        """,
    )

    # Test execution options
    test_group = parser.add_argument_group("Test Execution")
    test_group.add_argument(
        "--test-suite",
        type=str,
        default="evaluation",
        choices=["unit", "integration", "evaluation", "performance", "full"],
        help="Test suite to run (default: evaluation)",
    )
    test_group.add_argument(
        "--test-path", type=str, help="Specific test file or directory to run"
    )
    test_group.add_argument(
        "--agent",
        type=str,
        choices=["verification", "coordinator", "extraction", "analytics"],
        help="Filter by specific agent",
    )
    test_group.add_argument(
        "--skip-tests", action="store_true", help="Skip running tests (use existing results)"
    )
    test_group.add_argument(
        "--parallel", action="store_true", help="Run tests in parallel"
    )
    test_group.add_argument(
        "--maxfail", type=int, help="Stop after N test failures"
    )

    # Export options
    export_group = parser.add_argument_group("Export Options")
    export_group.add_argument(
        "--export-format",
        type=str,
        default="jsonl",
        choices=["jsonl", "openai", "anthropic", "csv", "parquet"],
        help="Export format (default: jsonl)",
    )
    export_group.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for results (default: ./evaluation_results)",
    )
    export_group.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum overall score to include (0.0-1.0, default: 0.0)",
    )
    export_group.add_argument(
        "--include-metadata",
        action="store_true",
        default=True,
        help="Include metadata in exported results (default: True)",
    )

    # Data source options
    data_group = parser.add_argument_group("Data Sources")
    data_group.add_argument(
        "--queries-file", type=str, help="Custom queries JSON file"
    )
    data_group.add_argument(
        "--ground-truth", type=str, help="Ground truth answers JSON file"
    )
    data_group.add_argument(
        "--generate-mock",
        action="store_true",
        help="Generate mock results for testing (no actual evaluation)",
    )

    # Other options
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    runner = EvaluationRunner(args)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())

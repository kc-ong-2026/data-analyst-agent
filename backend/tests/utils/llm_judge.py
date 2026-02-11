"""
LLM as Judge - Quality Evaluation using Claude.

This module implements an LLM-based evaluation system where Claude Sonnet
acts as a judge to evaluate the quality of generated responses across multiple
criteria including accuracy, completeness, clarity, and conciseness.

The judge provides structured scores (1-5 scale) with reasoning for each
criterion, enabling explainable evaluation.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


@dataclass
class JudgmentCriterion:
    """Single evaluation criterion with score and reasoning."""

    name: str
    score: float  # 1-5 scale
    reasoning: str
    issues: List[str] = field(default_factory=list)


@dataclass
class LLMJudgment:
    """Complete judgment from LLM judge."""

    overall_score: float
    criteria_scores: Dict[str, JudgmentCriterion]
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "criteria": {
                name: {
                    "score": criterion.score,
                    "reasoning": criterion.reasoning,
                    "issues": criterion.issues,
                }
                for name, criterion in self.criteria_scores.items()
            },
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendation": self.recommendation,
        }

    def meets_thresholds(self, thresholds: Dict[str, float]) -> bool:
        """Check if all criteria meet their thresholds."""
        # Check overall score
        overall_threshold = thresholds.get("overall_score", 0.0)
        if self.overall_score < overall_threshold:
            return False

        # Check individual criteria
        for criterion_name, criterion in self.criteria_scores.items():
            threshold = thresholds.get(criterion_name, 0.0)
            if criterion.score < threshold:
                return False

        return True


class LLMJudge:
    """
    LLM-based evaluator using Claude Sonnet as a judge.

    Provides structured evaluation of generated text across multiple
    quality criteria with explainable reasoning.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM judge.

        Args:
            config: Configuration dictionary with keys:
                - model: Model name (e.g., "claude-sonnet-4-5")
                - provider: Provider name (e.g., "anthropic")
                - temperature: Sampling temperature (0.0 for deterministic)
                - max_tokens: Maximum tokens for response
                - criteria: List of criteria to evaluate
                - thresholds: Dict of criterion -> threshold
        """
        self.config = config
        self.model = config.get("model", "claude-sonnet-4-5")
        self.provider = config.get("provider", "anthropic")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 2000)
        self.criteria = config.get("criteria", [
            "accuracy", "completeness", "clarity", "conciseness"
        ])
        self.thresholds = config.get("thresholds", {})

        # Initialize LLM
        self.llm = self._init_llm()

        logger.info(
            f"Initialized LLMJudge with model: {self.model}, "
            f"criteria: {self.criteria}"
        )

    def _init_llm(self):
        """Initialize LLM client."""
        try:
            if self.provider == "anthropic":
                return ChatAnthropic(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            return None

    def _build_evaluation_prompt(
        self,
        query: str,
        answer: str,
        context: Optional[str] = None,
        ground_truth: Optional[str] = None,
        criteria: Optional[List[str]] = None,
    ) -> str:
        """
        Build evaluation prompt for the judge.

        Args:
            query: User query
            answer: Generated answer to evaluate
            context: Optional context used for generation
            ground_truth: Optional ground truth answer
            criteria: Optional list of criteria to evaluate

        Returns:
            Formatted prompt string
        """
        criteria = criteria or self.criteria

        prompt = f"""You are an expert evaluator of data analytics systems. Your task is to evaluate the quality of a generated answer to a user query.

**User Query:**
{query}

**Generated Answer:**
{answer}
"""

        if context:
            prompt += f"""
**Context Used (Retrieved Data):**
{context}
"""

        if ground_truth:
            prompt += f"""
**Ground Truth Reference:**
{ground_truth}
"""

        prompt += f"""
**Evaluation Criteria:**
Evaluate the generated answer on the following criteria using a 1-5 scale (1=Poor, 2=Fair, 3=Good, 4=Very Good, 5=Excellent):

"""

        criterion_descriptions = {
            "accuracy": "Are all facts and numbers correct based on the provided context? Are there any hallucinations or unsupported claims?",
            "completeness": "Does the answer address all aspects of the user query? Is any important information missing?",
            "clarity": "Is the answer clear, well-structured, and easy to understand? Is the language appropriate?",
            "conciseness": "Is the answer appropriately brief without sacrificing completeness? Is there unnecessary verbosity?",
            "code_quality": "Is the generated code clean, efficient, and correct? Does it follow best practices?",
            "visualization": "Is the suggested visualization appropriate for the data and query? Does it effectively communicate insights?",
        }

        for criterion in criteria:
            description = criterion_descriptions.get(
                criterion,
                f"Evaluate the {criterion} of the answer."
            )
            prompt += f"{criterion.upper()}: {description}\n\n"

        prompt += """
**Response Format:**
Provide your evaluation in the following JSON format:

```json
{
  "criteria_scores": {
    "accuracy": {"score": <1-5>, "reasoning": "<explanation>", "issues": ["<issue1>", "<issue2>"]},
    "completeness": {"score": <1-5>, "reasoning": "<explanation>", "issues": []},
    ... (for each criterion)
  },
  "overall_score": <average of all criteria scores>,
  "strengths": ["<strength1>", "<strength2>"],
  "weaknesses": ["<weakness1>", "<weakness2>"],
  "recommendation": "<actionable recommendation for improvement>"
}
```

**Important Guidelines:**
1. Be strict but fair in your evaluation
2. Accuracy is most important - penalize heavily for factual errors
3. If context is provided, check that all claims are grounded in it
4. If ground truth is provided, compare factual accuracy against it
5. Provide specific, actionable feedback
6. Return ONLY the JSON response, no additional text
"""

        return prompt

    def evaluate(
        self,
        query: str,
        answer: str,
        context: Optional[str] = None,
        ground_truth: Optional[str] = None,
        criteria: Optional[List[str]] = None,
    ) -> LLMJudgment:
        """
        Evaluate a single query-answer pair.

        Args:
            query: User query
            answer: Generated answer to evaluate
            context: Optional context used for generation
            ground_truth: Optional ground truth answer
            criteria: Optional list of criteria (uses config default if None)

        Returns:
            LLMJudgment with scores and reasoning
        """
        if self.llm is None:
            logger.error("LLM not initialized, returning default judgment")
            return self._default_judgment(criteria)

        logger.info(f"Evaluating answer for query: {query[:50]}...")

        try:
            # Build prompt
            prompt = self._build_evaluation_prompt(
                query=query,
                answer=answer,
                context=context,
                ground_truth=ground_truth,
                criteria=criteria,
            )

            # Get judgment from LLM
            messages = [
                SystemMessage(content="You are an expert evaluator. Provide objective, structured assessments."),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            response_text = response.content

            # Extract JSON from response
            # Handle markdown code blocks
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()

            # Parse JSON
            judgment_data = json.loads(json_text)

            # Build LLMJudgment object
            criteria_scores = {}
            for criterion_name, criterion_data in judgment_data["criteria_scores"].items():
                criteria_scores[criterion_name] = JudgmentCriterion(
                    name=criterion_name,
                    score=criterion_data["score"],
                    reasoning=criterion_data["reasoning"],
                    issues=criterion_data.get("issues", []),
                )

            judgment = LLMJudgment(
                overall_score=judgment_data["overall_score"],
                criteria_scores=criteria_scores,
                strengths=judgment_data.get("strengths", []),
                weaknesses=judgment_data.get("weaknesses", []),
                recommendation=judgment_data.get("recommendation", ""),
            )

            logger.info(f"Overall score: {judgment.overall_score:.2f}")

            return judgment

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text}")
            return self._default_judgment(criteria)

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return self._default_judgment(criteria)

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> List[LLMJudgment]:
        """
        Evaluate a batch of test cases.

        Args:
            test_cases: List of dicts with keys:
                - query: User query
                - answer: Generated answer
                - context (optional): Context used
                - ground_truth (optional): Ground truth answer
                - criteria (optional): Criteria to evaluate

        Returns:
            List of LLMJudgment, one per test case
        """
        logger.info(f"Evaluating batch of {len(test_cases)} test cases")

        results = []
        for i, test_case in enumerate(test_cases):
            logger.info(f"Evaluating test case {i + 1}/{len(test_cases)}")

            judgment = self.evaluate(
                query=test_case["query"],
                answer=test_case["answer"],
                context=test_case.get("context"),
                ground_truth=test_case.get("ground_truth"),
                criteria=test_case.get("criteria"),
            )

            results.append(judgment)

        return results

    def compare_answers(
        self,
        query: str,
        answer_a: str,
        answer_b: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compare two answers using pairwise evaluation.

        Args:
            query: User query
            answer_a: First answer
            answer_b: Second answer
            context: Optional context

        Returns:
            Dict with comparison results
        """
        logger.info("Performing pairwise comparison")

        judgment_a = self.evaluate(query, answer_a, context)
        judgment_b = self.evaluate(query, answer_b, context)

        winner = "A" if judgment_a.overall_score > judgment_b.overall_score else "B"
        if abs(judgment_a.overall_score - judgment_b.overall_score) < 0.2:
            winner = "Tie"

        return {
            "winner": winner,
            "answer_a_score": judgment_a.overall_score,
            "answer_b_score": judgment_b.overall_score,
            "answer_a_judgment": judgment_a.to_dict(),
            "answer_b_judgment": judgment_b.to_dict(),
        }

    def _default_judgment(self, criteria: Optional[List[str]] = None) -> LLMJudgment:
        """Return default judgment when evaluation fails."""
        criteria = criteria or self.criteria

        criteria_scores = {
            criterion: JudgmentCriterion(
                name=criterion,
                score=0.0,
                reasoning="Evaluation failed",
                issues=["LLM evaluation error"],
            )
            for criterion in criteria
        }

        return LLMJudgment(
            overall_score=0.0,
            criteria_scores=criteria_scores,
            strengths=[],
            weaknesses=["Evaluation failed"],
            recommendation="Unable to evaluate",
        )

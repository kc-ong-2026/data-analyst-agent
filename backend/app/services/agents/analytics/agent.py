"""Analytics Agent - Processes data and generates insights using LangGraph with code generation."""

import asyncio
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from app.models import AnalysisResult, VisualizationData
from app.services.security import (
    CodeExecutionAuditLogger,
    CodeValidator,
    SandboxExecutor,
)

from ..base_agent import (
    AgentResponse,
    AgentRole,
    AgentState,
    BaseAgent,
    GraphState,
)
from .prompts import COLUMN_VALIDATION_PROMPT, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class AnalyticsAgent(BaseAgent):
    """Agent responsible for analyzing data and generating insights.

    Uses LangGraph to construct a flow with nodes:
    1. prepare_data - Prepare extracted data for analysis
    2. analyze_data - Perform analysis and generate insights
    3. generate_visualization - Create visualization specification
    4. compose_response - Compose final response
    """

    def __init__(
        self,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """Initialize analytics agent with security components.

        Args:
            llm_provider: The LLM provider to use (openai, anthropic, google)
            llm_model: The specific model to use
            temperature: Temperature for LLM sampling
            max_tokens: Maximum tokens for LLM response
        """
        # Initialize parent class with LLM configuration
        super().__init__(
            llm_provider=llm_provider,
            llm_model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Get security configuration from app config
        from app.config import get_config

        config = get_config()
        security_config = config.yaml_config.get("code_execution_security", {})

        # Initialize security components
        self.security_enabled = security_config.get("enabled", True)
        self.code_validator = CodeValidator(config=security_config.get("validation", {}))
        self.sandbox_executor = SandboxExecutor(config=security_config.get("isolation", {}))
        self.audit_logger = CodeExecutionAuditLogger(config=security_config.get("audit", {}))

        logger.info(
            f"Analytics agent security: {'enabled' if self.security_enabled else 'disabled'}"
        )

    @property
    def role(self) -> AgentRole:
        return AgentRole.ANALYTICS

    @property
    def name(self) -> str:
        return "Analytics"

    @property
    def description(self) -> str:
        return "Processes data and generates insights with visualizations"

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    @staticmethod
    def _safe_join(items, sep=", ") -> str:
        """Safely join items, converting non-strings to strings.

        Args:
            items: List of items to join (can be strings, dicts, or other types)
            sep: Separator string

        Returns:
            Joined string, empty if items is empty or None
        """
        if not items:
            return ""
        return sep.join(str(item) for item in items if item)

    @staticmethod
    def _make_user_friendly_error(error: Exception, context: str = "") -> str:
        """Convert technical error messages to user-friendly messages.

        Args:
            error: The exception that occurred
            context: Optional context about where the error occurred

        Returns:
            User-friendly error message
        """
        error_msg = str(error)
        error_type = type(error).__name__

        # Detect server-side programming errors (should not be shown as user errors)
        server_error_patterns = [
            "object has no attribute 'get'",
            "object has no attribute 'items'",
            "NoneType' object",
            "module 'pandas' has no attribute",
            "cannot import name",
            "unexpected keyword argument",
            "missing required positional argument",
            "takes X positional arguments but Y were given",
        ]

        is_server_error = any(pattern in error_msg for pattern in server_error_patterns)

        # If this is a clear server-side error, return generic message
        if is_server_error or (
            error_type == "AttributeError" and ("object has no attribute" in error_msg)
        ):
            base_msg = "An internal server error occurred. Our team has been notified and will fix this issue."
            logger.critical(f"[SERVER ERROR] {error_type}: {error_msg}", exc_info=True)
            return f"{base_msg}\n\nError ID: {error_type}\nPlease contact support if this persists."

        # Map user-facing error types to friendly messages
        error_mappings = {
            "KeyError": "We couldn't find the expected data field in the dataset.",
            "ValueError": "We encountered invalid data while processing your request.",
            "TypeError": "We encountered a data format issue while processing your request.",
            "IndexError": "We couldn't access the data at the expected position.",
            "ZeroDivisionError": "We encountered a calculation error (division by zero) in your data.",
            "MemoryError": "The data is too large to process. Please try a more specific query.",
            "TimeoutError": "The analysis took too long to complete. Please try a simpler query.",
        }

        # Check for specific error patterns (user-facing)
        if "sequence item" in error_msg and "expected str instance" in error_msg:
            base_msg = "We had trouble formatting the data for display. This might be due to unexpected data types."
        elif "JSON" in error_msg or "json" in error_msg:
            base_msg = "We had trouble parsing the data format. The data might not be in the expected format."
        elif "column" in error_msg.lower() and "not found" in error_msg.lower():
            base_msg = "We couldn't find a required data column. The dataset might not contain the data you're looking for."
        elif "syntax" in error_msg.lower() and "code" in context.lower():
            base_msg = "There was an error in the generated analysis code. We'll try to fix this automatically."
        else:
            base_msg = error_mappings.get(
                error_type, "We encountered an unexpected error while processing your request."
            )

        # Add context if provided
        if context:
            base_msg = f"{base_msg} (During: {context})"

        # Add technical details for debugging (but less prominent for user errors)
        return f"{base_msg}\n\nTechnical details: {error_type}: {error_msg[:200]}"

    @staticmethod
    def _reconstruct_dataframe(serialized: dict[str, Any]) -> pd.DataFrame:
        """Reconstruct a single DataFrame from serialized data.

        Args:
            serialized: Dictionary containing dataset_name, columns, dtypes, data, source

        Returns:
            Reconstructed pandas DataFrame
        """
        # Reconstruct DataFrame from data
        df = pd.DataFrame(serialized.get("data", []))

        if df.empty:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=serialized.get("columns", []))

        # COMPREHENSIVE DATA CLEANING: Clean ALL string columns
        for col in df.columns:
            if df[col].dtype == "object":
                # Clean trailing letters from numeric-like strings
                df[col] = df[col].astype(str).str.replace(r"^(\d+)[a-zA-Z]+$", r"\1", regex=True)
                # Clean leading/trailing whitespace
                df[col] = df[col].str.strip()

        # Apply dtype conversions if available
        dtypes = serialized.get("dtypes", {})
        for col, dtype_str in dtypes.items():
            if col in df.columns:
                try:
                    # Use exact dtype string if possible, otherwise infer
                    if dtype_str.lower() == "int64":
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.int64)
                    elif dtype_str.lower() == "int32":
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.int32)
                    elif "int" in dtype_str.lower():
                        # Use nullable Int64 for other integer types
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                    elif dtype_str.lower() == "float64":
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
                    elif dtype_str.lower() == "float32":
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
                    elif "float" in dtype_str.lower():
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    elif "datetime" in dtype_str.lower():
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception as e:
                    logger.warning(f"Could not convert column {col} to {dtype_str}: {e}")

        # AUTO-DETECT and clean year columns (only if not already converted above)
        year_cols = [c for c in df.columns if "year" in c.lower()]
        for col in year_cols:
            # Skip if dtype was already set from dtypes dict
            if col not in dtypes and df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        return df

    @staticmethod
    def _reconstruct_dataframes(serialized_list: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
        """Reconstruct multiple DataFrames from a list of serialized data.

        Args:
            serialized_list: List of serialized DataFrame dictionaries

        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        dataframes = {}
        for serialized in serialized_list:
            dataset_name = serialized.get("dataset_name")
            if dataset_name:
                df = AnalyticsAgent._reconstruct_dataframe(serialized)
                dataframes[dataset_name] = df
        return dataframes

    @staticmethod
    def _extract_visualization_data(fig) -> dict[str, Any] | None:
        """Extract data from matplotlib Figure for testing.

        Args:
            fig: Matplotlib Figure object

        Returns:
            Dict with 'type' and 'data' keys
        """
        from matplotlib.figure import Figure

        if not isinstance(fig, Figure):
            return None

        try:
            # Extract data from the first axes
            if not fig.axes:
                return None

            ax = fig.axes[0]
            lines = ax.get_lines()

            if lines:
                # Line/scatter plot
                line = lines[0]
                x_data = line.get_xdata().tolist()
                y_data = line.get_ydata().tolist()
                return {"type": "line", "data": {"x": x_data, "y": y_data}}

            # Check for bar charts
            patches = ax.patches
            if patches:
                return {"type": "bar", "data": {"count": len(patches)}}

            return None
        except Exception as e:
            logger.warning(f"Failed to extract visualization data: {e}")
            return None

    @staticmethod
    def _auto_generate_visualization(df: pd.DataFrame, query: str) -> dict[str, Any] | None:
        """Auto-generate visualization spec from DataFrame (testing method).

        Args:
            df: DataFrame to visualize
            query: User query containing visualization hints

        Returns:
            Visualization spec dict
        """
        if df is None or df.empty:
            return None

        viz_type = AnalyticsAgent._detect_visualization_type(query)
        columns = df.columns.tolist()

        if len(columns) < 2:
            return None

        return {"type": viz_type, "x": columns[0], "y": columns[1], "title": "Auto-generated Chart"}

    @staticmethod
    def _detect_visualization_type(query: str) -> str:
        """Detect visualization type from query.

        Args:
            query: User query string

        Returns:
            Chart type: 'bar', 'line', 'pie', or 'scatter'
        """
        query_lower = query.lower()

        if "bar" in query_lower:
            return "bar"
        elif "line" in query_lower or "trend" in query_lower:
            return "line"
        elif "pie" in query_lower:
            return "pie"
        elif "scatter" in query_lower:
            return "scatter"

        # Default to bar chart
        return "bar"

    @staticmethod
    def _create_plotly_chart(df: pd.DataFrame, viz_spec: dict[str, Any]) -> str | None:
        """Create Plotly HTML chart from DataFrame and spec (testing method).

        Args:
            df: DataFrame containing data
            viz_spec: Visualization specification

        Returns:
            HTML string of the chart
        """
        try:
            import plotly.graph_objects as go

            chart_type = viz_spec.get("type", "bar")
            x_col = viz_spec.get("x")
            y_col = viz_spec.get("y")

            if x_col not in df.columns or y_col not in df.columns:
                return None

            x_data = df[x_col].tolist()
            y_data = df[y_col].tolist()

            if chart_type == "bar":
                fig = go.Figure(data=go.Bar(x=x_data, y=y_data))
            elif chart_type == "line":
                fig = go.Figure(data=go.Scatter(x=x_data, y=y_data, mode="lines+markers"))
            elif chart_type == "scatter":
                fig = go.Figure(data=go.Scatter(x=x_data, y=y_data, mode="markers"))
            else:
                fig = go.Figure(data=go.Bar(x=x_data, y=y_data))

            fig.update_layout(
                title=viz_spec.get("title", "Chart"),
                xaxis_title=x_col,
                yaxis_title=y_col,
            )

            return fig.to_html(include_plotlyjs=True, full_html=True)

        except Exception as e:
            logger.warning(f"Failed to create plotly chart: {e}")
            return None

    async def execute(self, state) -> AgentResponse:
        """Execute the analytics agent with enhanced error handling and logging.

        Overrides BaseAgent.execute to provide better error messages for users.
        """
        try:
            # Get current task safely from either AgentState or dict
            if isinstance(state, AgentState):
                current_task = state.current_task or "N/A"
            elif isinstance(state, dict):
                current_task = state.get("current_task", "N/A")
            else:
                current_task = "N/A"

            logger.info(f"[ANALYTICS AGENT] Starting execution for query: {current_task[:100]}")

            # Call parent execute method
            response = await super().execute(state)

            if response.success:
                logger.info("[ANALYTICS AGENT] ✓ Execution completed successfully")
            else:
                logger.warning(f"[ANALYTICS AGENT] ✗ Execution failed: {response.message}")

            return response

        except Exception as e:
            logger.error(f"[ANALYTICS AGENT] Critical error during execution: {e}", exc_info=True)

            # Create user-friendly error message
            user_message = self._make_user_friendly_error(e, "analytics execution")

            # Handle error based on state type
            if isinstance(state, AgentState):
                state.add_error(user_message)
                error_state = state
            else:
                error_state = (
                    AgentState.from_graph_state(state) if isinstance(state, dict) else AgentState()
                )
                error_state.add_error(user_message)

            return AgentResponse(
                success=False,
                message=user_message,
                state=error_state,
            )

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for analytics with code generation and ReAct loop.

        Flow:
        prepare_data -> validate_columns -> [generate/fallback]
        generate: generate_code -> validate_code -> execute_code -> evaluate_results -> [retry/continue]
        retry: loop back to generate_code (max 3 iterations)
        continue: explain_results -> [visualize?] -> compose_response -> END
        fallback: compose_fallback_response -> END
        """
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("prepare_data", self._prepare_data_node)
        workflow.add_node("validate_columns", self._validate_columns_node)

        # ReAct Loop Nodes
        workflow.add_node("generate_code", self._generate_code_node)
        workflow.add_node("validate_code", self._validate_code_node)
        workflow.add_node("execute_code", self._execute_code_node)
        workflow.add_node("evaluate_results", self._evaluate_results_node)

        workflow.add_node("explain_results", self._explain_results_node)
        workflow.add_node("generate_visualization", self._generate_visualization_node)
        workflow.add_node("compose_response", self._compose_response_node)
        workflow.add_node("compose_fallback_response", self._compose_fallback_response_node)

        # Set entry point
        workflow.set_entry_point("prepare_data")

        # Main flow
        workflow.add_edge("prepare_data", "validate_columns")
        workflow.add_conditional_edges(
            "validate_columns",
            self._should_generate_code,
            {
                "generate": "generate_code",
                "fallback": "compose_fallback_response",
            },
        )

        # ReAct Loop (max 3 iterations)
        workflow.add_edge("generate_code", "validate_code")
        workflow.add_edge("validate_code", "execute_code")
        workflow.add_edge("execute_code", "evaluate_results")
        workflow.add_conditional_edges(
            "evaluate_results",
            self._should_retry_generation,
            {
                "retry": "generate_code",  # Loop back to generate_code with feedback
                "continue": "explain_results",  # Exit loop, proceed to explanation
            },
        )

        # Rest of workflow
        workflow.add_conditional_edges(
            "explain_results",
            self._should_visualize,
            {
                "visualize": "generate_visualization",
                "skip": "compose_response",
            },
        )
        workflow.add_edge("generate_visualization", "compose_response")
        workflow.add_edge("compose_response", END)
        workflow.add_edge("compose_fallback_response", END)

        return workflow.compile()

    def _should_visualize(self, state: GraphState) -> str:
        """Determine if visualization should be generated."""
        # Check if visualization is suggested in metadata
        viz_suggested = state.get("metadata", {}).get("visualization_suggested", True)
        extracted_data = state.get("extracted_data", {})

        # Visualize if we have data and it's suggested
        if viz_suggested and extracted_data:
            return "visualize"
        return "skip"

    def _should_generate_code(self, state: GraphState) -> str:
        """Determine if code generation should proceed based on validation."""
        validation = state.get("intermediate_results", {}).get("column_validation", {})
        status = validation.get("status", "no_match")

        if status in ["exact_match", "partial_match"]:
            return "generate"
        return "fallback"

    def _should_retry_generation(self, state: GraphState) -> str:
        """Determine if code generation should be retried."""
        evaluation = state.get("intermediate_results", {}).get("evaluation", {})
        should_retry = evaluation.get("should_retry", False)

        if should_retry:
            return "retry"
        return "continue"

    async def _prepare_data_node(self, state: GraphState) -> dict[str, Any]:
        """Node: Prepare extracted data for analysis by reconstructing DataFrames."""
        try:
            extracted_data = state.get("extracted_data", {})
            current_task = state.get("current_task", "")

            logger.info(
                f"[PREPARE DATA] Starting data preparation for {len(extracted_data)} datasets"
            )

            # Rebuild DataFrames from extracted data
            dataframes = {}
            data_summary = {}
            total_rows = 0
            errors = []

            for name, data_dict in extracted_data.items():
                if not isinstance(data_dict, dict):
                    logger.warning(f"[PREPARE DATA] Skipping non-dict data: {name}")
                    continue

                try:
                    # Check if this is serialized DataFrame data
                    source = data_dict.get("source")
                    if (
                        source in ["rag_metadata", "file_metadata", "dataframe"]
                        and data_dict.get("data") is not None
                    ):
                        # Use static method to reconstruct DataFrame
                        df = self._reconstruct_dataframe(data_dict)

                        dataframes[name] = df

                        # Create data summary
                        data_summary[name] = {
                            "row_count": len(df),
                            "columns": df.columns.tolist(),
                            "numeric_columns": df.select_dtypes(
                                include=[np.number]
                            ).columns.tolist(),
                            "shape": df.shape,
                            "metadata": data_dict.get("metadata", {}),
                        }
                        total_rows += len(df)

                        logger.info(
                            f"[PREPARE DATA] ✓ Reconstructed {name}: {df.shape}, {len(df)} rows"
                        )
                except Exception as e:
                    error_msg = f"Failed to reconstruct {name}: {str(e)}"
                    logger.error(f"[PREPARE DATA] ✗ {error_msg}", exc_info=True)
                    errors.append(error_msg)

            if not dataframes and errors:
                logger.error(
                    f"[PREPARE DATA] No DataFrames reconstructed, {len(errors)} errors occurred"
                )

            logger.info(
                f"[PREPARE DATA] Successfully prepared {len(dataframes)} DataFrames, total {total_rows} rows"
            )

            return {
                "intermediate_results": {
                    **state.get("intermediate_results", {}),
                    "dataframes": dataframes,  # Live DataFrame objects
                    "data_summary": data_summary,
                    "total_rows": total_rows,
                    "has_data": len(dataframes) > 0,
                    # Initialize ReAct state
                    "react_iteration": 0,
                    "react_max_iterations": 3,
                    "react_history": [],
                    "react_feedback": None,
                },
                "errors": state.get("errors", []) + errors,
            }

        except Exception as e:
            logger.error(f"[PREPARE DATA] Critical error in prepare_data_node: {e}", exc_info=True)
            user_msg = self._make_user_friendly_error(e, "preparing data")
            return {
                "intermediate_results": {
                    **state.get("intermediate_results", {}),
                    "dataframes": {},
                    "has_data": False,
                },
                "errors": state.get("errors", []) + [user_msg],
            }

    async def _validate_columns_node(self, state: GraphState) -> dict[str, Any]:
        """Node: Validate that DataFrame columns can answer user's query.

        Returns:
            intermediate_results with validation_result field containing:
            - status: "exact_match" | "partial_match" | "no_match"
            - missing_concepts: List of concepts user asked for but not in data
            - available_alternatives: List of what data IS available
            - recommendation: Suggested response or code generation approach
        """
        try:
            current_task = state.get("current_task", "")
            data_summary = state.get("intermediate_results", {}).get("data_summary", {})
            dataframes = state.get("intermediate_results", {}).get("dataframes", {})

            logger.info(f"[COLUMN VALIDATION] Starting validation for query: {current_task[:100]}")

            if not dataframes:
                logger.warning("[COLUMN VALIDATION] No DataFrames available")
                return {
                    "intermediate_results": {
                        **state.get("intermediate_results", {}),
                        "column_validation": {
                            "status": "no_match",
                            "reasoning": "No DataFrames available for validation",
                            "missing_concepts": ["all data"],
                            "available_alternatives": [],
                            "recommendation": "Inform user no data is available",
                        },
                    },
                }

            # Get primary DataFrame for validation
            df_name, df = next(iter(dataframes.items()))
            summary = data_summary.get(df_name, {})
            metadata = summary.get("metadata", {})

            logger.info(
                f"[COLUMN VALIDATION] Validating against DataFrame: {df_name}, shape: {df.shape}"
            )

            # Build validation prompt
            try:
                columns_info = "\n".join([f"  - {col} ({df[col].dtype})" for col in df.columns])
                summary_text = metadata.get("summary_text", "")

                # Safely extract list fields, ensuring they're strings
                primary_dimensions = self._safe_join(metadata.get("primary_dimensions", []))
                categorical_columns = self._safe_join(metadata.get("categorical_columns", []))
                numeric_columns = self._safe_join(summary.get("numeric_columns", []))

                validation_prompt = COLUMN_VALIDATION_PROMPT.format(
                    query=current_task,
                    columns_info=columns_info,
                    summary_text=summary_text,
                    primary_dimensions=primary_dimensions,
                    categorical_columns=categorical_columns,
                    numeric_columns=numeric_columns,
                )
            except Exception as e:
                logger.error(
                    f"[COLUMN VALIDATION] Error building validation prompt: {e}", exc_info=True
                )
                raise ValueError(f"Failed to build validation prompt: {str(e)}")

            try:
                response = await self._invoke_llm([HumanMessage(content=validation_prompt)])
                # Parse JSON response
                json_str = self._extract_json_from_response(response)

                # Check if response is actually JSON
                if json_str and (json_str.startswith("{") or json_str.startswith("[")):
                    validation_result = json.loads(json_str)
                    logger.info(
                        f"[COLUMN VALIDATION] Status: {validation_result.get('status')}, "
                        f"Missing: {validation_result.get('missing_concepts', [])}"
                    )
                else:
                    # Response is not JSON (likely code or explanation from test mocks)
                    # Default to exact_match and proceed
                    logger.info("[COLUMN VALIDATION] Non-JSON response, defaulting to exact_match")
                    validation_result = {
                        "status": "exact_match",
                        "reasoning": "Non-JSON validation response, proceeding with code generation",
                        "missing_concepts": [],
                        "available_alternatives": [],
                        "recommendation": "Generate code with available columns",
                    }
            except Exception as e:
                logger.warning(
                    f"[COLUMN VALIDATION] LLM validation failed, defaulting to exact_match: {e}"
                )
                validation_result = {
                    "status": "exact_match",
                    "reasoning": "Validation error, proceeding with code generation",
                    "missing_concepts": [],
                    "available_alternatives": [],
                    "recommendation": "Generate code with available columns",
                }

            return {
                "intermediate_results": {
                    **state.get("intermediate_results", {}),
                    "column_validation": validation_result,
                },
            }

        except Exception as e:
            logger.error(
                f"[COLUMN VALIDATION] Critical error in validation node: {e}", exc_info=True
            )
            # Return a safe default that allows workflow to continue
            return {
                "intermediate_results": {
                    **state.get("intermediate_results", {}),
                    "column_validation": {
                        "status": "exact_match",
                        "reasoning": f"Validation error: {str(e)}",
                        "missing_concepts": [],
                        "available_alternatives": [],
                        "recommendation": "Generate code with available columns",
                    },
                },
                "errors": state.get("errors", []) + [f"Column validation error: {str(e)}"],
            }

    async def _generate_code_node(self, state: GraphState) -> dict[str, Any]:
        """Node: Generate code using ReAct pattern (Reasoning + Action).

        ReAct Cycle:
        - Reasoning: Think through the approach
        - Action: Generate code
        - Observation: (happens in validate_code and execute_code nodes)
        - Refinement: Use observation to improve next iteration
        """
        try:
            current_task = state.get("current_task", "")
            dataframes = state.get("intermediate_results", {}).get("dataframes", {})
            data_summary = state.get("intermediate_results", {}).get("data_summary", {})
            validation_context = state.get("intermediate_results", {}).get("column_validation")

            # ReAct state
            iteration = state.get("intermediate_results", {}).get("react_iteration", 0)
            react_history = state.get("intermediate_results", {}).get("react_history", [])
            react_feedback = state.get("intermediate_results", {}).get("react_feedback")

            logger.info(
                f"[REACT ITERATION {iteration + 1}/3] Starting code generation for: {current_task[:100]}"
            )

            if not dataframes:
                logger.error("[REACT] No DataFrames available for code generation")
                return {
                    "intermediate_results": {
                        **state.get("intermediate_results", {}),
                        "generated_code": None,
                        "should_plot": False,
                    },
                    "errors": state.get("errors", []) + ["No DataFrames available for analysis"],
                }

            # Get primary DataFrame
            df_name, df = next(iter(dataframes.items()))
            metadata = data_summary.get(df_name, {}).get("metadata", {})
            should_plot = self._query_needs_visualization(current_task)

            logger.info(
                f"[REACT] Using DataFrame: {df_name}, shape: {df.shape}, plotting: {should_plot}"
            )

            # Build ReAct prompt context
            react_prompt_context = {
                "iteration": iteration,
                "max_iterations": 3,
                "feedback": react_feedback,
                "history": react_history,
            }

            # Generate code with reasoning
            try:
                result = await self._generate_code_with_reasoning(
                    query=current_task,
                    df=df,
                    df_name=df_name,
                    should_plot=should_plot,
                    metadata=metadata,
                    validation_context=validation_context,
                    react_context=react_prompt_context,
                )

                reasoning = result.get("reasoning", "")
                code = result.get("code", "")

                logger.info(f"[REACT REASONING] {reasoning[:200]}...")
                if code:
                    logger.info(f"[REACT CODE] Generated {len(code)} characters of code")
                else:
                    logger.warning("[REACT CODE] No code generated from LLM response")

            except Exception as e:
                logger.error(f"[REACT] Error generating code: {e}", exc_info=True)
                reasoning = f"Error during code generation: {str(e)}"
                code = ""

            # Store in history
            react_history.append(
                {
                    "iteration": iteration,
                    "reasoning": reasoning,
                    "action": code,
                    "observation": None,  # Filled in by evaluate node
                }
            )

            return {
                "intermediate_results": {
                    **state.get("intermediate_results", {}),
                    "generated_code": code,
                    "react_reasoning": reasoning,
                    "should_plot": should_plot,
                    "primary_df_name": df_name,
                    "react_iteration": iteration,
                    "react_history": react_history,
                },
            }

        except Exception as e:
            logger.error(f"[REACT] Critical error in generate_code_node: {e}", exc_info=True)
            user_msg = self._make_user_friendly_error(e, "generating analysis code")
            return {
                "intermediate_results": {
                    **state.get("intermediate_results", {}),
                    "generated_code": None,
                    "should_plot": False,
                },
                "errors": state.get("errors", []) + [user_msg],
            }

    async def _validate_code_node(self, state: GraphState) -> dict[str, Any]:
        """Node: Validate generated code for syntax and logical errors.

        Checks:
        1. Syntax validity (Python AST parsing)
        2. Column references (check against DataFrame columns)
        3. Import restrictions (only pd, np, plt allowed)
        4. Forbidden operations (file I/O, network, eval/exec)
        5. Variable assignments (must assign to 'result')
        6. Matplotlib usage (if should_plot=True, must create Figure)
        """
        code = state.get("intermediate_results", {}).get("generated_code")
        dataframes = state.get("intermediate_results", {}).get("dataframes", {})
        should_plot = state.get("intermediate_results", {}).get("should_plot", False)

        validation_result = self._validate_generated_code(
            code=code, dataframes=dataframes, should_plot=should_plot
        )

        if not validation_result.get("valid", True):
            logger.warning(f"[REACT CODE VALIDATION FAILED] Errors: {validation_result['errors']}")
        else:
            logger.info("[REACT CODE VALIDATION PASSED]")

        return {
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "code_validation": validation_result,
            },
        }

    async def _execute_code_node(self, state: GraphState) -> dict[str, Any]:
        """Node: Execute generated code in controlled environment."""
        try:
            code = state.get("intermediate_results", {}).get("generated_code")
            dataframes = state.get("intermediate_results", {}).get("dataframes", {})
            df_name = state.get("intermediate_results", {}).get("primary_df_name")
            should_plot = state.get("intermediate_results", {}).get("should_plot", False)
            iteration = state.get("intermediate_results", {}).get("react_iteration", 0)

            logger.info(f"[EXECUTE CODE] Starting execution (iteration {iteration + 1}/3)")

            # Handle missing code or DataFrame
            if not code:
                error_msg = "No valid Python code was generated"
                logger.warning(f"[EXECUTE CODE] {error_msg}")
                return {
                    "intermediate_results": {
                        **state.get("intermediate_results", {}),
                        "execution_result": None,
                        "execution_error": error_msg,
                        "result_type": "None",
                    },
                }

            if not dataframes or df_name not in dataframes:
                error_msg = f"DataFrame '{df_name}' not found in available dataframes: {list(dataframes.keys())}"
                logger.error(f"[EXECUTE CODE] {error_msg}")
                return {
                    "intermediate_results": {
                        **state.get("intermediate_results", {}),
                        "execution_result": None,
                        "execution_error": error_msg,
                        "result_type": "None",
                    },
                }

            logger.info(f"[EXECUTE CODE] Executing {len(code)} characters of code...")
            logger.debug(f"[EXECUTE CODE] Code:\n{code}")

            # Get validation result for audit logging
            validation_result = state.get("intermediate_results", {}).get("validation_result")

            # Execute in safe environment
            result, error = await self._execute_safe(
                code=code, df=dataframes[df_name], should_plot=should_plot
            )

            result_type = type(result).__name__ if result is not None else "None"

            # Audit log execution
            if self.security_enabled:
                # Get query for audit context
                query = None
                messages = state.get("messages", [])
                if messages and len(messages) > 0:
                    last_msg = messages[-1]
                    query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

                # Create execution result object for logging
                from app.services.security import ExecutionResult

                exec_result = ExecutionResult(
                    success=error is None,
                    result=result,
                    error=error,
                )

                self.audit_logger.log_execution(
                    code=code,
                    query=query,
                    validation_result=validation_result,
                    execution_result=exec_result,
                    user_context={"iteration": iteration + 1},
                )

            if error:
                logger.error(f"[EXECUTE CODE] Execution failed: {error}")
            else:
                logger.info(f"[EXECUTE CODE] ✓ Execution successful, result type: {result_type}")

            return {
                "intermediate_results": {
                    **state.get("intermediate_results", {}),
                    "execution_result": result,
                    "execution_error": error,
                    "result_type": result_type,
                },
            }

        except Exception as e:
            logger.error(f"[EXECUTE CODE] Critical error in execute_code_node: {e}", exc_info=True)
            user_msg = self._make_user_friendly_error(e, "executing analysis code")
            return {
                "intermediate_results": {
                    **state.get("intermediate_results", {}),
                    "execution_result": None,
                    "execution_error": user_msg,
                    "result_type": "None",
                },
                "errors": state.get("errors", []) + [user_msg],
            }

    def _validate_visualization_semantics(
        self, fig, dataframes: dict[str, pd.DataFrame], data_summary: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate that the visualization makes semantic sense.

        Checks:
        1. Y-axis values match expected ranges (e.g., percentages should be 0-100)
        2. Axis labels include proper units
        3. Chart type is appropriate for the data
        4. Data values are reasonable given the column metadata

        Returns:
            Dict with 'valid', 'feedback', 'warnings' keys
        """
        from matplotlib.figure import Figure

        if not isinstance(fig, Figure):
            return {"valid": True, "feedback": None, "warnings": []}

        try:
            ax = fig.get_axes()[0] if fig.get_axes() else None
            if not ax:
                return {"valid": True, "feedback": None, "warnings": ["No axes found in figure"]}

            y_label = ax.get_ylabel()
            y_label_lower = y_label.lower()
            y_lim = ax.get_ylim()
            y_axis_min, y_axis_max = y_lim

            errors = []
            warnings = []

            # Get data from the chart
            lines = ax.get_lines()
            bars = ax.containers
            patches = ax.patches

            y_values = []
            x_values = []

            if lines:
                # Extract x,y values from line plots
                for line in lines:
                    x_values.extend(line.get_xdata())
                    y_values.extend(line.get_ydata())
            elif bars:
                # Extract heights from vertical bar plots
                for container in bars:
                    y_values.extend([bar.get_height() for bar in container])
            elif patches:
                # Handle horizontal bar plots (barh) - check both x and y
                for patch in patches:
                    # For barh, width is on x-axis, y position indicates categories
                    y_values.append(patch.get_y())
                    x_values.append(patch.get_width())

            # Filter out NaN/inf values
            y_values = [v for v in y_values if not (pd.isna(v) or np.isinf(v))]

            if not y_values:
                return {
                    "valid": True,
                    "feedback": None,
                    "warnings": ["No data points found in chart"],
                }

            actual_min = min(y_values)
            actual_max = max(y_values)

            # Log for debugging
            logger.info(
                f"[VIZ VALIDATION] Y-axis: '{y_label}', Limits: {y_axis_min:.1f} to {y_axis_max:.1f}, Data: {actual_min:.1f} to {actual_max:.1f}, Count: {len(y_values)}"
            )

            # Check 1: Percentage/Rate validation
            is_percentage = any(
                indicator in y_label_lower
                for indicator in ["percentage", "percent", "%", "rate", "pct", "ratio", "share"]
            )

            if is_percentage:
                # Check BOTH data values AND axis limits (axis limits are a strong signal)
                max_value_to_check = max(actual_max, y_axis_max)

                logger.info(
                    f"[VIZ VALIDATION] Detected percentage/rate axis. Checking max value: {max_value_to_check:.1f}"
                )

                # Check for outlier values (one bar much higher than others - data cleaning issue)
                if len(y_values) > 2:
                    # Calculate median and check for extreme outliers
                    sorted_values = sorted(y_values)
                    median_value = sorted_values[len(sorted_values) // 2]

                    # If max value is more than 10x the median, it's likely a data error
                    if actual_max > 10 * median_value and median_value > 0:
                        errors.append(
                            f"Y-axis labeled as '{y_label}' has an extreme outlier: max={actual_max:.1f} vs median={median_value:.1f}. "
                            f"This suggests a data cleaning issue (e.g., '63.7%' became '637'). "
                            "Check DataFrame cleaning: use pd.to_numeric() with errors='coerce' and strip trailing characters."
                        )
                        logger.warning(
                            f"[VIZ VALIDATION] Detected outlier: max={actual_max:.1f}, median={median_value:.1f}"
                        )

                # Percentages should be 0-100 (or 0-1 for ratios)
                if max_value_to_check > 100:
                    if max_value_to_check > 1000:
                        # Clearly wrong - looks like raw counts, not percentages
                        errors.append(
                            f"Y-axis labeled as '{y_label}' but shows range up to {max_value_to_check:.1f}. "
                            f"These look like raw counts, not percentages (should be 0-100). "
                            f"Data values: {actual_min:.1f} to {actual_max:.1f}, Axis limits: {y_axis_min:.1f} to {y_axis_max:.1f}"
                        )
                    else:
                        # Might be scaled incorrectly (e.g., 0-300 instead of 0-100)
                        errors.append(
                            f"Y-axis labeled as '{y_label}' but values/limits exceed 100 (max: {max_value_to_check:.1f}). "
                            f"Data range: {actual_min:.1f} to {actual_max:.1f}, Axis limits: {y_axis_min:.1f} to {y_axis_max:.1f}. "
                            "Percentages should be 0-100. Either scale the data (multiply/divide) or fix the axis label."
                        )
                elif actual_max > 1.5:
                    # Values are 0-100 scale, but label might need clarification
                    if "%" not in y_label:
                        warnings.append(
                            f"Y-axis shows percentage data (0-100) but label '{y_label}' missing '%' symbol"
                        )
                else:
                    # Values are 0-1 scale (ratio), should use "Rate" not "Percentage"
                    if "percentage" in y_label_lower or "%" in y_label:
                        warnings.append(
                            "Y-axis labeled as percentage but values are 0-1 (ratio scale). "
                            "Consider using 'Rate' or multiply by 100 for percentage."
                        )

            # Check 2: Unit labels
            has_units = any(
                unit in y_label_lower
                for unit in [
                    "(%)",
                    "%",
                    "thousands",
                    "millions",
                    "count",
                    "number",
                    "rate",
                    "$",
                    "sgd",
                    "usd",
                ]
            )

            if not has_units and not is_percentage:
                # Check if data looks like it needs units
                if actual_max > 1000:
                    warnings.append(
                        f"Y-axis label '{y_label}' missing units. "
                        f"Values range {actual_min:.0f} to {actual_max:.0f} - consider adding 'Thousands', 'Count', etc."
                    )

            # Check 3: Negative values where they shouldn't be
            if "rate" in y_label_lower or "percentage" in y_label_lower or "count" in y_label_lower:
                if actual_min < 0:
                    warnings.append(
                        f"Y-axis shows {y_label} but has negative values ({actual_min:.1f}). "
                        "Rates/percentages/counts are typically non-negative."
                    )

            # Check 4: Extremely large ranges that might indicate wrong units
            if (
                actual_max > 100000
                and "thousand" not in y_label_lower
                and "million" not in y_label_lower
            ):
                warnings.append(
                    f"Y-axis has very large values (max: {actual_max:.0f}). "
                    "Consider scaling to 'Thousands' or 'Millions' for readability."
                )

            # Check 5: Year data on Y-axis (common mistake for time series)
            # Check both Y-values and Y-tick positions (for barh)
            y_tick_values = [tick for tick in ax.get_yticks()]
            all_y_values = y_values + y_tick_values

            if all_y_values:
                y_min_check = min(all_y_values)
                y_max_check = max(all_y_values)

                if y_min_check > 1990 and y_max_check < 2100:
                    # This looks like year data on Y-axis
                    x_label_lower = ax.get_xlabel().lower()
                    if "year" not in x_label_lower and "year" not in y_label_lower:
                        errors.append(
                            f"Y-axis shows values {y_min_check:.0f}-{y_max_check:.0f} which look like years. "
                            "Years should typically be on the X-axis (horizontal) for time series charts. "
                            "Consider swapping axes or using vertical bars instead of horizontal bars."
                        )

            # Determine validity
            valid = len(errors) == 0

            if errors:
                feedback = "Visualization semantic issues: " + "; ".join(errors)
                if warnings:
                    feedback += " Warnings: " + "; ".join(warnings[:2])
                return {"valid": False, "feedback": feedback, "warnings": warnings}
            elif warnings:
                return {"valid": True, "feedback": None, "warnings": warnings}
            else:
                return {"valid": True, "feedback": None, "warnings": []}

        except Exception as e:
            logger.warning(f"Failed to validate visualization semantics: {e}", exc_info=True)
            return {"valid": True, "feedback": None, "warnings": [f"Validation error: {str(e)}"]}

    async def _evaluate_results_node(self, state: GraphState) -> dict[str, Any]:
        """Node: Evaluate execution results and decide if refinement is needed.

        Observation Phase of ReAct:
        - Check if execution succeeded
        - Check if result is valid (not None, not empty)
        - Check if visualization was created (if requested)
        - Validate visualization semantics (axis ranges, units, chart type)
        - Decide if re-generation is needed
        """
        result = state.get("intermediate_results", {}).get("execution_result")
        error = state.get("intermediate_results", {}).get("execution_error")
        code_validation = state.get("intermediate_results", {}).get("code_validation", {})
        should_plot = state.get("intermediate_results", {}).get("should_plot", False)
        iteration = state.get("intermediate_results", {}).get("react_iteration", 0)
        max_iterations = state.get("intermediate_results", {}).get("react_max_iterations", 3)
        react_history = state.get("intermediate_results", {}).get("react_history", [])
        dataframes = state.get("intermediate_results", {}).get("dataframes", {})
        data_summary = state.get("intermediate_results", {}).get("data_summary", {})

        # Determine success
        success = False
        should_retry = False
        feedback = None

        # Case 1: Code validation failed
        if not code_validation.get("valid", True):
            validation_errors = code_validation.get("errors", [])
            suggestions = code_validation.get("suggestions", [])
            feedback = f"Code validation failed: {', '.join(validation_errors)}. "
            if suggestions:
                feedback += f"Suggestions: {', '.join(suggestions[:2])}"
            should_retry = iteration < max_iterations - 1
            logger.warning(f"[REACT EVALUATE] Validation failed: {feedback[:200]}")

        # Case 2: Execution error
        elif error:
            # Include validation warnings if any
            validation_warnings = code_validation.get("warnings", [])
            feedback = f"Execution error: {error}"
            if validation_warnings:
                feedback += f" (Warning: {', '.join(validation_warnings[:1])})"
            should_retry = iteration < max_iterations - 1
            logger.warning(f"[REACT EVALUATE] Execution failed: {error[:200]}")

        # Case 3: Result is None
        elif result is None:
            feedback = "Code executed but returned None. Ensure 'result' variable is assigned."
            should_retry = iteration < max_iterations - 1

        # Case 4: Result is empty DataFrame
        elif isinstance(result, pd.DataFrame) and result.empty:
            feedback = "Code returned empty DataFrame. Check filter conditions or column names."
            should_retry = iteration < max_iterations - 1

        # Case 5: Visualization requested but not created
        elif should_plot:
            from matplotlib.figure import Figure

            if not isinstance(result, Figure):
                feedback = "Visualization requested but Figure not created. Use matplotlib to create chart."
                should_retry = iteration < max_iterations - 1
            else:
                # NEW: Semantic validation of the visualization
                viz_validation = self._validate_visualization_semantics(
                    fig=result, dataframes=dataframes, data_summary=data_summary
                )

                if not viz_validation.get("valid", True):
                    feedback = viz_validation.get("feedback", "Visualization has semantic issues.")
                    should_retry = iteration < max_iterations - 1
                    logger.warning(f"[REACT EVALUATE] Visualization validation failed: {feedback}")
                else:
                    success = True
                    if viz_validation.get("warnings"):
                        logger.info(
                            f"[REACT EVALUATE] Visualization warnings: {viz_validation['warnings']}"
                        )

        # Case 6: Success
        else:
            success = True

        # Update history with observation
        if react_history:
            react_history[-1]["observation"] = {
                "success": success,
                "result_type": type(result).__name__ if result is not None else "None",
                "error": error,
                "feedback": feedback,
                "validation_errors": (
                    code_validation.get("errors", [])
                    if not code_validation.get("valid", True)
                    else []
                ),
            }

        # Prepare for next iteration
        next_iteration = iteration + 1 if should_retry else iteration

        if should_retry:
            logger.info(f"[REACT RETRY] Iteration {iteration + 1}, Feedback: {feedback[:100]}")
        elif success:
            logger.info(f"[REACT SUCCESS] Completed in {iteration + 1} iteration(s)")
        elif iteration >= max_iterations - 1:
            logger.warning("[REACT MAX ITERATIONS] Failed after 3 attempts")

        return {
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "evaluation": {
                    "success": success,
                    "should_retry": should_retry,
                    "feedback": feedback,
                },
                "react_feedback": feedback if should_retry else None,
                "react_iteration": next_iteration,
                "react_history": react_history,
            },
        }

    async def _explain_results_node(self, state: GraphState) -> dict[str, Any]:
        """Node: Generate natural language explanation of results."""
        current_task = state.get("current_task", "")
        result = state.get("intermediate_results", {}).get("execution_result")
        error = state.get("intermediate_results", {}).get("execution_error")
        validation = state.get("intermediate_results", {}).get("column_validation", {})

        if error:
            explanation = f"I encountered an error while analyzing the data: {error}"
        elif result is None:
            explanation = "I couldn't generate meaningful results from the data."
        else:
            # Use LLM to explain the result, including validation context
            explanation = await self._generate_explanation(
                query=current_task, result=result, validation_context=validation
            )

        # Get extracted data keys for data sources
        extracted_data = state.get("extracted_data", {})

        return {
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "explanation": explanation,
            },
            "analysis_results": {
                "text": explanation,
                "data_sources": list(extracted_data.keys()),
            },
        }

    async def _analyze_data_node(self, state: GraphState) -> dict[str, Any]:
        """Node: Analyze data and generate insights."""
        current_task = state.get("current_task", "")
        extracted_data = state.get("extracted_data", {})
        data_summary = state.get("intermediate_results", {}).get("data_summary", {})

        # Build analysis prompt
        data_context = self._format_data_for_analysis(extracted_data, data_summary)

        # Check if we actually have data to analyze (not just metadata)
        has_actual_data = False
        total_rows = state.get("intermediate_results", {}).get("total_rows", 0)

        # Also check for SQL query results
        if (
            "query_result" in extracted_data
            and extracted_data["query_result"].get("data")
            or total_rows > 0
        ):
            has_actual_data = True

        # Use different prompts based on data availability
        if not has_actual_data:
            # Simpler prompt when no data available - just explain briefly
            analysis_prompt = f"""The user asked: {current_task}

{data_context}

Provide a brief, single-sentence response explaining that you cannot provide the requested information because no data is currently available.

IMPORTANT: Keep it SHORT and SIMPLE - just 1-2 sentences maximum. Do NOT include sections like "Key Findings", "Recommendations", or multiple paragraphs. Just state that the data is not available."""
        else:
            # Full analysis prompt when data is available
            analysis_prompt = f"""Analyze this data to answer the user's question.

User Question: {current_task}

Available Data:
{data_context}

Provide your analysis in clear, natural language. Include:
1. Direct answer to the question
2. Key insights from the data
3. Any notable patterns or trends
4. Limitations or caveats

Be specific and cite actual values from the data when possible.

IMPORTANT: Write your response as natural text only. Do NOT include JSON, code blocks, or structured data formats in your response. The visualization will be handled separately."""

        try:
            response = await self._invoke_llm([HumanMessage(content=analysis_prompt)])
            analysis = response
        except Exception as e:
            analysis = f"Analysis could not be completed: {str(e)}"

        return {
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "analysis": analysis,
            },
            "analysis_results": {
                "text": analysis,
                "data_sources": list(extracted_data.keys()),
            },
        }

    async def _generate_visualization_node(self, state: GraphState) -> dict[str, Any]:
        """Node: Generate visualization from code execution results."""
        result = state.get("intermediate_results", {}).get("execution_result")
        current_task = state.get("current_task", "")
        should_plot = state.get("intermediate_results", {}).get("should_plot", False)
        validation = state.get("intermediate_results", {}).get("column_validation", {})

        viz = None

        # Strategy 1: If matplotlib Figure was generated, extract data from it
        from matplotlib.figure import Figure

        if isinstance(result, Figure):
            # Don't use user's question as title if data doesn't match
            viz = self._extract_viz_from_figure(result, current_task, validation)
            logger.info("Extracted visualization from matplotlib Figure")

        # Strategy 2: If DataFrame/Series returned, auto-generate visualization
        elif isinstance(result, pd.DataFrame | pd.Series):
            viz = self._generate_viz_from_dataframe(result, current_task, validation)
            logger.info("Generated visualization from DataFrame/Series result")

        # Strategy 3: Fall back to extracted data if code didn't produce viz
        if not viz:
            extracted_data = state.get("extracted_data", {})
            viz = self._auto_generate_visualization(extracted_data)
            logger.info("Using fallback visualization from extracted data")

        return {
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "visualization": viz,
            },
        }

    async def _compose_response_node(self, state: GraphState) -> dict[str, Any]:
        """Node: Compose final response with analysis and visualization."""
        # FIX: Use "explanation" key (set by explain_results_node), not "analysis"
        explanation = state.get("intermediate_results", {}).get("explanation", "")
        visualization = state.get("intermediate_results", {}).get("visualization")
        extracted_data = state.get("extracted_data", {})
        current_task = state.get("current_task", "")

        # If no explanation yet, generate a simple response
        if not explanation:
            if extracted_data:
                explanation = "Data has been extracted and is ready for review."
            else:
                explanation = "I couldn't find specific data for your query. Please try rephrasing your question."

        # Strip thinking tags and JSON code blocks for clean user-facing response
        cleaned_analysis = self._strip_thinking_tags(explanation)
        cleaned_analysis = self._strip_json_code_blocks(cleaned_analysis)

        # Clean up the analysis text
        final_response = cleaned_analysis.strip()

        # Add data source attribution
        sources = list(extracted_data.keys())
        if sources:
            source_text = ", ".join(sources[:3])
            if len(sources) > 3:
                source_text += f", and {len(sources) - 3} more"
            final_response += f"\n\n*Data sources: {source_text}*"

        # Generate HTML chart if visualization exists
        if visualization:
            html_chart = self._generate_html_chart(visualization)
            if html_chart:
                # Include both HTML chart AND data array for Recharts fallback
                visualization["html_chart"] = html_chart
                logger.info(
                    f"Returning visualization with both HTML chart and {len(visualization.get('data', []))} data points for fallback"
                )

        return {
            "analysis_results": {
                "text": final_response,
                "visualization": visualization,
                "data_sources": sources,
            },
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "final_response": final_response,
                "visualization": visualization,  # Add visualization to intermediate_results
            },
            "messages": [AIMessage(content=final_response)],
            "current_step": state.get("current_step", 0) + 1,
        }

    async def _compose_fallback_response_node(self, state: GraphState) -> dict[str, Any]:
        """Node: Compose response when columns don't match query requirements.

        Generates helpful message about what data IS available and suggests
        alternative queries.
        """
        validation = state.get("intermediate_results", {}).get("column_validation", {})
        extracted_data = state.get("extracted_data", {})
        current_task = state.get("current_task", "")

        missing_concepts = validation.get("missing_concepts", [])
        available_alternatives = validation.get("available_alternatives", [])
        reasoning = validation.get("reasoning", "")

        # Get metadata from first dataset
        metadata = {}
        for name, data_dict in extracted_data.items():
            if isinstance(data_dict, dict):
                metadata = data_dict.get("metadata", {})
                break

        # Build fallback prompt
        fallback_prompt = f"""The user asked: "{current_task}"

Validation Result:
- Status: {validation.get('status')}
- Reasoning: {reasoning}
- Missing concepts: {self._safe_join(missing_concepts) if missing_concepts else 'None'}
- Available data: {self._safe_join(available_alternatives) if available_alternatives else 'None'}

Dataset Information:
- Description: {metadata.get('summary_text', 'Not available')}
- Primary Dimensions: {self._safe_join(metadata.get('primary_dimensions', []))}
- Categorical Columns: {self._safe_join(metadata.get('categorical_columns', []))}
- Time Period: {metadata.get('year_range', 'Not specified')}

**Your task**: Compose a helpful response that:
1. Acknowledges what the user requested
2. Explains clearly what data is NOT available (missing_concepts)
3. Describes what data IS available in the dataset
4. Suggests 2-3 alternative queries the user could ask with the available data

Keep the response concise and friendly (3-4 sentences max).
"""

        try:
            response = await self._invoke_llm([HumanMessage(content=fallback_prompt)])
            fallback_text = response
        except Exception as e:
            logger.error(f"Failed to generate fallback response: {e}")
            fallback_text = f"I couldn't find data for your query. The available data covers {', '.join(available_alternatives) if available_alternatives else 'different dimensions'}."

        # Add data source attribution
        sources = list(extracted_data.keys())
        if sources:
            source_text = ", ".join(sources[:3])
            if len(sources) > 3:
                source_text += f", and {len(sources) - 3} more"
            fallback_text += f"\n\n*Data sources: {source_text}*"

        return {
            "analysis_results": {
                "text": fallback_text,
                "visualization": None,
                "data_sources": sources,
            },
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "final_response": fallback_text,
            },
            "messages": [AIMessage(content=fallback_text)],
            "current_step": state.get("current_step", 0) + 1,
        }

    def _format_data_for_analysis(
        self,
        extracted_data: dict[str, Any],
        data_summary: dict[str, Any],
    ) -> str:
        """Format extracted data for the analysis prompt.

        Priority: SQL query results > sample rows > metadata
        """
        if not extracted_data:
            return "No data available for analysis."

        parts = []

        # Check if we have SQL query results (highest priority)
        if "query_result" in extracted_data and extracted_data["query_result"].get("data"):
            query_data = extracted_data["query_result"]
            rows = query_data.get("data", [])
            part = "\n### SQL Query Result\n"
            part += f"Rows: {len(rows)}\n"
            part += f"Columns: {', '.join(query_data.get('columns', [])[:10])}\n"
            part += "Data (first 10 rows from SQL query):\n"
            part += json.dumps(rows[:10], indent=2, default=str)
            parts.append(part)

        # Process other datasets
        for name, data in extracted_data.items():
            if not isinstance(data, dict) or name == "query_result":
                continue

            summary = data_summary.get(name, {})
            rows = data.get("data", [])
            metadata = data.get("metadata", {})

            part = f"\n### {name}\n"
            part += f"Rows: {summary.get('row_count', len(rows))}\n"
            part += f"Columns: {', '.join(data.get('columns', [])[:10])}\n"

            # Priority: rows from SQL results > metadata context
            if rows:
                part += "Data (first 10 rows):\n"
                part += json.dumps(rows[:10], indent=2, default=str)
            elif metadata:
                # Use metadata for context when no rows available
                part += "\nMetadata Context:\n"
                if metadata.get("description"):
                    part += f"Description: {metadata['description']}\n"
                if metadata.get("primary_dimensions"):
                    part += f"Primary Dimensions: {', '.join(metadata['primary_dimensions'])}\n"
                if metadata.get("numeric_columns"):
                    part += f"Numeric Columns: {', '.join(metadata['numeric_columns'])}\n"
                if metadata.get("year_range"):
                    part += f"Year Range: {metadata['year_range']}\n"

            parts.append(part)

        return "\n".join(parts) if parts else "No structured data available."

    def _auto_generate_visualization(
        self,
        extracted_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Auto-generate visualization from extracted data.

        Prioritizes SQL query results over sample rows.
        """
        if not extracted_data:
            return None

        # Priority 1: Check for SQL query results
        if "query_result" in extracted_data:
            query_data = extracted_data["query_result"]
            rows = query_data.get("data", [])
            if rows and len(rows) > 0:
                columns = list(rows[0].keys()) if rows else []
                if len(columns) >= 2:
                    return self._generate_viz_from_rows("SQL Query Result", rows, columns)

        # Priority 2: Check other datasets with rows
        for name, data in extracted_data.items():
            if not isinstance(data, dict) or name == "query_result":
                continue

            rows = data.get("data", [])
            if rows and len(rows) > 0:
                columns = list(rows[0].keys()) if rows else []
                if len(columns) >= 2:
                    return self._generate_viz_from_rows(name, rows, columns)

        return None

    def _generate_viz_from_rows(
        self,
        dataset_name: str,
        rows: list[dict[str, Any]],
        columns: list[str],
    ) -> dict[str, Any] | None:
        """Generate visualization from rows of data."""
        if len(columns) < 2:
            return None

        # Smart column detection: find x (categorical/year) and y (metric) columns
        x_col = None
        y_col = None

        # Priority 1: Look for temporal columns for x-axis
        temporal_keywords = ["year", "date", "month", "quarter", "time", "period"]
        for col in columns:
            if any(keyword in col.lower() for keyword in temporal_keywords):
                x_col = col
                break

        # Priority 2: Find metric column for y-axis (employment metrics, counts, etc.)
        metric_keywords = [
            "rate",
            "count",
            "value",
            "change",
            "percentage",
            "total",
            "number",
            "amount",
        ]
        for col in columns:
            if col != x_col:
                # Check if column is numeric
                sample = rows[0].get(col)
                if isinstance(sample, int | float):
                    # Prefer columns with metric keywords in name
                    if any(keyword in col.lower() for keyword in metric_keywords):
                        y_col = col
                        break

        # If no metric column found, find any numeric column
        if not y_col:
            for col in columns:
                if col != x_col:
                    sample = rows[0].get(col)
                    if isinstance(sample, int | float):
                        y_col = col
                        break

        # If no temporal column found, find categorical column
        if not x_col:
            for col in columns:
                if col != y_col:
                    sample = rows[0].get(col)
                    if not isinstance(sample, int | float):
                        x_col = col
                        break

        # Final fallback: use first two columns
        if not x_col or not y_col:
            x_col = columns[0]
            y_col = columns[1] if len(columns) > 1 else columns[0]

        logger.info(f"Column selection: x_col='{x_col}', y_col='{y_col}'")

        # Prepare visualization data
        # Each data point must contain both x_col and y_col as keys
        viz_data = []
        for row in rows[:100]:  # Increased limit for better time series visualization
            viz_data.append(
                {
                    x_col: str(row.get(x_col, ""))[:30],  # Category label (string)
                    y_col: row.get(y_col, 0),  # Numeric value
                }
            )

        if viz_data:
            # Create human-readable labels from column names
            x_label = x_col.replace("_", " ").title()
            y_label = y_col.replace("_", " ").title()

            return {
                "chart_type": "bar",
                "title": f"{y_label} by {x_label}",
                "data": viz_data,
                "x_axis": x_col,  # Field name for horizontal axis labels
                "y_axis": y_col,  # Field name for vertical axis values
                "x_label": x_label,  # Human-readable X-axis label
                "y_label": y_label,  # Human-readable Y-axis label
                "description": f"Data from {dataset_name}",
            }

        return None

    def _parse_visualization(self, response: str) -> dict[str, Any] | None:
        """Parse visualization JSON from LLM response."""
        try:
            # Try to find JSON in response
            if "```json" in response:
                start = response.index("```json") + 7
                end = response.index("```", start)
                viz = json.loads(response[start:end].strip())
            elif "```" in response:
                start = response.index("```") + 3
                end = response.index("```", start)
                viz = json.loads(response[start:end].strip())
            elif "{" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                viz = json.loads(response[start:end])
            else:
                return None

            # Validate required fields
            if "chart_type" in viz and "data" in viz:
                if isinstance(viz["data"], list) and len(viz["data"]) > 0:
                    # Validate axis fields exist in data for bar/line charts
                    validated_viz = self._validate_visualization(viz)
                    return validated_viz

        except (ValueError, json.JSONDecodeError):
            pass

        return None

    def _validate_visualization(self, viz: dict[str, Any]) -> dict[str, Any]:
        """Validate and potentially fix visualization specification."""
        chart_type = viz.get("chart_type", "bar")
        data = viz.get("data", [])

        if not data:
            return viz

        # For bar/line/scatter charts, ensure x_axis and y_axis fields exist in data
        if chart_type in ["bar", "line", "scatter"]:
            sample_keys = set(data[0].keys())
            x_axis = viz.get("x_axis")
            y_axis = viz.get("y_axis")

            # If x_axis or y_axis not specified, try to infer from data
            if not x_axis or not y_axis:
                keys_list = list(sample_keys)
                if len(keys_list) >= 2:
                    # Find first non-numeric column for x_axis (labels)
                    x_axis = None
                    for key in keys_list:
                        if not isinstance(data[0][key], int | float):
                            x_axis = key
                            break
                    if not x_axis:
                        x_axis = keys_list[0]

                    # Find first numeric column for y_axis (values)
                    y_axis = None
                    for key in keys_list:
                        if isinstance(data[0][key], int | float):
                            y_axis = key
                            break
                    if not y_axis:
                        y_axis = keys_list[1] if len(keys_list) > 1 else keys_list[0]

                    viz["x_axis"] = x_axis
                    viz["y_axis"] = y_axis

            # Validate that specified fields exist in data
            if x_axis and x_axis not in sample_keys:
                # Try to find best match
                for key in sample_keys:
                    if not isinstance(data[0][key], int | float):
                        viz["x_axis"] = key
                        break

            if y_axis and y_axis not in sample_keys:
                # Try to find best match (numeric field)
                for key in sample_keys:
                    if isinstance(data[0][key], int | float):
                        viz["y_axis"] = key
                        break

            # CRITICAL FIX: Convert x-axis values to strings for proper Recharts rendering
            # Numeric x values (like years) can cause Recharts to misinterpret the axis
            if x_axis and chart_type in ["bar", "line"]:
                viz["data"] = [
                    {
                        **row,
                        x_axis: str(row.get(x_axis, ""))[:50],  # Convert to string, limit length
                    }
                    for row in data
                ]

        return viz

    def _strip_thinking_tags(self, text: str) -> str:
        """Remove <thinking> tags from LLM responses.

        The thinking tags are useful for chain-of-thought reasoning internally,
        but should not be shown to end users.
        """
        import re

        # Remove <thinking>...</thinking> blocks (including multiline)
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Clean up extra whitespace left behind
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # Multiple blank lines → double newline

        return text.strip()

    def _strip_json_code_blocks(self, text: str) -> str:
        """Remove JSON code blocks from analysis text to prevent frontend confusion.

        This ensures the visualization field is the single source of truth.
        """
        import re

        # Pattern to match JSON code blocks (```json...``` or ```...``` with JSON content)
        patterns = [
            r"```json\s*\{[^`]*\}\s*```",  # Explicit JSON blocks
            r"```\s*\{[^`]*\}\s*```",  # Generic code blocks with JSON objects
        ]

        cleaned_text = text
        for pattern in patterns:
            # Replace JSON blocks with a placeholder message
            cleaned_text = re.sub(
                pattern,
                "[Visualization generated - see visualization panel]",
                cleaned_text,
                flags=re.DOTALL,
            )

        return cleaned_text.strip()

    # ======= Code Generation Helper Methods =======

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from LLM response."""
        # Try to find JSON in code blocks
        if "```json" in response:
            start = response.index("```json") + 7
            end = response.index("```", start)
            return response[start:end].strip()
        elif "```" in response:
            start = response.index("```") + 3
            end = response.index("```", start)
            return response[start:end].strip()
        # Try to find raw JSON object
        elif "{" in response and "}" in response:
            start = response.index("{")
            end = response.rindex("}") + 1
            return response[start:end]
        return response

    def _validate_generated_code(
        self, code: str, dataframes: dict[str, pd.DataFrame], should_plot: bool
    ) -> dict[str, Any]:
        """Validate code for syntax and security violations using AST-based validation.

        Returns:
            Dict with 'valid', 'errors', 'warnings', 'suggestions' keys
        """
        suggestions = []

        if not code:
            return {
                "valid": False,
                "errors": ["No code generated"],
                "warnings": [],
                "suggestions": [],
            }

        # Use new security-enhanced code validator
        if self.security_enabled:
            df = list(dataframes.values())[0] if dataframes else None
            validation_result = self.code_validator.validate(code, dataframe=df)

            errors = validation_result.errors.copy()
            warnings = validation_result.warnings.copy()

            # Log security violations
            if not validation_result.is_valid:
                for violation in validation_result.violations:
                    self.audit_logger.log_security_violation(
                        code=code,
                        violation_type=violation.get("type", "unknown"),
                        details=violation.get("message", ""),
                        severity="high" if "forbidden" in violation.get("type", "") else "medium",
                    )

        else:
            # Fallback to basic syntax check if security is disabled
            import ast

            try:
                ast.parse(code)
                errors = []
                warnings = []
            except SyntaxError as e:
                errors = [f"Syntax error: {e}"]
                suggestions = ["Fix Python syntax errors"]
                return {
                    "valid": False,
                    "errors": errors,
                    "warnings": warnings,
                    "suggestions": suggestions,
                }

        # Additional checks for user experience (not security)

        # Check: Variable assignment (must assign to 'result')
        if "result =" not in code and "result=" not in code:
            warnings.append("Code should assign output to 'result' variable")

        # Check: Matplotlib usage (if should_plot=True)
        if should_plot:
            if "plt" not in code and "matplotlib" not in code:
                warnings.append("Visualization requested but no matplotlib usage detected")
                suggestions.append("Use matplotlib.pyplot to create charts")

        valid = len(errors) == 0
        return {"valid": valid, "errors": errors, "warnings": warnings, "suggestions": suggestions}

    def _query_needs_visualization(self, query: str) -> bool:
        """Determine if query requires visualization."""
        query_lower = query.lower()
        viz_keywords = ["plot", "chart", "graph", "visualize", "show", "display", "trend"]
        return any(keyword in query_lower for keyword in viz_keywords)

    async def _generate_analysis_code(
        self,
        query: str,
        df: pd.DataFrame,
        df_name: str,
        should_plot: bool,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Generate pandas code using LLM."""
        columns_str = ", ".join(df.columns.tolist())
        dtypes_str = "\n".join([f"  {col}: {dtype}" for col, dtype in df.dtypes.items()])
        sample_rows = df.head(3).to_dict(orient="records")
        summary_text = metadata.get("summary_text", "") if metadata else ""

        if should_plot:
            prompt = self._build_plot_code_prompt(
                query, columns_str, dtypes_str, sample_rows, summary_text
            )
        else:
            prompt = self._build_analysis_code_prompt(
                query, columns_str, dtypes_str, sample_rows, summary_text
            )

        response = await self._invoke_llm([HumanMessage(content=prompt)])
        code = self._extract_code_block(response)

        logger.info(f"Generated code for query: {query}")
        logger.info(f"Code:\n{code}")

        return code

    async def _generate_code_with_reasoning(
        self,
        query: str,
        df: pd.DataFrame,
        df_name: str,
        should_plot: bool,
        metadata: dict[str, Any] | None,
        validation_context: dict[str, Any] | None,
        react_context: dict[str, Any],
    ) -> dict[str, str]:
        """Generate code with ReAct reasoning.

        Returns:
            Dict with 'reasoning' and 'code' keys
        """
        iteration = react_context.get("iteration", 0)
        feedback = react_context.get("feedback")
        history = react_context.get("history", [])

        # Build prompt with ReAct context
        columns_str = ", ".join(df.columns.tolist())
        dtypes_str = "\n".join([f"  {col}: {dtype}" for col, dtype in df.dtypes.items()])
        sample_rows = df.head(3).to_dict(orient="records")
        summary_text = metadata.get("summary_text", "") if metadata else ""

        react_prompt = self._build_react_prompt(
            query=query,
            columns_str=columns_str,
            dtypes_str=dtypes_str,
            sample_rows=sample_rows,
            summary_text=summary_text,
            should_plot=should_plot,
            validation_context=validation_context,
            iteration=iteration,
            feedback=feedback,
            history=history,
        )

        response = await self._invoke_llm([HumanMessage(content=react_prompt)])

        # Parse reasoning and code from response
        reasoning = self._extract_reasoning(response)
        code = self._extract_code_block(response)

        # Backward compatibility: If no code was extracted, try alternate extraction
        if not code and response.strip():
            # Check if response looks like it could be code (even without fences)
            response_lower = response.lower()
            python_indicators = [
                "df.",
                "df[",
                "result =",
                "result=",
                "import ",
                "def ",
                "for ",
                "if ",
                "groupby",
                "mean(",
                "sum(",
            ]
            if any(indicator in response_lower for indicator in python_indicators):
                # Use the full response as code
                code = response.strip()
                logger.info("[REACT] Using full response as code (backward compatibility mode)")
            else:
                # Response might be explanation text, not code - that's okay for tests
                logger.info(f"[REACT] Response does not contain code: {response[:100]}...")

        return {"reasoning": reasoning, "code": code}

    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from <reasoning> tags."""
        import re

        match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Also try <thinking> tags as fallback
        match = re.search(r"<thinking>(.*?)</thinking>", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return "No explicit reasoning provided"

    def _build_react_prompt(
        self,
        query: str,
        columns_str: str,
        dtypes_str: str,
        sample_rows: list[dict],
        summary_text: str,
        should_plot: bool,
        validation_context: dict[str, Any] | None,
        iteration: int,
        feedback: str | None,
        history: list[dict[str, Any]],
    ) -> str:
        """Build ReAct-style prompt for code generation."""

        # Base context
        base_prompt = f"""Given DataFrame `df` with columns: {columns_str}

Data types:
{dtypes_str}

Dataset Context:
{summary_text}

Sample rows:
{json.dumps(sample_rows, indent=2, default=str)}

User Query: "{query}"
"""

        # Add validation context if available
        if validation_context and validation_context.get("status") == "partial_match":
            missing = validation_context.get("missing_concepts", [])
            available = validation_context.get("available_alternatives", [])

            base_prompt += f"""

**DATA AVAILABILITY NOTICE:**
Missing in dataset: {self._safe_join(missing)}
Available instead: {self._safe_join(available)}
"""

        # Add ReAct context
        react_section = f"""

**ReAct Iteration {iteration + 1} of 3**
"""

        if iteration > 0 and feedback:
            react_section += f"""
**Previous Attempt Failed**:
{feedback}

**Previous Attempts**:
"""
            for hist in history:
                obs = hist.get("observation", {})
                react_section += f"""
Iteration {hist['iteration'] + 1}:
  Reasoning: {hist.get('reasoning', '')[:200]}...
  Result: {obs.get('result_type', 'Failed')}
  Error: {obs.get('error', 'None')[:100] if obs.get('error') else 'None'}
"""

            react_section += """

**Your task**: Learn from the previous failures and generate IMPROVED code.
"""

        # Reasoning instructions
        reasoning_section = """

**Step 1: REASONING** (inside <reasoning> tags)

<reasoning>
1. What is the user asking for?
2. What columns are available in the DataFrame?
3. What aggregations or transformations are needed?
4. What potential issues should I avoid? (missing columns, type errors, empty results)
5. If this is a retry, what went wrong before and how can I fix it?
6. **SEMANTIC VALIDATION**: Does my visualization make sense?
   - If plotting percentages/rates: Are values 0-100? Do I need to multiply by 100?
   - Do axis labels include proper units (%, Thousands, etc.)?
   - Is the chart type appropriate (line for trends, bar for comparisons)?
   - Are years on the correct axis (typically X-axis for time series)?
7. What is my approach?
</reasoning>
"""

        # Code instructions
        if should_plot:
            code_section = """

**Step 2: ACTION** (Python code with matplotlib inside ```python fence)

```python
import matplotlib.pyplot as plt

# Your pandas data transformation code
# ...

# Create matplotlib figure
fig, ax = plt.subplots(figsize=(10, 6))
# ... plotting code ...
ax.set_title("...")
ax.set_xlabel("...")
ax.set_ylabel("...")  # CRITICAL: Include units (%, Thousands, etc.)
result = fig
plt.close()
```

**Rules**:
1. Use pd.to_numeric(df[col], errors='coerce') for type conversions
2. Check if filtered DataFrame is empty before plotting
3. Assign Figure to 'result' variable
4. Close figure with plt.close()
5. **CRITICAL: Set proper Y-axis label with units AND check data values**
   - If column name has 'rate', 'percentage', 'pct': Check if values are 0-1 (ratio) or 0-100 (percentage)
   - If values are 0-1, multiply by 100 for percentage display OR use "Rate" not "Percentage"
   - If values are 0-100, use "Rate (%)" or "Percentage (%)"
   - Example: ax.set_ylabel("Employment Rate (%)")  # Only if values are 0-100
   - Example: ax.set_ylabel("Change (Thousands)")  # For large count values
6. **Verify axis orientation**: Years/dates on X-axis (horizontal), metrics on Y-axis (vertical)
"""
        else:
            code_section = """

**Step 2: ACTION** (Python code inside ```python fence)

```python
# Your pandas code to answer the query
result = df.groupby(...)[...].agg(...)
result = result.head(100)  # Limit to 100 rows
```

**Rules**:
1. Use pd.to_numeric(df[col], errors='coerce') for type conversions
2. Handle missing values with dropna() if needed
3. Assign final result to 'result' variable
4. Limit DataFrame results to 100 rows with .head(100)
"""

        return base_prompt + react_section + reasoning_section + code_section

    def _build_analysis_code_prompt(
        self, query: str, columns: str, dtypes: str, sample_rows: list[dict], summary_text: str = ""
    ) -> str:
        """Build prompt for analysis code generation (no plotting)."""
        summary_context = f"\nDataset Context:\n{summary_text}\n" if summary_text else ""

        return f"""Given DataFrame `df` with columns: {columns}

Data types:
{dtypes}
{summary_context}
Sample rows:
{json.dumps(sample_rows, indent=2, default=str)}

Write Python code (pandas only, no plotting) to answer: "{query}"

**IMPORTANT: Use Chain of Thought reasoning before writing code.**

First, think through your approach inside <thinking> tags:
<thinking>
1. What is the user asking for?
2. Which columns are relevant?
3. What aggregations or transformations are needed?
4. What is the expected output format?
5. What edge cases should I handle?
</thinking>

Then provide the Python code inside markdown fence: ```python ... ```

Rules:
1. Use pandas operations on `df` only
2. Assign the final result to `result` variable
3. Handle missing values with dropna() before aggregations
4. Do NOT import any libraries (pandas already imported as pd, numpy as np)
5. Do NOT read files or fetch data - use only the provided `df`
6. If the result is a scalar value, return it as-is
7. If the result is a DataFrame or Series, limit to top 100 rows with .head(100) for better visualization
8. **IMPORTANT: Use pd.to_numeric(df[col], errors='coerce') instead of .astype(int) for type conversion**
   - Example: pd.to_numeric(df['year'], errors='coerce').between(2000, 2020)
   - This safely handles bad data values that can't be converted

Example:
<thinking>
User wants average employment by year. I need to:
1. Group by 'year' column
2. Calculate mean of 'employment' column
3. Sort the results
4. Limit to 100 rows
</thinking>

```python
result = df.groupby("year")["employment"].mean().sort_values().head(100)
```
"""

    def _build_plot_code_prompt(
        self, query: str, columns: str, dtypes: str, sample_rows: list[dict], summary_text: str = ""
    ) -> str:
        """Build prompt for plotting code generation."""
        summary_context = f"\nDataset Context:\n{summary_text}\n" if summary_text else ""

        return f"""Given DataFrame `df` with columns: {columns}

Data types:
{dtypes}
{summary_context}
Sample rows:
{json.dumps(sample_rows, indent=2, default=str)}

Write Python code using pandas and matplotlib to answer: "{query}"

**IMPORTANT: Use Chain of Thought reasoning before writing code.**

First, think through your visualization approach inside <thinking> tags:
<thinking>
1. What is the user asking to visualize?
2. Which chart type best shows this data? (bar, line, scatter, etc.)
3. What should be on X-axis? (typically: year, category, time)
4. What should be on Y-axis? (typically: metric, count, rate)
5. **CRITICAL: What are the UNITS of the Y-axis data?**
   - Check column names for clues: "rate", "percentage", "pct", "%", "share" → use "%"
   - Check data values: if values are 0-100 or 0-1 → likely percentage
   - Check for "change", "growth" → could be thousands, percentage, or index
   - Default: use the actual column name or infer from context
6. Are there multiple groups to show? (use colors/legend)
7. What data transformations are needed?
8. What makes this chart clear and informative?
</thinking>

Then provide the Python code inside markdown fence: ```python ... ```

Rules:
1. Use pandas for data manipulation
2. Use matplotlib.pyplot (as plt) for plotting
3. Assign the matplotlib Figure to `result` variable
4. Create only ONE relevant plot with clear title and axis labels
5. Handle missing values with dropna()
6. Set figsize=(10, 6) for better visibility
7. Do NOT import pandas or numpy (already imported as pd, np)
8. Close figure with plt.close() after assigning to result
9. Use appropriate chart type: line for trends, bar for comparisons, scatter for relationships
10. **CRITICAL: Set correct Y-axis label with proper units:**
    - Examine column names and data values to determine units
    - Use "Percentage (%)", "Rate (%)", "Growth Rate (%)" for percentage data
    - Use "Thousands", "Count", "Number" for count data
    - Use "Employment Change (Thousands)" only if data is actually in thousands
    - Check if values are 0-100 (percentage) vs larger numbers (thousands)
    - Example: ax.set_ylabel("Employment Rate (%)", fontsize=12)
11. **IMPORTANT: Use pd.to_numeric(df[col], errors='coerce') instead of .astype(int) for type conversion**
    - Example: pd.to_numeric(df['year'], errors='coerce').between(2000, 2020)
    - This safely handles bad data values that can't be converted

Example:
<thinking>
User wants to see employment trends over time.
- Chart type: Line chart (shows trends over time)
- X-axis: Year (temporal)
- Y-axis: Total employment (metric)
- Need to group by year and sum employment values
</thinking>

```python
import matplotlib.pyplot as plt

data = df.groupby("year")["employment"].sum().head(100)
fig, ax = plt.subplots(figsize=(10, 6))
data.plot(kind="line", ax=ax, marker='o')
ax.set_title("Employment Trend by Year")
ax.set_xlabel("Year")
ax.set_ylabel("Total Employment")
ax.grid(True, alpha=0.3)
result = fig
plt.close()
```
"""

    def _extract_code_block(self, response: str) -> str:
        """Extract Python code from markdown fence."""
        if "```python" in response:
            start = response.index("```python") + 9
            end = response.index("```", start)
            return response[start:end].strip()
        elif "```" in response:
            start = response.index("```") + 3
            end = response.index("```", start)
            return response[start:end].strip()

        # If no code blocks found, check if response looks like Python code
        # (starts with valid Python keywords or assignments)
        response_stripped = response.strip()
        python_keywords = [
            "import",
            "from",
            "def",
            "class",
            "if",
            "for",
            "while",
            "result =",
            "df.",
            "df[",
        ]
        if any(response_stripped.startswith(keyword) for keyword in python_keywords):
            return response_stripped

        # Otherwise, return empty string (not valid Python code)
        logger.warning(f"No valid Python code found in response: {response[:100]}...")
        return ""

    async def _execute_code_safely(
        self,
        code: str,
        timeout: float = 5.0,
        df: pd.DataFrame | None = None,
        should_plot: bool = False,
    ) -> dict[str, Any]:
        """Execute code safely using process-isolated sandbox with resource limits.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds (unused, configured in sandbox)
            df: Optional DataFrame to make available as 'df'
            should_plot: Whether to allow matplotlib plotting

        Returns:
            Dict containing execution result or error
        """
        # Prepare execution context
        context = {}
        if df is not None:
            context["df"] = df.copy()

        # Execute in sandbox (runs asynchronously in executor)
        loop = asyncio.get_event_loop()

        def execute_sync():
            return self.sandbox_executor.execute_code(code, context=context)

        try:
            execution_result = await loop.run_in_executor(None, execute_sync)

            if execution_result.success:
                logger.info(
                    f"Code executed successfully in {execution_result.execution_time:.3f}s, "
                    f"result type: {type(execution_result.result).__name__}"
                )
                return {"result": execution_result.result, "error": None}
            else:
                logger.error(
                    f"Code execution failed: {execution_result.error_type} - {execution_result.error}"
                )
                # Return error dict instead of raising
                return {"result": None, "error": execution_result.error}

        except Exception as e:
            logger.error(f"Sandbox execution error: {e}", exc_info=True)
            return {"result": None, "error": f"Sandbox error: {str(e)}"}

    async def _execute_safe(
        self, code: str, df: pd.DataFrame, should_plot: bool
    ) -> tuple[Any, str | None]:
        """Execute code in controlled environment with safety checks."""
        try:
            result_dict = await self._execute_code_safely(
                code, timeout=5.0, df=df, should_plot=should_plot
            )
            return result_dict["result"], result_dict["error"]
        except TimeoutError as e:
            return None, str(e)
        except Exception as e:
            return None, f"Execution error: {str(e)}"

    async def _generate_explanation(
        self, query: str, result: Any, validation_context: dict[str, Any] | None = None
    ) -> str:
        """Generate natural language explanation using LLM with validation awareness."""
        from matplotlib.figure import Figure

        if isinstance(result, Figure):
            result_desc = "[Matplotlib Figure - chart generated]"
        elif isinstance(result, pd.DataFrame):
            result_desc = f"DataFrame with {len(result)} rows:\n{result.head(10).to_string()}"
        elif isinstance(result, pd.Series):
            result_desc = f"Series with {len(result)} values:\n{result.head(10).to_string()}"
        else:
            result_desc = str(result)[:500]

        # Build validation disclaimer if data doesn't exactly match request
        validation_notice = ""
        if validation_context and validation_context.get("status") == "partial_match":
            missing = validation_context.get("missing_concepts", [])
            available = validation_context.get("available_alternatives", [])
            validation_notice = f"""

**IMPORTANT LIMITATION:**
The user asked about: {self._safe_join(missing)}
However, this data is NOT available in the dataset.

Instead, the analysis shows data for: {self._safe_join(available)}

You MUST acknowledge this limitation in your response. Start with a clear statement like:
"Note: The dataset doesn't contain data about {self._safe_join(missing)}. Instead, here's what the available data shows about {self._safe_join(available)}..."
"""

        prompt = f"""The user asked: "{query}"
{validation_notice}
The analysis produced this result:
{result_desc}

**Use Chain of Thought reasoning to analyze the results:**

<thinking>
1. Does this data actually answer what the user asked for?
   {f"   - NO: The user asked about {self._safe_join(missing if validation_context else [])} but data only has {self._safe_join(available if validation_context else [])}" if validation_context and validation_context.get("status") == "partial_match" else "   - YES: Data matches the request"}
2. What patterns or trends do I see in the available data?
3. What are the key insights from what IS available?
4. Are there any notable outliers or interesting observations?
5. What's the most important takeaway?
</thinking>

Then provide a clear, honest explanation:
- If data doesn't match the request, START with that acknowledgment
- Then explain what the available data actually shows
- Focus on insights from the ACTUAL data, not the requested data
- Be honest about limitations

Example format when data doesn't match:
<thinking>
User asked about technology sector, but data only has age and sex breakdowns...
I need to be clear this doesn't answer their question...
But I can explain what employment patterns by age/sex show...
</thinking>

Note: The dataset doesn't contain employment data by sector. Instead, here's what the data shows about employment by age and sex: [analysis of actual data]. [Key insight from available data]. [Pattern observed]."""

        response = await self._invoke_llm([HumanMessage(content=prompt)])
        return response

    # ======= Visualization Extraction Helper Methods =======

    def _extract_viz_from_figure(
        self, fig, original_query: str, validation: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Extract data from matplotlib Figure and convert to VisualizationData format.

        Uses chart's own title/labels instead of user's question when data doesn't match request.
        """
        try:
            ax = fig.get_axes()[0] if fig.get_axes() else None
            if not ax:
                logger.warning("No axes found in matplotlib figure")
                return None

            # Get title from chart itself (more accurate than user's question)
            chart_title = ax.get_title()

            # If validation shows partial match, use chart's descriptive title, not user's question
            if validation and validation.get("status") == "partial_match":
                # Use chart's own title or generate from axis labels
                if not chart_title:
                    x_label = ax.get_xlabel() or "Category"
                    y_label = ax.get_ylabel() or "Value"
                    chart_title = f"{y_label} by {x_label}"
                logger.info(f"[VIZ] Using descriptive title (partial match): {chart_title}")
            elif not chart_title:
                # Use original query as fallback only when exact match
                chart_title = original_query[:100]

            # Extract data based on plot type
            lines = ax.get_lines()
            bars = ax.containers

            if bars:  # Bar chart
                return self._extract_bar_data(ax, chart_title)
            elif lines:  # Line chart
                return self._extract_line_data(ax, chart_title)
            else:
                logger.warning("Could not identify plot type in figure")
                return None

        except Exception as e:
            logger.warning(f"Failed to extract viz from figure: {e}", exc_info=True)
            return None

    def _extract_bar_data(self, ax, title: str) -> dict[str, Any]:
        """Extract bar chart data from matplotlib axes."""
        try:
            bars = ax.containers[0] if ax.containers else None
            if not bars:
                return None

            # Get axis labels from the plot
            x_label = ax.get_xlabel() or "Category"
            y_label = ax.get_ylabel() or "Value"

            # Get x labels and heights
            x_labels = [label.get_text() for label in ax.get_xticklabels()]
            heights = [bar.get_height() for bar in bars]

            # SMART AXIS SWAP DETECTION
            # Check if axes are swapped (year/temporal on y-axis instead of x-axis)
            temporal_keywords = ["year", "date", "month", "quarter", "time"]
            y_is_temporal = any(keyword in y_label.lower() for keyword in temporal_keywords)
            x_is_temporal = any(keyword in x_label.lower() for keyword in temporal_keywords)

            # Check if y-axis values look like years
            y_looks_like_years = any(
                1900 <= h <= 2100 for h in heights if isinstance(h, int | float)
            )

            should_swap = y_is_temporal or (y_looks_like_years and not x_is_temporal)

            if should_swap:
                logger.info(f"Detected swapped axes! Swapping '{x_label}' (x) <-> '{y_label}' (y)")
                # Swap axes
                x_labels, heights = heights, x_labels
                x_label, y_label = y_label, x_label

            # Use axis labels as keys for better context
            x_key = x_label.lower().replace(" ", "_")[:20] or "category"
            y_key = y_label.lower().replace(" ", "_")[:20] or "value"

            # Filter out empty labels
            data = []
            for label, height in zip(x_labels, heights):
                if label and height is not None:
                    data.append({x_key: str(label), y_key: float(height)})

            if not data:
                return None

            return {
                "chart_type": "bar",
                "title": title or ax.get_title() or "Chart",
                "data": data,
                "x_axis": x_key,
                "y_axis": y_key,
                "x_label": x_label,  # Add human-readable label
                "y_label": y_label,  # Add human-readable label
                "description": f"{y_label} by {x_label}",
            }
        except Exception as e:
            logger.warning(f"Failed to extract bar data: {e}")
            return None

    def _extract_line_data(self, ax, title: str) -> dict[str, Any]:
        """Extract line chart data from matplotlib axes."""
        try:
            lines = ax.get_lines()
            if not lines:
                return None

            # If multiple lines, try to find the best line to represent the data
            line = lines[0]  # Default to first line
            if len(lines) > 1:
                # Strategy 1: Find line with "total" or "aggregate" in label
                for l in lines:
                    label = l.get_label().lower() if l.get_label() else ""
                    if any(
                        keyword in label
                        for keyword in ["total", "aggregate", "overall", "combined"]
                    ):
                        line = l
                        logger.info(f"Selected aggregate line: {l.get_label()}")
                        break
                else:
                    # Strategy 2: Use the thickest line (usually aggregate/emphasis line)
                    thickest = max(lines, key=lambda l: l.get_linewidth())
                    if thickest.get_linewidth() > lines[0].get_linewidth():
                        line = thickest
                        logger.info(
                            f"Selected thickest line (width={thickest.get_linewidth()}): {thickest.get_label()}"
                        )
                    else:
                        # Strategy 3: Use the line with the most data points (most complete time series)
                        longest = max(lines, key=lambda l: len(l.get_xdata()))
                        if len(longest.get_xdata()) > len(lines[0].get_xdata()):
                            line = longest
                            logger.info(
                                f"Selected longest line ({len(longest.get_xdata())} points): {longest.get_label()}"
                            )

            x_data = line.get_xdata()
            y_data = line.get_ydata()

            # Get axis labels from the plot
            x_label = ax.get_xlabel() or "Period"
            y_label = ax.get_ylabel() or "Value"

            # Get x labels if available (with safe conversion)
            x_labels = [label.get_text() for label in ax.get_xticklabels()]
            if not x_labels or all(not label for label in x_labels):
                # Safely convert x_data to labels, handling bad values
                safe_labels = []
                for x in x_data:
                    try:
                        if isinstance(x, int | float) and not pd.isna(x) and x == int(x):
                            safe_labels.append(str(int(x)))
                        else:
                            safe_labels.append(str(x))
                    except (ValueError, TypeError):
                        # Handle any conversion errors gracefully
                        safe_labels.append(str(x))
                x_labels = safe_labels

            # SMART AXIS SWAP DETECTION
            temporal_keywords = ["year", "date", "month", "quarter", "time"]
            y_is_temporal = any(keyword in y_label.lower() for keyword in temporal_keywords)
            x_is_temporal = any(keyword in x_label.lower() for keyword in temporal_keywords)

            # Check if y-data looks like years
            y_looks_like_years = any(
                1900 <= y <= 2100 for y in y_data if isinstance(y, int | float)
            )

            should_swap = y_is_temporal or (y_looks_like_years and not x_is_temporal)

            if should_swap:
                logger.info(
                    f"Detected swapped axes in line chart! Swapping '{x_label}' (x) <-> '{y_label}' (y)"
                )
                # Swap axes
                x_data, y_data = y_data, x_data
                x_labels = [str(x) for x in x_data]
                x_label, y_label = y_label, x_label

            # Use axis labels as keys for better context
            x_key = x_label.lower().replace(" ", "_")[:20] or "x"
            y_key = y_label.lower().replace(" ", "_")[:20] or "y"

            data = []
            for label, y in zip(x_labels[: len(y_data)], y_data):
                if label:
                    data.append({x_key: str(label), y_key: float(y)})

            if not data:
                return None

            return {
                "chart_type": "line",
                "title": title or ax.get_title() or "Line Chart",
                "data": data,
                "x_axis": x_key,
                "y_axis": y_key,
                "x_label": x_label,  # Add human-readable label
                "y_label": y_label,  # Add human-readable label
                "description": f"{y_label} over {x_label}",
            }
        except Exception as e:
            logger.warning(f"Failed to extract line data: {e}")
            return None

    def _generate_viz_from_dataframe(
        self, df_or_series, original_query: str, validation: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Generate visualization from DataFrame or Series.

        Uses descriptive titles based on actual data, not user's question when data doesn't match.
        """
        try:
            if isinstance(df_or_series, pd.Series):
                # Convert Series to DataFrame
                df = df_or_series.reset_index()
                df.columns = ["category", "value"]
            else:
                df = df_or_series

            if len(df.columns) < 2:
                logger.warning("DataFrame has fewer than 2 columns, cannot generate viz")
                return None

            # Smart column detection: find x (categorical/year) and y (metric) columns
            x_col = None
            y_col = None

            # Priority 1: Look for 'year', 'date', 'month' columns for x-axis (temporal)
            temporal_keywords = ["year", "date", "month", "quarter", "time", "period"]
            for col in df.columns:
                if any(keyword in col.lower() for keyword in temporal_keywords):
                    x_col = col
                    break

            # Priority 2: Find metric column for y-axis (employment_rate, count, value, etc.)
            metric_keywords = [
                "rate",
                "count",
                "value",
                "change",
                "percentage",
                "total",
                "number",
                "amount",
            ]
            for col in df.columns:
                if col != x_col and pd.api.types.is_numeric_dtype(df[col]):
                    # Prefer columns with metric keywords in name
                    if any(keyword in col.lower() for keyword in metric_keywords):
                        y_col = col
                        break

            # If no metric column found, use any numeric column
            if not y_col:
                for col in df.columns:
                    if col != x_col and pd.api.types.is_numeric_dtype(df[col]):
                        y_col = col
                        break

            # If no temporal column found, use first categorical column
            if not x_col:
                for col in df.columns:
                    if col != y_col and not pd.api.types.is_numeric_dtype(df[col]):
                        x_col = col
                        break

            # Final fallback: use first two columns
            if not x_col or not y_col:
                x_col = df.columns[0]
                y_col = df.columns[1]

            logger.info(f"[VIZ] Column selection: x_col='{x_col}', y_col='{y_col}'")

            # Limit to top 100 rows for better time series visualization
            df_limited = df.head(100)

            data = []
            for _, row in df_limited.iterrows():
                try:
                    data.append(
                        {
                            x_col: str(row[x_col]),
                            y_col: float(row[y_col]) if pd.notna(row[y_col]) else 0,
                        }
                    )
                except (ValueError, TypeError):
                    continue

            if not data:
                logger.warning("No valid data points after conversion")
                return None

            # Create human-readable labels from column names
            x_label = x_col.replace("_", " ").title()
            y_label = y_col.replace("_", " ").title()

            # Generate descriptive title based on ACTUAL data columns
            # Don't use original query if validation shows partial match
            if validation and validation.get("status") == "partial_match":
                # Use descriptive title based on what's actually plotted
                chart_title = f"{y_label} by {x_label}"
                logger.info(f"[VIZ] Using descriptive title (partial match): {chart_title}")
            else:
                # Use original query only for exact matches
                chart_title = original_query[:100] if original_query else f"{y_label} by {x_label}"

            return {
                "chart_type": "bar",
                "title": chart_title,
                "data": data,
                "x_axis": x_col,
                "y_axis": y_col,
                "x_label": x_label,
                "y_label": y_label,
                "description": f"{y_label} across different {x_label} values",
            }
        except Exception as e:
            logger.warning(f"Failed to generate viz from DataFrame: {e}", exc_info=True)
            return None

    def _generate_html_chart(self, viz_data: dict[str, Any]) -> str | None:
        """Generate interactive HTML chart using Plotly.

        Args:
            viz_data: Visualization data dict with chart_type, data, x_axis, y_axis, etc.

        Returns:
            HTML string for the chart, or None if generation fails
        """
        try:
            import plotly.graph_objects as go

            chart_type = viz_data.get("chart_type", "bar")
            data = viz_data.get("data", [])
            title = viz_data.get("title", "Chart")
            x_axis = viz_data.get("x_axis", "x")
            y_axis = viz_data.get("y_axis", "y")
            x_label = viz_data.get("x_label") or x_axis.replace("_", " ").title()
            y_label = viz_data.get("y_label") or y_axis.replace("_", " ").title()

            if not data:
                logger.warning("Cannot generate HTML chart: data array is empty")
                return None

            # Extract x and y values
            x_values = [row.get(x_axis) for row in data]
            y_values = [row.get(y_axis) for row in data]

            # Create figure based on chart type
            if chart_type == "line":
                fig = go.Figure(
                    data=go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode="lines+markers",
                        name=y_label,
                        line=dict(color="#3b82f6", width=3),
                        marker=dict(size=8),
                    )
                )
            elif chart_type == "bar":
                fig = go.Figure(
                    data=go.Bar(x=x_values, y=y_values, name=y_label, marker=dict(color="#3b82f6"))
                )
            elif chart_type == "scatter":
                fig = go.Figure(
                    data=go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode="markers",
                        name=y_label,
                        marker=dict(size=10, color="#3b82f6"),
                    )
                )
            else:
                # Default to bar chart
                fig = go.Figure(data=go.Bar(x=x_values, y=y_values))

            # Update layout
            fig.update_layout(
                title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18)),
                xaxis_title=x_label,
                yaxis_title=y_label,
                template="plotly_white",
                hovermode="x unified",
                showlegend=False,
                height=400,
                margin=dict(l=60, r=40, t=60, b=60),
            )

            # Generate HTML with responsive sizing
            # Use inline Plotly.js instead of CDN to avoid loading issues
            html = fig.to_html(
                include_plotlyjs=True,  # Inline the library instead of CDN
                full_html=True,  # Generate complete HTML document with DOCTYPE
                config={
                    "displayModeBar": True,
                    "responsive": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                },
                div_id="plotly-chart",
            )

            logger.info(f"Generated HTML chart: {chart_type}, {len(data)} points")
            return html

        except Exception as e:
            logger.warning(f"Failed to generate HTML chart: {e}", exc_info=True)
            return None

    def _build_response(self, result: GraphState, state: AgentState) -> AgentResponse:
        """Build AgentResponse from graph execution result."""
        analysis_results_dict = result.get("analysis_results", {})
        final_response = result.get("intermediate_results", {}).get("final_response", "")
        visualization_dict = result.get("intermediate_results", {}).get("visualization")
        generated_code = result.get("intermediate_results", {}).get("generated_code", "")

        # Create Pydantic models for type safety
        visualization_model = None
        if visualization_dict:
            try:
                visualization_model = VisualizationData(**visualization_dict)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Failed to create VisualizationData model: {e}"
                )

        analysis_result = AnalysisResult(
            text=final_response or analysis_results_dict.get("text", "Analysis complete."),
            visualization=visualization_model,
            data_sources=analysis_results_dict.get("data_sources", []),
        )

        # Update state (keep original dict for backward compatibility)
        state.analysis_results = analysis_results_dict

        return AgentResponse(
            success=True,
            message=analysis_result.text,
            data={
                "analysis": analysis_result.model_dump(),
                "generated_code": generated_code,  # Include for testing/debugging
            },
            visualization=visualization_model.model_dump() if visualization_model else None,
            next_agent=None,  # Analytics is the final agent
            state=state,
        )

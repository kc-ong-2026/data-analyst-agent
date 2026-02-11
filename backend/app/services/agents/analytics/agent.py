"""Analytics Agent - Processes data and generates insights using LangGraph with code generation."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from app.models import AnalysisResult, VisualizationData
from ..base_agent import (
    AgentRole,
    AgentResponse,
    AgentState,
    BaseAgent,
    GraphState,
)
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class AnalyticsAgent(BaseAgent):
    """Agent responsible for analyzing data and generating insights.

    Uses LangGraph to construct a flow with nodes:
    1. prepare_data - Prepare extracted data for analysis
    2. analyze_data - Perform analysis and generate insights
    3. generate_visualization - Create visualization specification
    4. compose_response - Compose final response
    """

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

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for analytics with code generation.

        Flow:
        prepare_data -> generate_code -> execute_code -> explain_results -> [visualize?] -> compose_response -> END
        """
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("prepare_data", self._prepare_data_node)
        workflow.add_node("generate_code", self._generate_code_node)
        workflow.add_node("execute_code", self._execute_code_node)
        workflow.add_node("explain_results", self._explain_results_node)
        workflow.add_node("generate_visualization", self._generate_visualization_node)
        workflow.add_node("compose_response", self._compose_response_node)

        # Set entry point
        workflow.set_entry_point("prepare_data")

        # Add edges
        workflow.add_edge("prepare_data", "generate_code")
        workflow.add_edge("generate_code", "execute_code")
        workflow.add_edge("execute_code", "explain_results")
        workflow.add_conditional_edges(
            "explain_results",
            self._should_visualize,
            {
                "visualize": "generate_visualization",
                "skip": "compose_response",
            }
        )
        workflow.add_edge("generate_visualization", "compose_response")
        workflow.add_edge("compose_response", END)

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

    async def _prepare_data_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Prepare extracted data for analysis by reconstructing DataFrames."""
        extracted_data = state.get("extracted_data", {})
        current_task = state.get("current_task", "")

        # Rebuild DataFrames from extracted data
        dataframes = {}
        data_summary = {}
        total_rows = 0

        for name, data_dict in extracted_data.items():
            if not isinstance(data_dict, dict):
                continue

            try:
                # Check if this is serialized DataFrame data
                source = data_dict.get("source")
                if source in ["rag_metadata", "file_metadata"] and data_dict.get("data"):
                    # Reconstruct DataFrame
                    df = pd.DataFrame(data_dict["data"])

                    # COMPREHENSIVE DATA CLEANING: Clean ALL string columns
                    # Remove footnote markers (like 'a', 'b', 'c') from all text data
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            # Clean trailing letters from numeric-like strings
                            # This handles cases like '2007a', '123b', etc.
                            df[col] = df[col].astype(str).str.replace(r'^(\d+)[a-zA-Z]+$', r'\1', regex=True)
                            # Also clean leading/trailing whitespace
                            df[col] = df[col].str.strip()

                    # Apply dtype conversions if available
                    dtypes = data_dict.get("dtypes", {})
                    for col, dtype_str in dtypes.items():
                        if col in df.columns:
                            try:
                                # Handle common dtype conversions
                                if "int" in dtype_str.lower():
                                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                                elif "float" in dtype_str.lower():
                                    df[col] = pd.to_numeric(df[col], errors='coerce')
                                elif "datetime" in dtype_str.lower():
                                    df[col] = pd.to_datetime(df[col], errors='coerce')
                            except Exception as e:
                                logger.warning(f"Could not convert column {col} to {dtype_str}: {e}")

                    # AUTO-DETECT and clean common year columns
                    year_cols = [c for c in df.columns if 'year' in c.lower()]
                    for col in year_cols:
                        if df[col].dtype == 'object':
                            # Convert year columns to numeric, coercing errors to NaN
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                            logger.info(f"Auto-cleaned year column: {col}")

                    dataframes[name] = df

                    # Create data summary
                    data_summary[name] = {
                        "row_count": len(df),
                        "columns": df.columns.tolist(),
                        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                        "shape": df.shape,
                        "metadata": data_dict.get("metadata", {}),
                    }
                    total_rows += len(df)

                    logger.info(f"Reconstructed DataFrame {name}: {df.shape}")
            except Exception as e:
                logger.error(f"Failed to reconstruct DataFrame from {name}: {e}", exc_info=True)

        return {
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "dataframes": dataframes,  # Live DataFrame objects
                "data_summary": data_summary,
                "total_rows": total_rows,
                "has_data": len(dataframes) > 0,
            },
        }

    async def _generate_code_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Generate pandas code to answer user query."""
        current_task = state.get("current_task", "")
        dataframes = state.get("intermediate_results", {}).get("dataframes", {})
        data_summary = state.get("intermediate_results", {}).get("data_summary", {})

        if not dataframes:
            return {
                "intermediate_results": {
                    **state.get("intermediate_results", {}),
                    "generated_code": None,
                    "should_plot": False,
                },
                "errors": state.get("errors", []) + ["No DataFrames available for code generation"],
            }

        # Get primary DataFrame (first one)
        df_name, df = next(iter(dataframes.items()))

        # Get metadata for the primary DataFrame
        metadata = data_summary.get(df_name, {}).get("metadata", {})

        # Determine if visualization is needed
        should_plot = self._query_needs_visualization(current_task)

        # Generate code using LLM
        code = await self._generate_analysis_code(
            query=current_task,
            df=df,
            df_name=df_name,
            should_plot=should_plot,
            metadata=metadata
        )

        return {
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "generated_code": code,
                "should_plot": should_plot,
                "primary_df_name": df_name,
            },
        }

    async def _execute_code_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Execute generated code in controlled environment."""
        code = state.get("intermediate_results", {}).get("generated_code")
        dataframes = state.get("intermediate_results", {}).get("dataframes", {})
        df_name = state.get("intermediate_results", {}).get("primary_df_name")
        should_plot = state.get("intermediate_results", {}).get("should_plot", False)

        if not code or df_name not in dataframes:
            return {
                "intermediate_results": {
                    **state.get("intermediate_results", {}),
                    "execution_result": None,
                    "execution_error": "No code to execute or DataFrame not found",
                },
            }

        # Execute in safe environment
        result, error = await self._execute_safe(
            code=code,
            df=dataframes[df_name],
            should_plot=should_plot
        )

        result_type = type(result).__name__ if result is not None else "None"

        return {
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "execution_result": result,
                "execution_error": error,
                "result_type": result_type,
            },
        }

    async def _explain_results_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Generate natural language explanation of results."""
        current_task = state.get("current_task", "")
        result = state.get("intermediate_results", {}).get("execution_result")
        error = state.get("intermediate_results", {}).get("execution_error")

        if error:
            explanation = f"I encountered an error while analyzing the data: {error}"
        elif result is None:
            explanation = "I couldn't generate meaningful results from the data."
        else:
            # Use LLM to explain the result
            explanation = await self._generate_explanation(
                query=current_task,
                result=result
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

    async def _analyze_data_node(self, state: GraphState) -> Dict[str, Any]:
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
        if "query_result" in extracted_data and extracted_data["query_result"].get("data"):
            has_actual_data = True
        elif total_rows > 0:
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

    async def _generate_visualization_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Generate visualization from code execution results."""
        result = state.get("intermediate_results", {}).get("execution_result")
        current_task = state.get("current_task", "")
        should_plot = state.get("intermediate_results", {}).get("should_plot", False)

        viz = None

        # Strategy 1: If matplotlib Figure was generated, extract data from it
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        if isinstance(result, Figure):
            viz = self._extract_viz_from_figure(result, current_task)
            logger.info("Extracted visualization from matplotlib Figure")

        # Strategy 2: If DataFrame/Series returned, auto-generate visualization
        elif isinstance(result, (pd.DataFrame, pd.Series)):
            viz = self._generate_viz_from_dataframe(result, current_task)
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

    async def _compose_response_node(self, state: GraphState) -> Dict[str, Any]:
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
                logger.info(f"Returning visualization with both HTML chart and {len(visualization.get('data', []))} data points for fallback")

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

    def _format_data_for_analysis(
        self,
        extracted_data: Dict[str, Any],
        data_summary: Dict[str, Any],
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
            part = f"\n### SQL Query Result\n"
            part += f"Rows: {len(rows)}\n"
            part += f"Columns: {', '.join(query_data.get('columns', [])[:10])}\n"
            part += f"Data (first 10 rows from SQL query):\n"
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
                part += f"Data (first 10 rows):\n"
                part += json.dumps(rows[:10], indent=2, default=str)
            elif metadata:
                # Use metadata for context when no rows available
                part += f"\nMetadata Context:\n"
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
        extracted_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
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
        rows: List[Dict[str, Any]],
        columns: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Generate visualization from rows of data."""
        if len(columns) < 2:
            return None

        # Smart column detection: find x (categorical/year) and y (metric) columns
        x_col = None
        y_col = None

        # Priority 1: Look for temporal columns for x-axis
        temporal_keywords = ['year', 'date', 'month', 'quarter', 'time', 'period']
        for col in columns:
            if any(keyword in col.lower() for keyword in temporal_keywords):
                x_col = col
                break

        # Priority 2: Find metric column for y-axis (employment metrics, counts, etc.)
        metric_keywords = ['rate', 'count', 'value', 'change', 'percentage', 'total', 'number', 'amount']
        for col in columns:
            if col != x_col:
                # Check if column is numeric
                sample = rows[0].get(col)
                if isinstance(sample, (int, float)):
                    # Prefer columns with metric keywords in name
                    if any(keyword in col.lower() for keyword in metric_keywords):
                        y_col = col
                        break

        # If no metric column found, find any numeric column
        if not y_col:
            for col in columns:
                if col != x_col:
                    sample = rows[0].get(col)
                    if isinstance(sample, (int, float)):
                        y_col = col
                        break

        # If no temporal column found, find categorical column
        if not x_col:
            for col in columns:
                if col != y_col:
                    sample = rows[0].get(col)
                    if not isinstance(sample, (int, float)):
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
            viz_data.append({
                x_col: str(row.get(x_col, ""))[:30],  # Category label (string)
                y_col: row.get(y_col, 0),  # Numeric value
            })

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

    def _parse_visualization(self, response: str) -> Optional[Dict[str, Any]]:
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

    def _validate_visualization(self, viz: Dict[str, Any]) -> Dict[str, Any]:
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
                        if not isinstance(data[0][key], (int, float)):
                            x_axis = key
                            break
                    if not x_axis:
                        x_axis = keys_list[0]

                    # Find first numeric column for y_axis (values)
                    y_axis = None
                    for key in keys_list:
                        if isinstance(data[0][key], (int, float)):
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
                    if not isinstance(data[0][key], (int, float)):
                        viz["x_axis"] = key
                        break

            if y_axis and y_axis not in sample_keys:
                # Try to find best match (numeric field)
                for key in sample_keys:
                    if isinstance(data[0][key], (int, float)):
                        viz["y_axis"] = key
                        break

            # CRITICAL FIX: Convert x-axis values to strings for proper Recharts rendering
            # Numeric x values (like years) can cause Recharts to misinterpret the axis
            if x_axis and chart_type in ["bar", "line"]:
                viz["data"] = [
                    {
                        **row,
                        x_axis: str(row.get(x_axis, ""))[:50]  # Convert to string, limit length
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
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Clean up extra whitespace left behind
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple blank lines → double newline

        return text.strip()

    def _strip_json_code_blocks(self, text: str) -> str:
        """Remove JSON code blocks from analysis text to prevent frontend confusion.

        This ensures the visualization field is the single source of truth.
        """
        import re

        # Pattern to match JSON code blocks (```json...``` or ```...``` with JSON content)
        patterns = [
            r'```json\s*\{[^`]*\}\s*```',  # Explicit JSON blocks
            r'```\s*\{[^`]*\}\s*```',       # Generic code blocks with JSON objects
        ]

        cleaned_text = text
        for pattern in patterns:
            # Replace JSON blocks with a placeholder message
            cleaned_text = re.sub(
                pattern,
                '[Visualization generated - see visualization panel]',
                cleaned_text,
                flags=re.DOTALL
            )

        return cleaned_text.strip()

    # ======= Code Generation Helper Methods =======

    def _query_needs_visualization(self, query: str) -> bool:
        """Determine if query requires visualization."""
        query_lower = query.lower()
        viz_keywords = ["plot", "chart", "graph", "visualize", "show", "display", "trend"]
        return any(keyword in query_lower for keyword in viz_keywords)

    async def _generate_analysis_code(
        self, query: str, df: pd.DataFrame, df_name: str, should_plot: bool, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate pandas code using LLM."""
        columns_str = ", ".join(df.columns.tolist())
        dtypes_str = "\n".join([f"  {col}: {dtype}" for col, dtype in df.dtypes.items()])
        sample_rows = df.head(3).to_dict(orient="records")
        summary_text = metadata.get("summary_text", "") if metadata else ""

        if should_plot:
            prompt = self._build_plot_code_prompt(query, columns_str, dtypes_str, sample_rows, summary_text)
        else:
            prompt = self._build_analysis_code_prompt(query, columns_str, dtypes_str, sample_rows, summary_text)

        response = await self._invoke_llm([HumanMessage(content=prompt)])
        code = self._extract_code_block(response)

        logger.info(f"Generated code for query: {query}")
        logger.info(f"Code:\n{code}")

        return code

    def _build_analysis_code_prompt(
        self, query: str, columns: str, dtypes: str, sample_rows: List[Dict], summary_text: str = ""
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
        self, query: str, columns: str, dtypes: str, sample_rows: List[Dict], summary_text: str = ""
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
        return response.strip()

    async def _execute_safe(
        self, code: str, df: pd.DataFrame, should_plot: bool
    ) -> Tuple[Any, Optional[str]]:
        """Execute code in controlled environment with safety checks."""
        # Setup restricted environment
        env = {
            "pd": pd,
            "np": np,
            "df": df.copy(),  # Work on copy to prevent mutations
            "result": None
        }

        if should_plot:
            import matplotlib
            import matplotlib.pyplot as plt
            # Use non-interactive backend
            matplotlib.use('Agg')
            env["plt"] = plt
            env["matplotlib"] = matplotlib

        try:
            # Execute with timeout (5 seconds)
            loop = asyncio.get_event_loop()

            def execute_sync():
                exec(code, env, env)
                return env.get("result")

            result = await asyncio.wait_for(
                loop.run_in_executor(None, execute_sync),
                timeout=5.0
            )

            logger.info(f"Code executed successfully, result type: {type(result).__name__}")
            return result, None

        except asyncio.TimeoutError:
            logger.error("Code execution timeout (>5 seconds)")
            return None, "Code execution timeout (>5 seconds)"
        except Exception as e:
            logger.error(f"Code execution error: {e}", exc_info=True)
            return None, f"Execution error: {str(e)}"

    async def _generate_explanation(self, query: str, result: Any) -> str:
        """Generate natural language explanation using LLM."""
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        if isinstance(result, Figure):
            result_desc = "[Matplotlib Figure - chart generated]"
        elif isinstance(result, pd.DataFrame):
            result_desc = f"DataFrame with {len(result)} rows:\n{result.head(10).to_string()}"
        elif isinstance(result, pd.Series):
            result_desc = f"Series with {len(result)} values:\n{result.head(10).to_string()}"
        else:
            result_desc = str(result)[:500]

        prompt = f"""The user asked: "{query}"

The analysis produced this result:
{result_desc}

**Use Chain of Thought reasoning to analyze the results:**

<thinking>
1. What patterns or trends do I see in the data?
2. What are the key insights?
3. Are there any notable outliers or interesting observations?
4. How does this answer the user's question?
5. What's the most important takeaway?
</thinking>

Then provide a clear, concise explanation (2-3 sentences) of what this result tells us about the data.
Focus on insights and patterns, not code or technical details.

Example format:
<thinking>
The data shows employment increasing from 2010 to 2020...
Key insight is the 15% growth rate...
Notable: 2015 had a dip due to...
</thinking>

The analysis reveals [key insight]. [Supporting detail or pattern]. [Implication or conclusion]."""

        response = await self._invoke_llm([HumanMessage(content=prompt)])
        return response

    # ======= Visualization Extraction Helper Methods =======

    def _extract_viz_from_figure(
        self, fig, title: str
    ) -> Optional[Dict[str, Any]]:
        """Extract data from matplotlib Figure and convert to VisualizationData format."""
        try:
            ax = fig.get_axes()[0] if fig.get_axes() else None
            if not ax:
                logger.warning("No axes found in matplotlib figure")
                return None

            # Extract data based on plot type
            lines = ax.get_lines()
            bars = ax.containers

            if bars:  # Bar chart
                return self._extract_bar_data(ax, title)
            elif lines:  # Line chart
                return self._extract_line_data(ax, title)
            else:
                logger.warning("Could not identify plot type in figure")
                return None

        except Exception as e:
            logger.warning(f"Failed to extract viz from figure: {e}", exc_info=True)
            return None

    def _extract_bar_data(self, ax, title: str) -> Dict[str, Any]:
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
            temporal_keywords = ['year', 'date', 'month', 'quarter', 'time']
            y_is_temporal = any(keyword in y_label.lower() for keyword in temporal_keywords)
            x_is_temporal = any(keyword in x_label.lower() for keyword in temporal_keywords)

            # Check if y-axis values look like years
            y_looks_like_years = any(1900 <= h <= 2100 for h in heights if isinstance(h, (int, float)))

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
                "description": f"{y_label} by {x_label}"
            }
        except Exception as e:
            logger.warning(f"Failed to extract bar data: {e}")
            return None

    def _extract_line_data(self, ax, title: str) -> Dict[str, Any]:
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
                    if any(keyword in label for keyword in ['total', 'aggregate', 'overall', 'combined']):
                        line = l
                        logger.info(f"Selected aggregate line: {l.get_label()}")
                        break
                else:
                    # Strategy 2: Use the thickest line (usually aggregate/emphasis line)
                    thickest = max(lines, key=lambda l: l.get_linewidth())
                    if thickest.get_linewidth() > lines[0].get_linewidth():
                        line = thickest
                        logger.info(f"Selected thickest line (width={thickest.get_linewidth()}): {thickest.get_label()}")
                    else:
                        # Strategy 3: Use the line with the most data points (most complete time series)
                        longest = max(lines, key=lambda l: len(l.get_xdata()))
                        if len(longest.get_xdata()) > len(lines[0].get_xdata()):
                            line = longest
                            logger.info(f"Selected longest line ({len(longest.get_xdata())} points): {longest.get_label()}")

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
                        if isinstance(x, (int, float)) and not pd.isna(x) and x == int(x):
                            safe_labels.append(str(int(x)))
                        else:
                            safe_labels.append(str(x))
                    except (ValueError, TypeError):
                        # Handle any conversion errors gracefully
                        safe_labels.append(str(x))
                x_labels = safe_labels

            # SMART AXIS SWAP DETECTION
            temporal_keywords = ['year', 'date', 'month', 'quarter', 'time']
            y_is_temporal = any(keyword in y_label.lower() for keyword in temporal_keywords)
            x_is_temporal = any(keyword in x_label.lower() for keyword in temporal_keywords)

            # Check if y-data looks like years
            y_looks_like_years = any(1900 <= y <= 2100 for y in y_data if isinstance(y, (int, float)))

            should_swap = y_is_temporal or (y_looks_like_years and not x_is_temporal)

            if should_swap:
                logger.info(f"Detected swapped axes in line chart! Swapping '{x_label}' (x) <-> '{y_label}' (y)")
                # Swap axes
                x_data, y_data = y_data, x_data
                x_labels = [str(x) for x in x_data]
                x_label, y_label = y_label, x_label

            # Use axis labels as keys for better context
            x_key = x_label.lower().replace(" ", "_")[:20] or "x"
            y_key = y_label.lower().replace(" ", "_")[:20] or "y"

            data = []
            for label, y in zip(x_labels[:len(y_data)], y_data):
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
                "description": f"{y_label} over {x_label}"
            }
        except Exception as e:
            logger.warning(f"Failed to extract line data: {e}")
            return None

    def _generate_viz_from_dataframe(
        self, df_or_series, title: str
    ) -> Optional[Dict[str, Any]]:
        """Generate visualization from DataFrame or Series."""
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
            temporal_keywords = ['year', 'date', 'month', 'quarter', 'time', 'period']
            for col in df.columns:
                if any(keyword in col.lower() for keyword in temporal_keywords):
                    x_col = col
                    break

            # Priority 2: Find metric column for y-axis (employment_rate, count, value, etc.)
            metric_keywords = ['rate', 'count', 'value', 'change', 'percentage', 'total', 'number', 'amount']
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

            logger.info(f"Column selection: x_col='{x_col}', y_col='{y_col}'")

            # Limit to top 100 rows for better time series visualization
            df_limited = df.head(100)

            data = []
            for _, row in df_limited.iterrows():
                try:
                    data.append({
                        x_col: str(row[x_col]),
                        y_col: float(row[y_col]) if pd.notna(row[y_col]) else 0
                    })
                except (ValueError, TypeError):
                    continue

            if not data:
                logger.warning("No valid data points after conversion")
                return None

            # Create human-readable labels from column names
            x_label = x_col.replace("_", " ").title()
            y_label = y_col.replace("_", " ").title()

            return {
                "chart_type": "bar",
                "title": title or f"{y_label} by {x_label}",
                "data": data,
                "x_axis": x_col,
                "y_axis": y_col,
                "x_label": x_label,
                "y_label": y_label,
                "description": f"{y_label} across different {x_label} values"
            }
        except Exception as e:
            logger.warning(f"Failed to generate viz from DataFrame: {e}", exc_info=True)
            return None

    def _generate_html_chart(self, viz_data: Dict[str, Any]) -> Optional[str]:
        """Generate interactive HTML chart using Plotly.

        Args:
            viz_data: Visualization data dict with chart_type, data, x_axis, y_axis, etc.

        Returns:
            HTML string for the chart, or None if generation fails
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px

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
                fig = go.Figure(data=go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines+markers',
                    name=y_label,
                    line=dict(color='#3b82f6', width=3),
                    marker=dict(size=8)
                ))
            elif chart_type == "bar":
                fig = go.Figure(data=go.Bar(
                    x=x_values,
                    y=y_values,
                    name=y_label,
                    marker=dict(color='#3b82f6')
                ))
            elif chart_type == "scatter":
                fig = go.Figure(data=go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='markers',
                    name=y_label,
                    marker=dict(size=10, color='#3b82f6')
                ))
            else:
                # Default to bar chart
                fig = go.Figure(data=go.Bar(x=x_values, y=y_values))

            # Update layout
            fig.update_layout(
                title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18)),
                xaxis_title=x_label,
                yaxis_title=y_label,
                template='plotly_white',
                hovermode='x unified',
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
                    'displayModeBar': True,
                    'responsive': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
                },
                div_id='plotly-chart'
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
            data={"analysis": analysis_result.model_dump()},
            visualization=visualization_model.model_dump() if visualization_model else None,
            next_agent=None,  # Analytics is the final agent
            state=state,
        )

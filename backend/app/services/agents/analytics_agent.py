"""Analytics Agent - Processes data and generates insights using LangGraph."""

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from .base_agent import (
    AgentRole,
    AgentResponse,
    AgentState,
    BaseAgent,
    GraphState,
)


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
        return """You are an Analytics Agent for Singapore government data analysis.

Your responsibilities:
1. Analyze extracted data to identify patterns and trends
2. Generate statistical insights and summaries
3. Provide clear, actionable interpretations of the data

IMPORTANT: When writing your analysis:
- Focus on insights, patterns, and conclusions in natural language
- DO NOT include JSON code blocks or structured data formats in your analysis text
- DO NOT include visualization specifications in your response text
- Describe what the data shows using clear, non-technical language
- The visualization will be handled separately by the system

Your analysis should be conversational and explanatory, NOT technical or code-like.

Examples:
Good: "The data shows employment rates increased by 15% from 2020 to 2023, with the strongest growth in the technology sector."
Bad: "```json\n{\"chart_type\": \"bar\", ...}\n```"
"""

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for analytics.

        Flow:
        prepare_data -> analyze_data -> generate_visualization -> compose_response -> END
        """
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("prepare_data", self._prepare_data_node)
        workflow.add_node("analyze_data", self._analyze_data_node)
        workflow.add_node("generate_visualization", self._generate_visualization_node)
        workflow.add_node("compose_response", self._compose_response_node)

        # Set entry point
        workflow.set_entry_point("prepare_data")

        # Add edges
        workflow.add_edge("prepare_data", "analyze_data")
        workflow.add_conditional_edges(
            "analyze_data",
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
        """Node: Prepare extracted data for analysis."""
        extracted_data = state.get("extracted_data", {})
        current_task = state.get("current_task", "")

        # Prepare data summary
        data_summary = {}
        total_rows = 0

        for name, data in extracted_data.items():
            if isinstance(data, dict):
                rows = data.get("data", [])
                columns = data.get("columns", [])

                # Calculate basic stats
                numeric_cols = []
                if rows:
                    for col in columns:
                        sample_val = rows[0].get(col)
                        if isinstance(sample_val, (int, float)):
                            numeric_cols.append(col)

                data_summary[name] = {
                    "row_count": len(rows),
                    "columns": columns,
                    "numeric_columns": numeric_cols,
                    "sample": rows[:3] if rows else [],
                }
                total_rows += len(rows)

        return {
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "data_summary": data_summary,
                "total_rows": total_rows,
                "has_data": total_rows > 0,
            },
        }

    async def _analyze_data_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Analyze data and generate insights."""
        current_task = state.get("current_task", "")
        extracted_data = state.get("extracted_data", {})
        data_summary = state.get("intermediate_results", {}).get("data_summary", {})

        # Build analysis prompt
        data_context = self._format_data_for_analysis(extracted_data, data_summary)

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
        """Node: Generate visualization specification."""
        extracted_data = state.get("extracted_data", {})
        current_task = state.get("current_task", "")
        analysis = state.get("intermediate_results", {}).get("analysis", "")

        # Try to auto-generate visualization first
        auto_viz = self._auto_generate_visualization(extracted_data)

        # Use LLM to enhance or create visualization
        if extracted_data:
            # Get sample data for LLM
            sample_data = None
            sample_dataset = None
            for name, data in extracted_data.items():
                if isinstance(data, dict) and data.get("data"):
                    sample_data = data["data"][:10]
                    sample_dataset = name
                    break

            if sample_data:
                viz_prompt = f"""Create a visualization for this data analysis.

Question: {current_task}
Dataset: {sample_dataset}
Sample Data: {json.dumps(sample_data[:5], default=str, indent=2)}

Create a JSON visualization specification with this EXACT format:
{{
    "chart_type": "bar|line|pie|scatter|table",
    "title": "Chart title",
    "data": [...prepared data...],
    "x_axis": "field_name_for_horizontal_axis_labels",
    "y_axis": "field_name_for_vertical_axis_values",
    "description": "What this shows"
}}

IMPORTANT axis mapping for bar/line charts:
- "x_axis": The field containing CATEGORY LABELS (e.g., "year", "industry", "month")
  → These will appear as labels on the HORIZONTAL axis
- "y_axis": The field containing NUMERIC VALUES (e.g., "count", "rate", "amount")
  → These will determine the HEIGHT of bars or points

Example for employment data:
- If data has {{"year": "2020", "employment_count": 1500}}, {{"year": "2021", "employment_count": 1700}}
- Use "x_axis": "year" (categories on horizontal axis)
- Use "y_axis": "employment_count" (values for bar heights)

Choose the most appropriate chart type for the data and question."""

                try:
                    response = await self._invoke_llm([HumanMessage(content=viz_prompt)])
                    viz = self._parse_visualization(response)
                    if viz:
                        return {
                            "intermediate_results": {
                                **state.get("intermediate_results", {}),
                                "visualization": viz,
                            },
                        }
                except Exception:
                    pass

        # Fall back to auto-generated visualization
        return {
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "visualization": auto_viz,
            },
        }

    async def _compose_response_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Compose final response with analysis and visualization."""
        analysis = state.get("intermediate_results", {}).get("analysis", "")
        visualization = state.get("intermediate_results", {}).get("visualization")
        extracted_data = state.get("extracted_data", {})
        current_task = state.get("current_task", "")

        # If no analysis yet, generate a simple response
        if not analysis:
            if extracted_data:
                analysis = "Data has been extracted and is ready for review."
            else:
                analysis = "I couldn't find specific data for your query. Please try rephrasing your question."

        # Strip JSON code blocks to ensure visualization field is single source of truth
        cleaned_analysis = self._strip_json_code_blocks(analysis)

        # Clean up the analysis text
        final_response = cleaned_analysis.strip()

        # Add data source attribution
        sources = list(extracted_data.keys())
        if sources:
            source_text = ", ".join(sources[:3])
            if len(sources) > 3:
                source_text += f", and {len(sources) - 3} more"
            final_response += f"\n\n*Data sources: {source_text}*"

        return {
            "analysis_results": {
                "text": final_response,
                "visualization": visualization,
                "data_sources": sources,
            },
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "final_response": final_response,
            },
            "messages": [AIMessage(content=final_response)],
            "current_step": state.get("current_step", 0) + 1,
        }

    def _format_data_for_analysis(
        self,
        extracted_data: Dict[str, Any],
        data_summary: Dict[str, Any],
    ) -> str:
        """Format extracted data for the analysis prompt."""
        if not extracted_data:
            return "No data available for analysis."

        parts = []
        for name, data in extracted_data.items():
            if not isinstance(data, dict):
                continue

            summary = data_summary.get(name, {})
            rows = data.get("data", [])

            part = f"\n### {name}\n"
            part += f"Rows: {summary.get('row_count', len(rows))}\n"
            part += f"Columns: {', '.join(data.get('columns', [])[:10])}\n"

            if rows:
                part += f"Data (first 10 rows):\n"
                part += json.dumps(rows[:10], indent=2, default=str)

            parts.append(part)

        return "\n".join(parts) if parts else "No structured data available."

    def _auto_generate_visualization(
        self,
        extracted_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Auto-generate visualization from extracted data."""
        if not extracted_data:
            return None

        for name, data in extracted_data.items():
            if not isinstance(data, dict):
                continue

            rows = data.get("data", [])
            if not rows:
                continue

            columns = list(rows[0].keys()) if rows else []
            if len(columns) < 2:
                continue

            # Find x (label/category) and y (numeric value) columns
            # x_col: categorical field for horizontal axis labels (e.g., "year", "industry")
            # y_col: numeric field for vertical axis values (e.g., "count", "rate")
            x_col = columns[0]  # First column is typically the category/label
            y_col = None

            # Find first numeric column for y-axis values
            for col in columns[1:]:
                sample = rows[0].get(col)
                if isinstance(sample, (int, float)):
                    y_col = col
                    break

            if not y_col:
                y_col = columns[1] if len(columns) > 1 else columns[0]

            # Prepare visualization data
            # Each data point must contain both x_col and y_col as keys
            viz_data = []
            for row in rows[:20]:
                viz_data.append({
                    x_col: str(row.get(x_col, ""))[:30],  # Category label (string)
                    y_col: row.get(y_col, 0),  # Numeric value
                })

            if viz_data:
                return {
                    "chart_type": "bar",
                    "title": f"{y_col} by {x_col}",
                    "data": viz_data,
                    "x_axis": x_col,  # Field name for horizontal axis labels
                    "y_axis": y_col,  # Field name for vertical axis values
                    "description": f"Data from {name}",
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

    def _build_response(self, result: GraphState, state: AgentState) -> AgentResponse:
        """Build AgentResponse from graph execution result."""
        analysis_results = result.get("analysis_results", {})
        final_response = result.get("intermediate_results", {}).get("final_response", "")
        visualization = result.get("intermediate_results", {}).get("visualization")

        # Update state
        state.analysis_results = analysis_results

        return AgentResponse(
            success=True,
            message=final_response or analysis_results.get("text", "Analysis complete."),
            data={
                "analysis": analysis_results,
            },
            visualization=visualization,
            next_agent=None,  # Analytics is the final agent
            state=state,
        )

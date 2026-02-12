"""Data Coordinator Agent - Plans research workflows using LangGraph."""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from app.models import (
    CoordinatorResult,
    DelegationInfo,
    WorkflowPlan,
    WorkflowStep,
)

from ..base_agent import (
    AgentResponse,
    AgentRole,
    AgentState,
    BaseAgent,
    GraphState,
)
from .prompts import SYSTEM_PROMPT


class DataCoordinatorAgent(BaseAgent):
    """Agent responsible for planning research workflows and delegating tasks.

    Uses LangGraph to construct a flow with nodes:
    1. analyze_query - Understand user's request
    2. identify_data_sources - Determine required datasets/APIs
    3. create_plan - Build workflow plan
    4. determine_delegation - Decide which agent handles next
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.COORDINATOR

    @property
    def name(self) -> str:
        return "Data Coordinator"

    @property
    def description(self) -> str:
        return "Plans research workflows and delegates tasks to specialized agents"

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for the coordinator.

        Flow:
        analyze_query -> identify_data_sources -> create_plan -> determine_delegation -> END
        """
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("identify_data_sources", self._identify_data_sources_node)
        workflow.add_node("create_plan", self._create_plan_node)
        workflow.add_node("determine_delegation", self._determine_delegation_node)

        # Set entry point
        workflow.set_entry_point("analyze_query")

        # Add edges
        workflow.add_edge("analyze_query", "identify_data_sources")
        workflow.add_edge("identify_data_sources", "create_plan")
        workflow.add_edge("create_plan", "determine_delegation")
        workflow.add_edge("determine_delegation", END)

        return workflow.compile()

    async def _analyze_query_node(self, state: GraphState) -> dict[str, Any]:
        """Node: Analyze the user's query to understand intent."""
        messages = state.get("messages", [])

        # Get the latest user message
        user_message = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break

        if not user_message:
            return {
                "errors": state.get("errors", []) + ["No user message found"],
                "should_continue": False,
            }

        # Analyze query with LLM
        analysis_prompt = f"""Analyze this user query and identify:
1. Main intent (what does the user want to know?)
2. Data type needed (employment, income, weather, etc.)
3. Time scope (if mentioned)
4. Any specific filters or conditions

User Query: {user_message}

Respond in JSON format:
{{
    "intent": "brief description of intent",
    "data_type": "category of data needed",
    "time_scope": "time range if mentioned or null",
    "filters": ["list of filters"],
    "complexity": "simple|moderate|complex"
}}"""

        try:
            response = await self._invoke_llm([HumanMessage(content=analysis_prompt)])
            analysis = self._parse_json_response(response)
        except Exception:
            analysis = {
                "intent": user_message,
                "data_type": "general",
                "complexity": "moderate",
            }

        return {
            "current_task": user_message,
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "query_analysis": analysis,
            },
            "metadata": {
                **state.get("metadata", {}),
                "query_analysis": analysis,
            },
        }

    async def _identify_data_sources_node(self, state: GraphState) -> dict[str, Any]:
        """Node: Identify required data sources based on query analysis."""
        analysis = state.get("intermediate_results", {}).get("query_analysis", {})
        current_task = state.get("current_task", "")

        # Try to get richer dataset descriptions from PostgreSQL
        datasets_info = await self._get_datasets_info_from_db()

        if not datasets_info:
            # Fallback to filesystem-based listing
            from app.services.data_service import data_service

            available_datasets = data_service.get_available_datasets()
            datasets_info = "\n".join(
                [f"- {ds['name']} ({ds['path']})" for ds in available_datasets[:15]]
            )

        source_prompt = f"""Based on this query analysis, identify which data sources are needed.

Query: {current_task}
Analysis: {json.dumps(analysis, indent=2)}

Available datasets:
{datasets_info}

Respond in JSON format:
{{
    "required_datasets": ["list of dataset paths that are relevant"],
    "reasoning": "why these datasets were chosen",
    "data_gaps": "any data that might not be available"
}}"""

        try:
            response = await self._invoke_llm([HumanMessage(content=source_prompt)])
            sources = self._parse_json_response(response)
        except Exception:
            sources = {
                "required_datasets": [],
                "reasoning": "Unable to identify specific datasets",
            }

        return {
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "data_sources": sources,
            },
            "metadata": {
                **state.get("metadata", {}),
                "required_data": sources.get("required_datasets", []),
            },
        }

    async def _create_plan_node(self, state: GraphState) -> dict[str, Any]:
        """Node: Create execution workflow plan."""
        analysis = state.get("intermediate_results", {}).get("query_analysis", {})
        sources = state.get("intermediate_results", {}).get("data_sources", {})
        current_task = state.get("current_task", "")

        plan_prompt = f"""Create a workflow plan to answer this query.

Query: {current_task}
Analysis: {json.dumps(analysis, indent=2)}
Data Sources: {json.dumps(sources, indent=2)}

Create a step-by-step plan. Respond in JSON format:
{{
    "query_understanding": "brief interpretation of user's request",
    "steps": [
        {{
            "step": 1,
            "agent": "extraction",
            "task": "what this step should do",
            "inputs": ["required inputs"],
            "expected_output": "what this step produces"
        }},
        {{
            "step": 2,
            "agent": "analytics",
            "task": "analyze extracted data",
            "inputs": ["data from step 1"],
            "expected_output": "analysis and insights"
        }}
    ],
    "final_output": "description of expected final response",
    "visualization_suggested": true/false
}}"""

        try:
            response = await self._invoke_llm([HumanMessage(content=plan_prompt)])
            plan = self._parse_json_response(response)
        except Exception:
            # Default plan
            plan = {
                "query_understanding": current_task,
                "steps": [
                    {
                        "step": 1,
                        "agent": "extraction",
                        "task": "Extract relevant data",
                        "inputs": ["user_query"],
                        "expected_output": "raw data",
                    },
                    {
                        "step": 2,
                        "agent": "analytics",
                        "task": "Analyze and respond",
                        "inputs": ["extracted_data"],
                        "expected_output": "analysis response",
                    },
                ],
                "final_output": "Analysis response with visualization if applicable",
            }

        return {
            "workflow_plan": plan.get("steps", []),
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "full_plan": plan,
            },
            "metadata": {
                **state.get("metadata", {}),
                "final_output": plan.get("final_output", ""),
                "visualization_suggested": plan.get("visualization_suggested", True),
            },
        }

    async def _determine_delegation_node(self, state: GraphState) -> dict[str, Any]:
        """Node: Determine which agent should handle the task next."""
        workflow_plan = state.get("workflow_plan", [])
        full_plan = state.get("intermediate_results", {}).get("full_plan", {})

        # Determine next agent from plan
        next_agent_str = "extraction"  # Default
        if workflow_plan:
            first_step = workflow_plan[0]
            next_agent_str = first_step.get("agent", "extraction")

        # Build summary message
        summary = full_plan.get("query_understanding", state.get("current_task", ""))

        return {
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "delegation": {
                    "next_agent": next_agent_str,
                    "summary": summary,
                },
            },
            "messages": [AIMessage(content=f"Workflow planned: {summary}")],
        }

    async def _get_datasets_info_from_db(self) -> str:
        """Get dataset descriptions from PostgreSQL if available."""
        try:
            from app.db.session import async_session_factory

            if async_session_factory is None:
                return ""

            from sqlalchemy import select

            from app.db.models import DatasetMetadata
            from app.db.session import get_db

            async with get_db() as session:
                result = await session.execute(
                    select(DatasetMetadata).order_by(DatasetMetadata.category)
                )
                rows = result.scalars().all()

                if not rows:
                    return ""

                lines = []
                for row in rows:
                    year_info = ""
                    yr = row.year_range or {}
                    if yr.get("min") and yr.get("max"):
                        year_info = f", years {yr['min']}-{yr['max']}"
                    lines.append(
                        f"- {row.dataset_name} [{row.category}] "
                        f"({row.row_count} rows{year_info}): "
                        f"{(row.description or '')[:150]}"
                    )
                return "\n".join(lines)

        except Exception:
            return ""

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from LLM response."""
        try:
            # Try to find JSON block
            if "```json" in response:
                start = response.index("```json") + 7
                end = response.index("```", start)
                return json.loads(response[start:end].strip())
            elif "```" in response:
                start = response.index("```") + 3
                end = response.index("```", start)
                return json.loads(response[start:end].strip())
            elif "{" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                return json.loads(response[start:end])
        except (ValueError, json.JSONDecodeError):
            pass
        return {}

    def _build_response(self, result: GraphState, state: AgentState) -> AgentResponse:
        """Build AgentResponse from graph execution result."""
        delegation_dict = result.get("intermediate_results", {}).get("delegation", {})
        full_plan_dict = result.get("intermediate_results", {}).get("full_plan", {})

        next_agent_str = delegation_dict.get("next_agent", "extraction")
        next_agent = AgentRole.EXTRACTION
        if next_agent_str == "analytics":
            next_agent = AgentRole.ANALYTICS

        summary = delegation_dict.get("summary", state.current_task)

        # Create Pydantic models for type safety and validation
        workflow_steps = []
        for step_dict in full_plan_dict.get("steps", []):
            workflow_steps.append(
                WorkflowStep(
                    step=step_dict.get("step", 1),
                    agent=step_dict.get("agent", "extraction"),
                    task=step_dict.get("task", ""),
                    inputs=step_dict.get("inputs", []),
                    expected_output=step_dict.get("expected_output", ""),
                )
            )

        workflow_plan = WorkflowPlan(
            query_understanding=full_plan_dict.get("query_understanding", state.current_task),
            steps=workflow_steps,
            final_output=full_plan_dict.get("final_output", "Analysis response"),
            visualization_suggested=full_plan_dict.get("visualization_suggested", False),
        )

        delegation = DelegationInfo(
            next_agent=next_agent_str,
            summary=summary,
        )

        coordinator_result = CoordinatorResult(
            workflow=workflow_plan,
            delegation=delegation,
        )

        return AgentResponse(
            success=True,
            message=f"Workflow planned: {summary}",
            data=coordinator_result.model_dump(),  # Convert to dict for GraphState
            next_agent=next_agent,
            state=state,
        )

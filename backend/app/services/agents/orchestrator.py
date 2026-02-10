"""Agent Orchestrator - Manages the multi-agent LangGraph workflow."""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from .base_agent import AgentRole, AgentResponse, AgentState, BaseAgent, GraphState
from .coordinator_agent import DataCoordinatorAgent
from .extraction_agent import DataExtractionAgent
from .analytics_agent import AnalyticsAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrates the execution of multiple LangGraph-based agents.

    The Orchestrator manages the flow between agents:
    1. Coordinator Agent - Plans the workflow
    2. Extraction Agent - Extracts data
    3. Analytics Agent - Analyzes and generates response

    Each agent has its own internal LangGraph workflow.
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        max_iterations: int = 10,
    ):
        """Initialize the orchestrator with agent instances.

        Args:
            llm_provider: LLM provider for all agents
            llm_model: LLM model for all agents
            max_iterations: Maximum number of agent iterations
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.max_iterations = max_iterations

        # Initialize agents
        self.agents: Dict[AgentRole, BaseAgent] = {
            AgentRole.COORDINATOR: DataCoordinatorAgent(
                llm_provider=llm_provider,
                llm_model=llm_model,
            ),
            AgentRole.EXTRACTION: DataExtractionAgent(
                llm_provider=llm_provider,
                llm_model=llm_model,
            ),
            AgentRole.ANALYTICS: AnalyticsAgent(
                llm_provider=llm_provider,
                llm_model=llm_model,
            ),
        }

        # Build orchestration graph
        self._orchestration_graph = self._build_orchestration_graph()

    def _build_orchestration_graph(self) -> StateGraph:
        """Build the top-level orchestration graph.

        Flow:
        coordinator -> route_after_coordinator -> extraction/analytics ->
        route_after_extraction -> analytics -> END
        """
        workflow = StateGraph(GraphState)

        # Add agent nodes
        workflow.add_node("coordinator", self._run_coordinator)
        workflow.add_node("extraction", self._run_extraction)
        workflow.add_node("analytics", self._run_analytics)

        # Set entry point
        workflow.set_entry_point("coordinator")

        # Add conditional routing after coordinator
        workflow.add_conditional_edges(
            "coordinator",
            self._route_after_coordinator,
            {
                "extraction": "extraction",
                "analytics": "analytics",
            }
        )

        # Add conditional routing after extraction
        workflow.add_conditional_edges(
            "extraction",
            self._route_after_extraction,
            {
                "analytics": "analytics",
                "end": END,
            }
        )

        # Analytics always ends
        workflow.add_edge("analytics", END)

        return workflow.compile()

    def _route_after_coordinator(self, state: GraphState) -> str:
        """Determine next agent after coordinator."""
        metadata = state.get("metadata", {})
        workflow_plan = state.get("workflow_plan", [])

        # Check if there's a workflow plan
        if workflow_plan:
            first_step = workflow_plan[0]
            agent = first_step.get("agent", "extraction")
            if agent == "analytics":
                logger.info("Routing: coordinator -> analytics")
                return "analytics"

        logger.info("Routing: coordinator -> extraction")
        return "extraction"

    def _route_after_extraction(self, state: GraphState) -> str:
        """Determine next agent after extraction."""
        # Check for errors that should stop execution
        errors = state.get("errors", [])
        if len(errors) > 3:
            logger.warning(f"Too many errors ({len(errors)}), stopping workflow")
            return "end"

        logger.info("Routing: extraction -> analytics")
        return "analytics"

    async def _run_coordinator(self, state: GraphState) -> Dict[str, Any]:
        """Run the coordinator agent."""
        logger.info("=" * 60)
        logger.info("Running COORDINATOR AGENT")
        logger.info("=" * 60)
        agent = self.agents[AgentRole.COORDINATOR]
        agent_state = AgentState.from_graph_state(state)

        # Execute the agent's internal graph
        response = await agent.execute(agent_state)
        logger.info(f"Coordinator result: success={response.success}, message={response.message[:100] if response.message else 'N/A'}...")

        # Merge response state back
        if response.state:
            new_state = response.state.to_graph_state()
            return {
                "workflow_plan": new_state.get("workflow_plan", []),
                "metadata": {
                    **state.get("metadata", {}),
                    **new_state.get("metadata", {}),
                },
                "intermediate_results": {
                    **state.get("intermediate_results", {}),
                    "coordinator_response": response.to_dict(),
                },
                "current_task": new_state.get("current_task") or state.get("current_task"),
            }

        return {}

    async def _run_extraction(self, state: GraphState) -> Dict[str, Any]:
        """Run the extraction agent."""
        logger.info("=" * 60)
        logger.info("Running EXTRACTION AGENT")
        logger.info("=" * 60)
        agent = self.agents[AgentRole.EXTRACTION]
        agent_state = AgentState.from_graph_state(state)

        # Execute the agent's internal graph
        response = await agent.execute(agent_state)
        logger.info(f"Extraction result: success={response.success}, data_sources={len(state.get('extracted_data', {}))}")

        # Merge response state back
        if response.state:
            new_state = response.state.to_graph_state()
            return {
                "extracted_data": new_state.get("extracted_data", {}),
                "current_step": new_state.get("current_step", 0),
                "errors": list(set(state.get("errors", []) + new_state.get("errors", []))),
                "intermediate_results": {
                    **state.get("intermediate_results", {}),
                    "extraction_response": response.to_dict(),
                },
            }

        return {}

    async def _run_analytics(self, state: GraphState) -> Dict[str, Any]:
        """Run the analytics agent."""
        logger.info("=" * 60)
        logger.info("Running ANALYTICS AGENT")
        logger.info("=" * 60)
        agent = self.agents[AgentRole.ANALYTICS]
        agent_state = AgentState.from_graph_state(state)

        # Execute the agent's internal graph
        response = await agent.execute(agent_state)
        logger.info(f"Analytics result: success={response.success}, has_visualization={response.visualization is not None}")

        # Merge response state back
        result = {
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "analytics_response": response.to_dict(),
                "final_message": response.message,
                "visualization": response.visualization,
            },
        }

        if response.state:
            new_state = response.state.to_graph_state()
            result["analysis_results"] = new_state.get("analysis_results", {})
            result["current_step"] = new_state.get("current_step", 0)

        return result

    async def execute(
        self,
        message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Execute the multi-agent workflow for a user message.

        Args:
            message: User's message
            chat_history: Previous conversation history

        Returns:
            Dictionary with response, visualization, and metadata
        """
        # Build initial state
        messages = []
        if chat_history:
            for msg in chat_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))

        messages.append(HumanMessage(content=message))

        initial_state: GraphState = {
            "messages": messages,
            "current_task": message,
            "extracted_data": {},
            "analysis_results": {},
            "workflow_plan": [],
            "current_step": 0,
            "errors": [],
            "metadata": {},
            "intermediate_results": {},
            "should_continue": True,
            "retrieval_context": {},
        }

        try:
            logger.info("\n" + "=" * 80)
            logger.info("STARTING MULTI-AGENT WORKFLOW")
            logger.info(f"User message: {message}")
            logger.info("=" * 80 + "\n")

            # Run the orchestration graph
            result = await self._orchestration_graph.ainvoke(initial_state)

            logger.info("\n" + "=" * 80)
            logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
            logger.info("=" * 80 + "\n")

            # Extract final response
            intermediate = result.get("intermediate_results", {})
            analytics_response = intermediate.get("analytics_response", {})

            final_message = intermediate.get("final_message", "")
            if not final_message:
                final_message = analytics_response.get("message", "I processed your request.")

            visualization = intermediate.get("visualization")
            if not visualization:
                visualization = analytics_response.get("visualization")

            # Build agent trace
            agent_trace = []
            if intermediate.get("coordinator_response"):
                agent_trace.append({
                    "agent": "Data Coordinator",
                    "success": intermediate["coordinator_response"].get("success", True),
                })
            if intermediate.get("extraction_response"):
                agent_trace.append({
                    "agent": "Data Extraction",
                    "success": intermediate["extraction_response"].get("success", True),
                })
            if intermediate.get("analytics_response"):
                agent_trace.append({
                    "agent": "Analytics",
                    "success": intermediate["analytics_response"].get("success", True),
                })

            return {
                "message": final_message,
                "visualization": visualization,
                "sources": list(result.get("extracted_data", {}).keys()),
                "error": result.get("errors", [])[-1] if result.get("errors") else None,
                "agent_trace": agent_trace,
                "metadata": {
                    "iterations": len(agent_trace),
                    "workflow_plan": result.get("workflow_plan", []),
                    "agents_used": [t["agent"] for t in agent_trace],
                },
            }

        except Exception as e:
            return {
                "message": f"An error occurred while processing your request: {str(e)}",
                "visualization": None,
                "sources": [],
                "error": str(e),
                "agent_trace": [],
                "metadata": {},
            }

    def get_agent(self, role: AgentRole) -> BaseAgent:
        """Get an agent by role."""
        return self.agents[role]

    def get_agent_info(self) -> List[Dict[str, str]]:
        """Get information about all available agents."""
        return [
            {
                "role": agent.role.value,
                "name": agent.name,
                "description": agent.description,
            }
            for agent in self.agents.values()
        ]


def get_orchestrator(
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> AgentOrchestrator:
    """Factory function to create an orchestrator."""
    return AgentOrchestrator(
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

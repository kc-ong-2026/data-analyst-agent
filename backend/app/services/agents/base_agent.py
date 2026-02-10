"""Abstract base class for all agents using LangGraph."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Annotated, Dict, List, Optional, TypedDict, Union
import operator

from pydantic import BaseModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END


class AgentRole(Enum):
    """Enum defining agent roles in the system."""
    VERIFICATION = "verification"
    COORDINATOR = "coordinator"
    EXTRACTION = "extraction"
    ANALYTICS = "analytics"


class GraphState(TypedDict):
    """Base state for LangGraph nodes."""
    messages: Annotated[List[BaseMessage], operator.add]
    current_task: str
    extracted_data: Dict[str, Any]
    analysis_results: Dict[str, Any]
    workflow_plan: List[Dict[str, Any]]
    current_step: int
    errors: List[str]
    metadata: Dict[str, Any]
    intermediate_results: Dict[str, Any]
    should_continue: bool
    retrieval_context: Dict[str, Any]  # Now includes table_schemas for SQL generation
    sql_queries: List[Dict[str, Any]]  # Generated SQL queries [{sql, params, table_name}]
    sql_results: Dict[str, List[Dict[str, Any]]]  # Execution results {table_name: [rows]}
    query_validation: Dict[str, Any]  # Validation result and context
    available_years: Dict[str, Dict[str, int]]  # Year ranges by category


@dataclass
class AgentState:
    """State shared between agents during task execution."""

    messages: List[BaseMessage] = field(default_factory=list)
    current_task: Optional[str] = None
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    workflow_plan: List[Dict[str, Any]] = field(default_factory=list)
    current_step: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the state."""
        if role == "user":
            self.messages.append(HumanMessage(content=content))
        elif role == "assistant":
            self.messages.append(AIMessage(content=content))
        elif role == "system":
            self.messages.append(SystemMessage(content=content))

    def add_error(self, error: str) -> None:
        """Add an error to the state."""
        self.errors.append(error)

    def to_graph_state(self) -> GraphState:
        """Convert to LangGraph state."""
        return {
            "messages": self.messages.copy(),
            "current_task": self.current_task or "",
            "extracted_data": self.extracted_data.copy(),
            "analysis_results": self.analysis_results.copy(),
            "workflow_plan": self.workflow_plan.copy(),
            "current_step": self.current_step,
            "errors": self.errors.copy(),
            "metadata": self.metadata.copy(),
            "intermediate_results": {},
            "should_continue": True,
            "retrieval_context": {},
            "sql_queries": [],
            "sql_results": {},
            "query_validation": {},
            "available_years": {},
        }

    @classmethod
    def from_graph_state(cls, graph_state: GraphState) -> "AgentState":
        """Create AgentState from LangGraph state."""
        state = cls()
        state.messages = list(graph_state.get("messages", []))
        state.current_task = graph_state.get("current_task")
        state.extracted_data = dict(graph_state.get("extracted_data", {}))
        state.analysis_results = dict(graph_state.get("analysis_results", {}))
        state.workflow_plan = list(graph_state.get("workflow_plan", []))
        state.current_step = graph_state.get("current_step", 0)
        state.errors = list(graph_state.get("errors", []))
        state.metadata = dict(graph_state.get("metadata", {}))
        return state

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "current_task": self.current_task,
            "extracted_data": self.extracted_data,
            "analysis_results": self.analysis_results,
            "workflow_plan": self.workflow_plan,
            "current_step": self.current_step,
            "errors": self.errors,
            "metadata": self.metadata,
        }


@dataclass
class AgentResponse:
    """Response from an agent execution.

    Supports both Pydantic models and dictionaries for data/visualization fields.
    This provides backward compatibility during migration to Pydantic models.
    """

    success: bool
    message: str
    data: Union[BaseModel, Dict[str, Any]] = field(default_factory=dict)
    visualization: Optional[Union[BaseModel, Dict[str, Any]]] = None
    next_agent: Optional[AgentRole] = None
    state: Optional[AgentState] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary, handling Pydantic models."""
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data.model_dump() if isinstance(self.data, BaseModel) else self.data,
            "visualization": (
                self.visualization.model_dump()
                if isinstance(self.visualization, BaseModel)
                else self.visualization
            ),
            "next_agent": self.next_agent.value if self.next_agent else None,
        }


class BaseAgent(ABC):
    """Abstract base class for all agents using LangGraph.

    All agents must inherit from this class and implement the required methods.
    Each agent constructs its own LangGraph workflow with nodes and edges.
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        """Initialize the base agent.

        Args:
            llm_provider: The LLM provider to use (openai, anthropic, google)
            llm_model: The specific model to use
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self._llm = None
        self._graph = None

    @property
    @abstractmethod
    def role(self) -> AgentRole:
        """Return the role of this agent."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this agent."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what this agent does."""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass

    def get_llm(self):
        """Get the LLM instance for this agent."""
        if self._llm is None:
            from app.services.llm_service import get_llm_service
            llm_service = get_llm_service(
                provider=self.llm_provider,
                model=self.llm_model
            )
            self._llm = llm_service.llm
        return self._llm

    @abstractmethod
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for this agent.

        Returns:
            Compiled StateGraph for the agent's workflow
        """
        pass

    @property
    def graph(self) -> StateGraph:
        """Get the compiled graph, building it if necessary."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    async def execute(self, state: AgentState) -> AgentResponse:
        """Execute the agent's LangGraph workflow.

        Args:
            state: The current agent state

        Returns:
            AgentResponse with the result of execution
        """
        try:
            # Convert to graph state
            graph_state = state.to_graph_state()

            # Run the graph
            result = await self.graph.ainvoke(graph_state)

            # Convert back to AgentState
            updated_state = AgentState.from_graph_state(result)

            # Build response from result
            return self._build_response(result, updated_state)

        except Exception as e:
            state.add_error(f"{self.name} error: {str(e)}")
            return AgentResponse(
                success=False,
                message=f"Agent execution failed: {str(e)}",
                state=state,
            )

    @abstractmethod
    def _build_response(self, result: GraphState, state: AgentState) -> AgentResponse:
        """Build AgentResponse from graph execution result.

        Args:
            result: The final graph state
            state: The updated AgentState

        Returns:
            AgentResponse with appropriate data
        """
        pass

    async def _invoke_llm(
        self,
        messages: List[BaseMessage],
        include_system_prompt: bool = True,
    ) -> str:
        """Invoke the LLM with messages.

        Args:
            messages: List of messages to send to the LLM
            include_system_prompt: Whether to include the system prompt

        Returns:
            The LLM response content
        """
        all_messages = []

        if include_system_prompt:
            all_messages.append(SystemMessage(content=self.system_prompt))

        all_messages.extend(messages)

        llm = self.get_llm()
        response = await llm.ainvoke(all_messages)
        return response.content

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(role={self.role.value}, name={self.name})>"

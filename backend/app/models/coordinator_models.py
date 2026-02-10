"""Models for the Data Coordinator Agent."""

from typing import List

from pydantic import BaseModel, Field


class WorkflowStep(BaseModel):
    """A single step in the workflow plan."""

    step: int = Field(..., ge=1, description="Step number in the workflow")
    agent: str = Field(..., description="Agent responsible for this step")
    task: str = Field(..., description="Task description for this step")
    inputs: List[str] = Field(
        default_factory=list, description="Required inputs for this step"
    )
    expected_output: str = Field(..., description="Expected output from this step")


class WorkflowPlan(BaseModel):
    """Complete workflow plan for handling a user query."""

    query_understanding: str = Field(
        ..., description="Interpretation of the user's request"
    )
    steps: List[WorkflowStep] = Field(
        ..., description="Ordered list of workflow steps"
    )
    final_output: str = Field(..., description="Description of expected final response")
    visualization_suggested: bool = Field(
        default=False, description="Whether a visualization should be generated"
    )


class DelegationInfo(BaseModel):
    """Information about task delegation to the next agent."""

    next_agent: str = Field(
        ..., description="Name of the next agent to handle the task"
    )
    summary: str = Field(..., description="Summary of the task for the next agent")


class CoordinatorResult(BaseModel):
    """Complete result from the coordinator agent."""

    workflow: WorkflowPlan = Field(..., description="The created workflow plan")
    delegation: DelegationInfo = Field(..., description="Delegation information")

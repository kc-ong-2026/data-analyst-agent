"""Chat API routes with multi-agent system."""

import uuid
from typing import Dict, List

from fastapi import APIRouter, HTTPException

from app.models import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    VisualizationData,
)
from app.services.agents.orchestrator import get_orchestrator

router = APIRouter(prefix="/chat", tags=["chat"])

# In-memory conversation storage (use Redis/DB in production)
conversations: Dict[str, List[ChatMessage]] = {}


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message using multi-agent system.

    The request flows through:
    1. Data Coordinator Agent - Plans the workflow
    2. Data Extraction Agent - Extracts relevant data
    3. Analytics Agent - Analyzes data and generates response
    """
    try:
        # Get or create conversation
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Get chat history
        chat_history = []
        if conversation_id in conversations:
            chat_history = [
                {"role": msg.role, "content": msg.content}
                for msg in conversations[conversation_id]
            ]

        # Initialize agents through orchestrator
        orchestrator = get_orchestrator(
            llm_provider=request.llm_provider,
            llm_model=request.llm_model,
        )

        # Execute multi-agent workflow
        result = await orchestrator.execute(
            message=request.message,
            chat_history=chat_history,
        )

        if result.get("error") and not result.get("message"):
            raise HTTPException(status_code=500, detail=result["error"])

        # Store conversation
        if conversation_id not in conversations:
            conversations[conversation_id] = []

        conversations[conversation_id].append(
            ChatMessage(role="user", content=request.message)
        )
        conversations[conversation_id].append(
            ChatMessage(role="assistant", content=result["message"])
        )

        # Prepare visualization data
        visualization = None
        if request.include_visualization and result.get("visualization"):
            viz = result["visualization"]
            visualization = VisualizationData(
                chart_type=viz.get("chart_type", "bar"),
                title=viz.get("title", ""),
                data=viz.get("data", []),
                x_axis=viz.get("x_axis"),
                y_axis=viz.get("y_axis"),
                description=viz.get("description"),
            )

        return ChatResponse(
            message=result["message"],
            conversation_id=conversation_id,
            visualization=visualization,
            sources=result.get("sources"),
            metadata={
                "agents_used": result.get("metadata", {}).get("agents_used", []),
                "iterations": result.get("metadata", {}).get("iterations", 0),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents")
async def list_agents() -> Dict:
    """List all available agents in the system."""
    orchestrator = get_orchestrator()
    return {
        "agents": orchestrator.get_agent_info(),
        "workflow": [
            {
                "step": 1,
                "agent": "Data Coordinator",
                "description": "Plans research workflows and delegates tasks",
            },
            {
                "step": 2,
                "agent": "Data Extraction",
                "description": "Extracts data from government sources",
            },
            {
                "step": 3,
                "agent": "Analytics",
                "description": "Processes data and generates insights",
            },
        ],
    }


@router.get("/history/{conversation_id}")
async def get_conversation_history(conversation_id: str) -> Dict:
    """Get conversation history."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "conversation_id": conversation_id,
        "messages": [
            {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp}
            for msg in conversations[conversation_id]
        ],
    }


@router.delete("/history/{conversation_id}")
async def clear_conversation(conversation_id: str) -> Dict:
    """Clear conversation history."""
    if conversation_id in conversations:
        del conversations[conversation_id]

    return {"status": "cleared", "conversation_id": conversation_id}


@router.get("/conversations")
async def list_conversations() -> Dict:
    """List all conversation IDs."""
    return {
        "conversations": list(conversations.keys()),
        "count": len(conversations),
    }

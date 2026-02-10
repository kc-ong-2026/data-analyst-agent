"""Chat API routes with LangGraph checkpoint support."""

import logging
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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# In-memory conversation storage (use Redis/DB in production)
conversations: Dict[str, List[ChatMessage]] = {}


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message with LangGraph checkpoints for human-in-the-loop.

    Uses LangGraph's native checkpoint system:
    - If graph hits interrupt(), it pauses and waits for input
    - User sends next message with SAME conversation_id
    - Graph automatically resumes from interrupt point!

    No manual checkpoint detection needed - LangGraph handles it all!
    """
    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())
        logger.info(f"Processing chat request: conversation_id={conversation_id}")

        if not request.message:
            raise HTTPException(status_code=400, detail="Message is required")

        logger.info(f"Message: {request.message[:100]}...")

        # Get chat history
        chat_history = []
        if conversation_id in conversations:
            chat_history = [
                {"role": msg.role, "content": msg.content}
                for msg in conversations[conversation_id]
            ]
            logger.info(f"Loaded {len(chat_history)} previous messages")

        # LangGraph config with thread_id for checkpoint support
        # Same thread_id = automatic resume if graph was interrupted
        config = {
            "configurable": {
                "thread_id": conversation_id  # This is the magic!
            }
        }

        # Initialize orchestrator
        logger.info("Initializing orchestrator and agents...")
        orchestrator = get_orchestrator()

        # Execute multi-agent workflow with LangGraph checkpoints
        # If graph hits interrupt(), it will pause and return
        # Next call with same thread_id will automatically resume!
        logger.info("Starting multi-agent workflow execution...")
        result = await orchestrator.execute(
            message=request.message,
            chat_history=chat_history,
            config=config,  # Pass config with thread_id
        )
        logger.info(f"Workflow completed. Agents used: {result.get('metadata', {}).get('agents_used', [])}")

        # Check if verification failed (topic invalid - not pausable)
        validation = result.get("query_validation", {})
        if validation and not validation.get("valid", True):
            # Return validation error message directly
            return ChatResponse(
                message=validation.get("reason", "Query validation failed"),
                conversation_id=conversation_id,
                visualization=None,
                sources=[],
                metadata={
                    "validation_failed": True,
                    "validation_details": validation,
                    "agents_used": result.get("metadata", {}).get("agents_used", []),
                }
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
                x_label=viz.get("x_label"),
                y_label=viz.get("y_label"),
                description=viz.get("description"),
                html_chart=viz.get("html_chart"),  # Include HTML chart!
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
                "agent": "Query Verification",
                "description": "Validates query topic and year specification",
            },
            {
                "step": 2,
                "agent": "Data Coordinator",
                "description": "Plans research workflows and delegates tasks",
            },
            {
                "step": 3,
                "agent": "Data Extraction",
                "description": "Extracts data from government sources",
            },
            {
                "step": 4,
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

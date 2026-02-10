"""Chat API routes with simple context append for natural conversation."""

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
    """Process a chat message with smart context append for clarifications.

    Simple approach: If last message was asking for clarification (like year),
    combine user's response with original query for full context.
    """
    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())
        logger.info(f"Processing chat request: conversation_id={conversation_id}")

        if not request.message:
            raise HTTPException(status_code=400, detail="Message is required")

        logger.info(f"Message: {request.message[:100]}...")

        # CONTEXT APPEND: Check if this is a clarification response
        combined_message = request.message

        if conversation_id in conversations and len(conversations[conversation_id]) >= 2:
            # Get last assistant message and the user message before it
            last_assistant_msg = None
            original_user_msg = None

            for i in range(len(conversations[conversation_id]) - 1, -1, -1):
                msg = conversations[conversation_id][i]
                if msg.role == "assistant" and last_assistant_msg is None:
                    last_assistant_msg = msg.content.lower()
                elif msg.role == "user" and last_assistant_msg is not None and original_user_msg is None:
                    original_user_msg = msg.content
                    break

            # Check if assistant was asking for clarification
            if last_assistant_msg and original_user_msg:
                clarification_keywords = [
                    "which year", "specify year", "year range", "what year",
                    "available data", "please specify", "interested in", "provide"
                ]
                is_asking_for_clarification = any(kw in last_assistant_msg for kw in clarification_keywords)

                if is_asking_for_clarification:
                    # Combine messages for full context!
                    combined_message = f"{original_user_msg} {request.message}"
                    logger.info(f"🔗 Context Append Detected!")
                    logger.info(f"   Original query: '{original_user_msg}'")
                    logger.info(f"   User response: '{request.message}'")
                    logger.info(f"   Combined query: '{combined_message}'")

        # Get chat history - DISABLED to make each query independent
        # Each new query should NOT have access to previous conversation context
        chat_history = []
        logger.info("Each query processes independently without previous context")

        # Initialize orchestrator and execute
        logger.info("Initializing orchestrator and agents...")
        orchestrator = get_orchestrator()

        logger.info("Starting multi-agent workflow execution...")
        result = await orchestrator.execute(
            message=combined_message,  # Use combined message!
            chat_history=chat_history,  # Always empty - independent queries
        )
        logger.info(f"Workflow completed. Agents used: {result.get('metadata', {}).get('agents_used', [])}")

        # Check if verification failed (topic invalid) BEFORE storing conversation
        validation = result.get("query_validation", {})
        if validation and not validation.get("valid", True):
            # Clear conversation history to prevent context pollution
            if conversation_id in conversations:
                logger.info(f"🗑️ Clearing conversation history for {conversation_id} due to validation failure")
                del conversations[conversation_id]

            # Return validation error without storing messages
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

        # Store conversation AFTER validation check (only if query is valid)
        if conversation_id not in conversations:
            conversations[conversation_id] = []

        conversations[conversation_id].append(
            ChatMessage(role="user", content=request.message)  # Store original message
        )
        conversations[conversation_id].append(
            ChatMessage(role="assistant", content=result["message"])
        )

        if result.get("error") and not result.get("message"):
            raise HTTPException(status_code=500, detail=result["error"])

        # Prepare visualization
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
                "context_combined": combined_message != request.message,  # Show if we combined
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents")
async def list_agents() -> Dict:
    """List all available agents."""
    orchestrator = get_orchestrator()
    return {
        "agents": orchestrator.get_agent_info(),
        "workflow": [
            {"step": 1, "agent": "Data Coordinator", "description": "Plans research workflows"},
            {"step": 2, "agent": "Data Extraction", "description": "Extracts data from sources"},
            {"step": 3, "agent": "Analytics", "description": "Analyzes and generates insights"},
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

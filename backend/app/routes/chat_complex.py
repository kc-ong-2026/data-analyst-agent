"""Chat API routes with multi-agent system."""

import logging
import uuid

from fastapi import APIRouter, HTTPException

from app.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    VisualizationData,
)
from app.services.agents.orchestrator import get_orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# In-memory conversation storage (use Redis/DB in production)
conversations: dict[str, list[ChatMessage]] = {}


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message or resume from checkpoint.

    The request flows through:
    1. Query Verification Agent - Validates query topic and year specification
    2. Data Coordinator Agent - Plans the workflow
    3. Data Extraction Agent - Extracts relevant data
    4. Analytics Agent - Analyzes data and generates response
    """
    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())

        logger.info(f"Processing chat request: conversation_id={conversation_id}")

        # Handle checkpoint resumption
        if request.checkpoint_id and request.user_input:
            logger.info(f"Resuming from checkpoint: {request.checkpoint_id}")

            orchestrator = get_orchestrator()
            result = await orchestrator.execute(
                message="",  # Not needed for resumption
                chat_history=[],
                checkpoint_id=request.checkpoint_id,
                user_input=request.user_input,
            )

            # Store resumed conversation messages
            # NOTE: conversation_id MUST exist from the initial pause
            if conversation_id in conversations:
                # Add user's year input as a message
                user_input_str = str(request.user_input.get("year", ""))
                conversations[conversation_id].append(
                    ChatMessage(role="user", content=user_input_str)
                )
                conversations[conversation_id].append(
                    ChatMessage(role="assistant", content=result["message"])
                )
            else:
                # Conversation not found - checkpoint might be from different session
                logger.warning(f"Conversation {conversation_id} not found for checkpoint resume")
                # Create new conversation with context from result
                conversations[conversation_id] = [
                    ChatMessage(role="assistant", content=result["message"])
                ]

            # Build response
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
                    "agents_used": result.get("metadata", {}).get(
                        "agents_used",
                        ["Query Verification", "Data Coordinator", "Data Extraction", "Analytics"],
                    ),
                    "resumed_from_checkpoint": True,
                    "original_query": result.get("metadata", {}).get("original_query"),
                    "user_provided": str(request.user_input.get("year", "")),
                },
            )

        # Normal flow - existing message handling
        if not request.message:
            raise HTTPException(
                status_code=400, detail="Message is required when not resuming from checkpoint"
            )

        logger.info(f"Message: {request.message[:100]}...")

        # AUTOMATIC CHECKPOINT DETECTION
        # Check if there's a pending checkpoint for this conversation
        from app.services.checkpoint_service import get_checkpoint_service

        checkpoint_service = get_checkpoint_service()
        pending_checkpoint = checkpoint_service.find_checkpoint_by_conversation(conversation_id)

        logger.info(f"Checking for pending checkpoint for conversation: {conversation_id}")
        logger.info(f"Found checkpoint: {pending_checkpoint.id if pending_checkpoint else 'None'}")
        if pending_checkpoint:
            logger.info(f"Checkpoint waiting for: {pending_checkpoint.waiting_for}")
            # Check if message looks like it's answering the checkpoint question
            waiting_for = pending_checkpoint.waiting_for

            if waiting_for.get("type") == "year_specification":
                # Message might be a year - try to parse it
                import re

                year_patterns = [
                    r"\b(19\d{2}|20[0-4]\d)\b",
                    r"\b(19\d{2}|20[0-4]\d)\s*(?:to|-|through|until)\s*(19\d{2}|20[0-4]\d)\b",
                ]

                has_year = False
                for pattern in year_patterns:
                    if re.search(pattern, request.message):
                        has_year = True
                        break

                # Also check if message is ONLY a year (no other meaningful content)
                # This catches cases like "2020" or "2020-2022"
                message_words = request.message.strip().split()
                is_year_only = len(message_words) <= 3 and has_year

                if is_year_only or has_year:
                    # Automatically resume from checkpoint
                    logger.info(
                        f"🔄 Auto-detected year input for checkpoint {pending_checkpoint.id}"
                    )
                    logger.info(f"Original query: {pending_checkpoint.original_query}")

                    orchestrator = get_orchestrator()
                    result = await orchestrator.execute(
                        message="",
                        chat_history=[],
                        checkpoint_id=pending_checkpoint.id,
                        user_input={"year": request.message},
                        conversation_id=conversation_id,
                    )

                    # Store conversation
                    if conversation_id in conversations:
                        conversations[conversation_id].append(
                            ChatMessage(role="user", content=request.message)
                        )
                        conversations[conversation_id].append(
                            ChatMessage(role="assistant", content=result["message"])
                        )

                    # Build response
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
                            "resumed_from_checkpoint": True,
                            "auto_detected": True,
                            "original_query": result.get("metadata", {}).get("original_query"),
                            "user_provided": request.message,
                        },
                    )

        # Get chat history
        chat_history = []
        if conversation_id in conversations:
            chat_history = [
                {"role": msg.role, "content": msg.content} for msg in conversations[conversation_id]
            ]
            logger.info(f"Loaded {len(chat_history)} previous messages")

        # Initialize agents through orchestrator with default config
        logger.info("Initializing orchestrator and agents...")
        orchestrator = get_orchestrator()

        # Execute multi-agent workflow
        logger.info("Starting multi-agent workflow execution...")
        result = await orchestrator.execute(
            message=request.message,
            chat_history=chat_history,
            conversation_id=conversation_id,
        )
        logger.info(
            f"Workflow completed. Agents used: {result.get('metadata', {}).get('agents_used', [])}"
        )

        # Check if workflow paused for input
        if result.get("metadata", {}).get("status") == "paused":
            # Store conversation up to this point
            if conversation_id not in conversations:
                conversations[conversation_id] = []

            conversations[conversation_id].append(ChatMessage(role="user", content=request.message))
            conversations[conversation_id].append(
                ChatMessage(role="assistant", content=result["message"])
            )

            return ChatResponse(
                message=result["message"],
                conversation_id=conversation_id,
                visualization=None,
                sources=[],
                metadata={
                    "status": "paused",
                    "checkpoint_id": result["metadata"]["checkpoint_id"],
                    "waiting_for": result["metadata"]["waiting_for"],
                    "agents_used": ["Query Verification"],
                },
            )

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
                },
            )

        if result.get("error") and not result.get("message"):
            raise HTTPException(status_code=500, detail=result["error"])

        # Store conversation
        if conversation_id not in conversations:
            conversations[conversation_id] = []

        conversations[conversation_id].append(ChatMessage(role="user", content=request.message))
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
async def list_agents() -> dict:
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
async def get_conversation_history(conversation_id: str) -> dict:
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
async def clear_conversation(conversation_id: str) -> dict:
    """Clear conversation history."""
    if conversation_id in conversations:
        del conversations[conversation_id]

    return {"status": "cleared", "conversation_id": conversation_id}


@router.get("/conversations")
async def list_conversations() -> dict:
    """List all conversation IDs."""
    return {
        "conversations": list(conversations.keys()),
        "count": len(conversations),
    }

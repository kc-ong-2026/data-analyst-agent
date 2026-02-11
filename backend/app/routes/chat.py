"""Chat API routes with simple context append for natural conversation."""

import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

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


@router.post("/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream chat response using Server-Sent Events (SSE) for real-time feedback."""
    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())
        logger.info(f"[SSE] Starting stream for conversation: {conversation_id}")

        if not request.message:
            raise HTTPException(status_code=400, detail="Message is required")

        async def event_generator() -> AsyncGenerator[str, None]:
            """Generate Server-Sent Events."""
            try:
                # Immediate acknowledgment
                yield f"data: {json.dumps({'type': 'start', 'conversation_id': conversation_id})}\n\n"
                yield f"data: {json.dumps({'type': 'agent', 'agent': 'verification', 'status': 'running', 'message': 'Validating query...'})}\n\n"
                await asyncio.sleep(0.05)

                # CONTEXT APPEND: Check if this is a clarification response
                combined_message = request.message

                if conversation_id in conversations and len(conversations[conversation_id]) >= 2:
                    # Get last assistant message and ALL previous user messages
                    last_assistant_msg = None
                    all_user_messages = []

                    for i in range(len(conversations[conversation_id]) - 1, -1, -1):
                        msg = conversations[conversation_id][i]
                        if msg.role == "assistant" and last_assistant_msg is None:
                            last_assistant_msg = msg.content.lower()
                        elif msg.role == "user":
                            all_user_messages.insert(0, msg.content)  # Insert at start to maintain order

                    # Check if assistant was asking for clarification
                    if last_assistant_msg and all_user_messages:
                        clarification_keywords = [
                            "which year", "specify year", "year range", "what year",
                            "available data", "please specify", "interested in", "provide",
                            "dimension", "age group", "sex/gender", "industry", "qualification"
                        ]
                        is_asking_for_clarification = any(kw in last_assistant_msg for kw in clarification_keywords)

                        if is_asking_for_clarification:
                            # Combine ALL previous user messages with current one for full context!
                            combined_message = " ".join(all_user_messages + [request.message])
                            logger.info(f"🔗 [SSE] Context Append Detected!")
                            logger.info(f"   Previous messages: {all_user_messages}")
                            logger.info(f"   Current response: '{request.message}'")
                            logger.info(f"   Combined query: '{combined_message}'")

                # Execute workflow with combined message
                orchestrator = get_orchestrator()
                result = await orchestrator.execute(message=combined_message, chat_history=[])

                # Check validation
                validation = result.get("query_validation", {})
                if validation and not validation.get("valid", True):
                    # Validation failed - stream error
                    error_msg = validation.get("reason", "Query validation failed")
                    yield f"data: {json.dumps({'type': 'agent', 'agent': 'verification', 'status': 'complete'})}\n\n"

                    for char in error_msg:
                        yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"
                        await asyncio.sleep(0.008)

                    full_message = error_msg
                else:
                    # Success - show progression
                    yield f"data: {json.dumps({'type': 'agent', 'agent': 'verification', 'status': 'complete'})}\n\n"
                    yield f"data: {json.dumps({'type': 'agent', 'agent': 'coordinator', 'status': 'running'})}\n\n"
                    await asyncio.sleep(0.05)
                    yield f"data: {json.dumps({'type': 'agent', 'agent': 'coordinator', 'status': 'complete'})}\n\n"
                    yield f"data: {json.dumps({'type': 'agent', 'agent': 'extraction', 'status': 'running'})}\n\n"
                    await asyncio.sleep(0.05)
                    yield f"data: {json.dumps({'type': 'agent', 'agent': 'extraction', 'status': 'complete'})}\n\n"
                    yield f"data: {json.dumps({'type': 'agent', 'agent': 'analytics', 'status': 'running'})}\n\n"
                    await asyncio.sleep(0.05)
                    yield f"data: {json.dumps({'type': 'agent', 'agent': 'analytics', 'status': 'complete'})}\n\n"

                    # Stream response
                    full_message = result.get("message", "Analysis complete")
                    for char in full_message:
                        yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"
                        await asyncio.sleep(0.008)

                    # Send visualization
                    if request.include_visualization and result.get("visualization"):
                        yield f"data: {json.dumps({'type': 'visualization', 'visualization': result['visualization']})}\n\n"

                # Store conversation
                if conversation_id not in conversations:
                    conversations[conversation_id] = []
                conversations[conversation_id].append(ChatMessage(role="user", content=request.message))
                conversations[conversation_id].append(ChatMessage(role="assistant", content=full_message))

                # Clear conversation after successful query (keep only for validation failures)
                if not (validation and not validation.get("valid", True)):
                    logger.info(f"🧹 [SSE] Clearing conversation context after successful query")
                    del conversations[conversation_id]

                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            except Exception as e:
                logger.error(f"[SSE] Stream error: {str(e)}", exc_info=True)

                # Sanitize error message for user
                user_error = "I encountered an issue while processing your request."
                if "invalid literal" in str(e).lower() or "cannot convert" in str(e).lower():
                    user_error = "I found some data quality issues in the dataset. The data contains invalid values that prevent analysis. Please try a different query or time period."
                elif "no data" in str(e).lower() or "empty" in str(e).lower():
                    user_error = "No data was found matching your query criteria. Please try different parameters or check if data exists for this period."
                elif "timeout" in str(e).lower():
                    user_error = "The query took too long to process. Please try a more specific query with fewer years or dimensions."
                else:
                    user_error = f"I encountered an error while processing your request. Please try rephrasing your query or contact support if the issue persists."

                yield f"data: {json.dumps({'type': 'error', 'error': user_error, 'technical_details': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )

    except Exception as e:
        logger.error(f"[SSE] Failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
            # Get last assistant message and ALL previous user messages
            last_assistant_msg = None
            all_user_messages = []

            for i in range(len(conversations[conversation_id]) - 1, -1, -1):
                msg = conversations[conversation_id][i]
                if msg.role == "assistant" and last_assistant_msg is None:
                    last_assistant_msg = msg.content.lower()
                elif msg.role == "user":
                    all_user_messages.insert(0, msg.content)  # Insert at start to maintain order

            # Check if assistant was asking for clarification
            if last_assistant_msg and all_user_messages:
                clarification_keywords = [
                    "which year", "specify year", "year range", "what year",
                    "available data", "please specify", "interested in", "provide",
                    "dimension", "age group", "sex/gender", "industry", "qualification"
                ]
                is_asking_for_clarification = any(kw in last_assistant_msg for kw in clarification_keywords)

                if is_asking_for_clarification:
                    # Combine ALL previous user messages with current one for full context!
                    combined_message = " ".join(all_user_messages + [request.message])
                    logger.info(f"🔗 Context Append Detected!")
                    logger.info(f"   Previous messages: {all_user_messages}")
                    logger.info(f"   Current response: '{request.message}'")
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

        # Check if verification failed (topic invalid)
        validation = result.get("query_validation", {})
        if validation and not validation.get("valid", True):
            # KEEP conversation history for context append on follow-up
            # Store the validation error so user can respond
            if conversation_id not in conversations:
                conversations[conversation_id] = []

            conversations[conversation_id].append(
                ChatMessage(role="user", content=request.message)
            )
            conversations[conversation_id].append(
                ChatMessage(role="assistant", content=validation.get("reason", "Query validation failed"))
            )
            logger.info(f"💬 Keeping conversation history for clarification follow-up")

            # Return validation error
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

        # Store conversation for successful query
        if conversation_id not in conversations:
            conversations[conversation_id] = []

        conversations[conversation_id].append(
            ChatMessage(role="user", content=request.message)  # Store original message
        )
        conversations[conversation_id].append(
            ChatMessage(role="assistant", content=result["message"])
        )

        # Clear conversation after successful query (so next query is independent)
        logger.info(f"🧹 Clearing conversation context after successful query")
        del conversations[conversation_id]

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

        # Sanitize error message for user
        user_message = "I encountered an issue while processing your request."
        if "invalid literal" in str(e).lower() or "cannot convert" in str(e).lower():
            user_message = "The dataset contains invalid values that prevent analysis. Please try a different query or time period."
        elif "no data" in str(e).lower() or "empty" in str(e).lower():
            user_message = "No data was found matching your query. Please try different parameters."
        elif "timeout" in str(e).lower():
            user_message = "The query took too long to process. Please try a more specific query."

        raise HTTPException(status_code=500, detail=user_message)


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

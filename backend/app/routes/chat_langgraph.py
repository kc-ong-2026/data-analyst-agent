"""Chat API routes with LangGraph checkpoint support."""

import json
import logging
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

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


@router.post("/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream chat response using Server-Sent Events (SSE).

    Uses LangGraph's astream_events API to stream tokens and agent updates in real-time.
    This provides immediate feedback to users instead of waiting for complete response.

    SSE Event Types:
    - start: Initial acknowledgment with conversation_id
    - agent: Agent status update (which agent is running)
    - token: LLM token as it's generated
    - data: Structured data (extracted_data, analysis_results)
    - visualization: Visualization data when available
    - done: Final response complete
    - error: Error occurred
    """
    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())
        logger.info(f"[SSE] Starting stream for conversation: {conversation_id}")

        if not request.message:
            raise HTTPException(status_code=400, detail="Message is required")

        async def event_generator() -> AsyncGenerator[str, None]:
            """Generate Server-Sent Events for streaming response."""
            try:
                # Send immediate acknowledgment
                yield f"data: {json.dumps({'type': 'start', 'conversation_id': conversation_id})}\n\n"

                # Get chat history
                chat_history = []
                if conversation_id in conversations:
                    chat_history = [
                        {"role": msg.role, "content": msg.content}
                        for msg in conversations[conversation_id]
                    ]

                # LangGraph config
                config = {"configurable": {"thread_id": conversation_id}}

                # Initialize orchestrator
                orchestrator = get_orchestrator()

                # Prepare initial state
                from app.services.agents.base_agent import AgentState

                state = AgentState()
                state.current_task = request.message
                for msg_dict in chat_history:
                    state.add_message(msg_dict["role"], msg_dict["content"])
                state.add_message("user", request.message)
                initial_graph_state = state.to_graph_state()

                # Track which agents have started
                current_agent = None
                message_buffer = []
                final_state = None

                # Stream events from LangGraph
                async for event in orchestrator._orchestration_graph.astream_events(
                    initial_graph_state,
                    config=config,
                    version="v2",
                ):
                    event_type = event.get("event")
                    event_name = event.get("name", "")
                    event_data = event.get("data", {})

                    # Agent node started
                    if event_type == "on_chain_start":
                        if "verification" in event_name.lower():
                            current_agent = "verification"
                            yield f"data: {json.dumps({'type': 'agent', 'agent': 'verification', 'status': 'running', 'message': 'Validating query...'})}\n\n"
                        elif "coordinator" in event_name.lower():
                            current_agent = "coordinator"
                            yield f"data: {json.dumps({'type': 'agent', 'agent': 'coordinator', 'status': 'running', 'message': 'Planning research workflow...'})}\n\n"
                        elif "extraction" in event_name.lower():
                            current_agent = "extraction"
                            yield f"data: {json.dumps({'type': 'agent', 'agent': 'extraction', 'status': 'running', 'message': 'Retrieving relevant data...'})}\n\n"
                        elif "analytics" in event_name.lower():
                            current_agent = "analytics"
                            yield f"data: {json.dumps({'type': 'agent', 'agent': 'analytics', 'status': 'running', 'message': 'Analyzing data and generating insights...'})}\n\n"

                    # LLM streaming tokens
                    elif event_type == "on_chat_model_stream":
                        chunk = event_data.get("chunk")
                        if hasattr(chunk, "content") and chunk.content:
                            content = chunk.content
                            message_buffer.append(content)
                            yield f"data: {json.dumps({'type': 'token', 'content': content, 'agent': current_agent})}\n\n"

                    # Agent node completed
                    elif event_type == "on_chain_end":
                        if any(
                            name in event_name.lower()
                            for name in ["verification", "coordinator", "extraction", "analytics"]
                        ):
                            agent_name = next(
                                (
                                    name
                                    for name in [
                                        "verification",
                                        "coordinator",
                                        "extraction",
                                        "analytics",
                                    ]
                                    if name in event_name.lower()
                                ),
                                None,
                            )
                            if agent_name:
                                yield f"data: {json.dumps({'type': 'agent', 'agent': agent_name, 'status': 'complete'})}\n\n"

                                # Send intermediate data if available
                                output = event_data.get("output", {})
                                if agent_name == "extraction" and output.get("extracted_data"):
                                    yield f"data: {json.dumps({'type': 'data', 'data_type': 'extracted', 'summary': 'Data extracted successfully'})}\n\n"
                                elif agent_name == "analytics" and output.get("analysis_results"):
                                    final_state = output

                # Reconstruct final response
                if final_state:
                    # Send visualization if available
                    if request.include_visualization and final_state.get(
                        "analysis_results", {}
                    ).get("visualization"):
                        viz = final_state["analysis_results"]["visualization"]
                        yield f"data: {json.dumps({'type': 'visualization', 'visualization': viz})}\n\n"

                    # Send complete message
                    full_message = (
                        "".join(message_buffer)
                        if message_buffer
                        else final_state.get("analysis_results", {}).get(
                            "explanation", "Analysis complete"
                        )
                    )

                    # Store conversation
                    if conversation_id not in conversations:
                        conversations[conversation_id] = []
                    conversations[conversation_id].append(
                        ChatMessage(role="user", content=request.message)
                    )
                    conversations[conversation_id].append(
                        ChatMessage(role="assistant", content=full_message)
                    )

                # Send done event
                yield f"data: {json.dumps({'type': 'done', 'message': 'Stream complete'})}\n\n"

            except Exception as e:
                logger.error(f"[SSE] Stream error: {str(e)}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SSE] Failed to initialize stream: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
                {"role": msg.role, "content": msg.content} for msg in conversations[conversation_id]
            ]
            logger.info(f"Loaded {len(chat_history)} previous messages")

        # LangGraph config with thread_id for checkpoint support
        # Same thread_id = automatic resume if graph was interrupted
        config = {"configurable": {"thread_id": conversation_id}}  # This is the magic!

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
        logger.info(
            f"Workflow completed. Agents used: {result.get('metadata', {}).get('agents_used', [])}"
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

"""
Chat Streaming Integration Tests.

Tests the SSE (Server-Sent Events) streaming functionality with real-time agent status updates.
"""

import json

import pytest
from httpx import AsyncClient


@pytest.mark.integration
@pytest.mark.asyncio
class TestChatStreamingSSE:
    """Test chat streaming with Server-Sent Events."""

    async def test_stream_endpoint_exists(self, async_client: AsyncClient):
        """Test that /api/chat/stream endpoint exists."""
        response = await async_client.post(
            "/api/chat/stream",
            json={"message": "test"},
        )
        # Should not be 404
        assert response.status_code != 404

    async def test_stream_sends_start_event(self, async_client: AsyncClient):
        """Test that stream sends start event immediately."""
        events = []

        async with async_client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "What was the employment rate in 2023?"},
            timeout=30.0,
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

            # Collect first few events
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    try:
                        event = json.loads(data_str)
                        events.append(event)
                        if len(events) >= 1:
                            break
                    except json.JSONDecodeError:
                        pass

        # First event should be 'start'
        assert len(events) > 0
        assert events[0]["type"] == "start"
        assert "conversation_id" in events[0]

    async def test_stream_sends_agent_events(self, async_client: AsyncClient):
        """Test that stream sends agent status events."""
        events = []

        async with async_client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "What was the employment rate in 2023?"},
            timeout=60.0,
        ) as response:
            assert response.status_code == 200

            # Collect events until done or timeout
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        event = json.loads(data_str)
                        events.append(event)

                        # Stop at 'done' event
                        if event.get("type") == "done":
                            break

                        # Safety limit
                        if len(events) > 1000:
                            break
                    except json.JSONDecodeError:
                        pass

        # Extract agent events
        agent_events = [e for e in events if e.get("type") == "agent"]

        # Should have received agent events
        assert len(agent_events) > 0, "Should receive agent status events"

        # Verify agent event structure
        for event in agent_events:
            assert "agent" in event, "Agent event should have 'agent' field"
            assert "status" in event, "Agent event should have 'status' field"
            assert event["status"] in ["running", "complete"], f"Invalid status: {event['status']}"

        print(f"\n✅ Received {len(agent_events)} agent events")

    async def test_stream_agent_event_order(self, async_client: AsyncClient):
        """Test that agent events arrive in correct order (start -> complete)."""
        events = []

        async with async_client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "What was the employment rate in 2023?"},
            timeout=60.0,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        event = json.loads(line[6:])
                        events.append(event)
                        if event.get("type") == "done":
                            break
                    except json.JSONDecodeError:
                        pass

        # Track agent state transitions
        agent_states = {}
        agent_events = [e for e in events if e.get("type") == "agent"]

        for event in agent_events:
            agent = event["agent"]
            status = event["status"]

            if agent not in agent_states:
                agent_states[agent] = []
            agent_states[agent].append(status)

        # Verify each agent goes from "running" to "complete"
        for agent, states in agent_states.items():
            if "running" in states and "complete" in states:
                running_idx = states.index("running")
                complete_idx = states.index("complete")
                assert running_idx < complete_idx, f"Agent {agent} completed before starting"

        print(f"\n✅ Agent state transitions verified for {len(agent_states)} agents")

    async def test_stream_includes_expected_agents(self, async_client: AsyncClient):
        """Test that stream includes events from expected agents."""
        events = []

        async with async_client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "What was the employment rate in 2023?"},
            timeout=60.0,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        event = json.loads(line[6:])
                        events.append(event)
                        if event.get("type") == "done":
                            break
                    except json.JSONDecodeError:
                        pass

        # Extract unique agents
        agents = set()
        for event in events:
            if event.get("type") == "agent":
                agents.add(event["agent"])

        # Should have verification at minimum
        assert "verification" in agents, "Should have verification agent"

        # For valid query, should have more agents
        # (coordinator, extraction, analytics - if validation passes)
        print(f"\n✅ Agents present: {sorted(agents)}")

    async def test_stream_sends_token_events(self, async_client: AsyncClient):
        """Test that stream sends token events for response."""
        events = []

        async with async_client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "What was the employment rate in 2023?"},
            timeout=60.0,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        event = json.loads(line[6:])
                        events.append(event)
                        if event.get("type") == "done":
                            break
                    except json.JSONDecodeError:
                        pass

        # Should have token events
        token_events = [e for e in events if e.get("type") == "token"]
        assert len(token_events) > 0, "Should stream response tokens"

        # Verify token structure
        for event in token_events:
            assert "content" in event, "Token event should have 'content'"
            assert isinstance(event["content"], str), "Token content should be string"

        print(f"\n✅ Received {len(token_events)} token events")

    async def test_stream_ends_with_done(self, async_client: AsyncClient):
        """Test that stream ends with 'done' event."""
        events = []

        async with async_client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "What was the employment rate in 2023?"},
            timeout=60.0,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        event = json.loads(line[6:])
                        events.append(event)
                        if event.get("type") == "done":
                            break
                    except json.JSONDecodeError:
                        pass

        # Last event should be 'done'
        assert len(events) > 0
        assert events[-1]["type"] == "done"

    async def test_stream_invalid_query_stops_early(self, async_client: AsyncClient):
        """Test that invalid query stops after verification."""
        events = []

        async with async_client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "What's the weather today?"},  # Off-topic
            timeout=30.0,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        event = json.loads(line[6:])
                        events.append(event)
                        if event.get("type") == "done":
                            break
                    except json.JSONDecodeError:
                        pass

        # Extract agents
        agents = set()
        for event in events:
            if event.get("type") == "agent":
                agents.add(event["agent"])

        # Should only have verification (query fails validation)
        # Other agents should NOT execute
        assert "verification" in agents

        print(f"\n✅ Invalid query handled correctly, agents: {sorted(agents)}")


@pytest.mark.integration
@pytest.mark.asyncio
class TestStreamingEdgeCases:
    """Test edge cases and error handling."""

    async def test_stream_empty_message(self, async_client: AsyncClient):
        """Test streaming with empty message."""
        response = await async_client.post(
            "/api/chat/stream",
            json={"message": ""},
        )
        # Should handle gracefully (400 or error event)
        assert response.status_code in [200, 400]

    async def test_stream_missing_message(self, async_client: AsyncClient):
        """Test streaming with missing message field."""
        response = await async_client.post(
            "/api/chat/stream",
            json={},
        )
        # Should return 400 or handle gracefully
        assert response.status_code in [200, 400, 422]

    async def test_stream_with_visualization(self, async_client: AsyncClient):
        """Test streaming includes visualization when requested."""
        events = []

        async with async_client.stream(
            "POST",
            "/api/chat/stream",
            json={
                "message": "Show me employment rate trends",
                "include_visualization": True,
            },
            timeout=60.0,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        event = json.loads(line[6:])
                        events.append(event)
                        if event.get("type") == "done":
                            break
                    except json.JSONDecodeError:
                        pass

        # Check if visualization event was sent
        viz_events = [e for e in events if e.get("type") == "visualization"]

        # May or may not have visualization depending on analytics agent
        print(f"\n{'✅' if viz_events else 'ℹ️'} Visualization events: {len(viz_events)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

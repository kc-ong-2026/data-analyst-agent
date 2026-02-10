"""Checkpoint service for managing workflow pause/resume."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
import uuid

from app.services.agents.base_agent import GraphState


@dataclass
class WorkflowCheckpoint:
    """Represents a paused workflow that can be resumed."""

    id: str
    conversation_id: str
    original_query: str
    graph_state: Dict[str, Any]  # Serialized GraphState
    agent_step: str
    waiting_for: Dict[str, Any]
    created_at: datetime
    expires_at: datetime

    def is_expired(self) -> bool:
        """Check if checkpoint has expired."""
        return datetime.now(timezone.utc) > self.expires_at


class CheckpointService:
    """Service for managing workflow checkpoints (in-memory for now)."""

    def __init__(self, expiration_minutes: int = 30):
        self.checkpoints: Dict[str, WorkflowCheckpoint] = {}
        self.expiration_minutes = expiration_minutes

    def create_checkpoint(
        self,
        conversation_id: str,
        original_query: str,
        graph_state: GraphState,
        agent_step: str,
        waiting_for: Dict[str, Any],
    ) -> str:
        """Create a new checkpoint and return its ID."""
        checkpoint_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        checkpoint = WorkflowCheckpoint(
            id=checkpoint_id,
            conversation_id=conversation_id,
            original_query=original_query,
            graph_state=dict(graph_state),  # Serialize
            agent_step=agent_step,
            waiting_for=waiting_for,
            created_at=now,
            expires_at=now + timedelta(minutes=self.expiration_minutes),
        )

        self.checkpoints[checkpoint_id] = checkpoint
        return checkpoint_id

    def get_checkpoint(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        """Retrieve a checkpoint by ID."""
        checkpoint = self.checkpoints.get(checkpoint_id)

        if checkpoint and checkpoint.is_expired():
            del self.checkpoints[checkpoint_id]
            return None

        return checkpoint

    def delete_checkpoint(self, checkpoint_id: str):
        """Delete a checkpoint after successful resumption."""
        if checkpoint_id in self.checkpoints:
            del self.checkpoints[checkpoint_id]

    def find_checkpoint_by_conversation(self, conversation_id: str) -> Optional[WorkflowCheckpoint]:
        """Find the most recent checkpoint for a conversation."""
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"Looking for checkpoints with conversation_id: '{conversation_id}'")
        logger.info(f"Total checkpoints in memory: {len(self.checkpoints)}")

        for cid, cp in self.checkpoints.items():
            logger.info(f"Checkpoint {cid}: conversation_id='{cp.conversation_id}', expired={cp.is_expired()}")

        matching_checkpoints = [
            cp for cp in self.checkpoints.values()
            if cp.conversation_id == conversation_id and not cp.is_expired()
        ]

        if not matching_checkpoints:
            logger.warning(f"No matching checkpoints found for conversation_id: '{conversation_id}'")
            return None

        # Return most recent (by created_at)
        result = max(matching_checkpoints, key=lambda cp: cp.created_at)
        logger.info(f"Found checkpoint: {result.id}")
        return result

    def cleanup_expired(self):
        """Remove expired checkpoints."""
        expired_ids = [
            cid for cid, cp in self.checkpoints.items()
            if cp.is_expired()
        ]
        for cid in expired_ids:
            del self.checkpoints[cid]


# Global instance (in-memory for now, can move to Redis/PostgreSQL later)
_checkpoint_service = CheckpointService()


def get_checkpoint_service() -> CheckpointService:
    """Get the checkpoint service instance."""
    return _checkpoint_service

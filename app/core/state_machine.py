"""
Execution State Machine for Vector Desktop AI Assistant.
Tracks and enforces the 7-stage execution lifecycle of tool requests.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from app.config.constants import ExecutionState


class ExecutionStateMachine:
    """
    Manages state transitions for a tool execution request:
    REQUESTED -> PARSED -> VALIDATED -> PERMISSION_CHECK -> CONFIRMATION -> EXECUTING -> SUCCESS / FAILED
    """

    # Allowed forward transitions
    VALID_TRANSITIONS: Dict[ExecutionState, List[ExecutionState]] = {
        ExecutionState.REQUESTED: [ExecutionState.PARSED, ExecutionState.FAILED],
        ExecutionState.PARSED: [ExecutionState.VALIDATED, ExecutionState.FAILED],
        ExecutionState.VALIDATED: [ExecutionState.PERMISSION_CHECK, ExecutionState.FAILED],
        ExecutionState.PERMISSION_CHECK: [ExecutionState.CONFIRMATION, ExecutionState.EXECUTING, ExecutionState.FAILED],
        ExecutionState.CONFIRMATION: [ExecutionState.EXECUTING, ExecutionState.FAILED],
        ExecutionState.EXECUTING: [ExecutionState.SUCCESS, ExecutionState.FAILED],
        ExecutionState.SUCCESS: [],
        ExecutionState.FAILED: []
    }

    def __init__(self, request_id: str, tool_name: str = ""):
        self.request_id: str = request_id
        self.tool_name: str = tool_name
        self.current_state: ExecutionState = ExecutionState.REQUESTED
        self.history: List[Dict[str, Any]] = [
            {
                "state": ExecutionState.REQUESTED.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": "Request initialized"
            }
        ]

    def transition_to(self, target_state: ExecutionState, details: str = "") -> bool:
        """
        Transitions the state machine to target_state if valid.
        Returns True if transition succeeded, False if invalid.
        """
        allowed_targets = self.VALID_TRANSITIONS.get(self.current_state, [])
        if target_state not in allowed_targets:
            # Illegal transition attempt
            self.history.append({
                "state": self.current_state.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": f"Illegal transition attempt from {self.current_state.value} to {target_state.value}"
            })
            return False

        self.current_state = target_state
        self.history.append({
            "state": target_state.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or f"State changed to {target_state.value}"
        })
        return True

    @property
    def is_terminal(self) -> bool:
        """Returns True if state machine is in terminal SUCCESS or FAILED state."""
        return self.current_state in (ExecutionState.SUCCESS, ExecutionState.FAILED)

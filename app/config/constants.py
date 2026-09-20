"""
Core constants, enums, and guardrails for Vector Desktop AI Assistant.
"""

from enum import Enum


class PermissionLevel(str, Enum):
    """
    Security categories assigned to tools.
    - SAFE: Automatic execution without user prompt.
    - CONFIRM: Prompts user with standard confirmation modal before execution.
    - DANGEROUS: Requires explicit high-friction confirmation modal.
    - BLOCKED: Strictly prohibited from execution under any context.
    """
    SAFE = "SAFE"
    CONFIRM = "CONFIRM"
    DANGEROUS = "DANGEROUS"
    BLOCKED = "BLOCKED"


class ExecutionState(str, Enum):
    """
    7-stage lifecycle state machine tracking tool call execution.
    """
    REQUESTED = "REQUESTED"
    PARSED = "PARSED"
    VALIDATED = "VALIDATED"
    PERMISSION_CHECK = "PERMISSION_CHECK"
    CONFIRMATION = "CONFIRMATION"
    EXECUTING = "EXECUTING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class IntentTier(str, Enum):
    """
    Multi-tiered routing levels.
    - TIER_0_DIRECT: Sub-10ms direct pattern matcher (reflexes).
    - TIER_1_NEEDLE: Fast local intent recognition model.
    - TIER_2_GEMINI: Cloud multi-step reasoning & planning fallback.
    """
    TIER_0_DIRECT = "TIER_0_DIRECT"
    TIER_1_NEEDLE = "TIER_1_NEEDLE"
    TIER_2_GEMINI = "TIER_2_GEMINI"


# Safety & System Guardrails
MAX_TOOL_CALLS: int = 10
DEFAULT_NEEDLE_THRESHOLD: float = 0.85
DEFAULT_LOG_LEVEL: str = "INFO"
DEFAULT_DATABASE_PATH: str = "data/vector.db"

# Terminal Whitelist (Permitted CLI tools)
ALLOWED_TERMINAL_COMMANDS: set[str] = {
    "git",
    "python",
    "pip",
    "npm",
    "node",
    "docker",
    "systemctl"
}

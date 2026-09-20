"""
Permission policy engine for Vector Desktop AI Assistant.
Enforces tool access control outside the LLM layer.
"""

from dataclasses import dataclass
from app.config.constants import PermissionLevel
from app.config.settings import Settings, get_settings


@dataclass
class PermissionDecision:
    """
    Data class representing the security decision for a tool execution request.
    """
    allowed: bool
    requires_confirmation: bool
    permission_level: PermissionLevel
    reason: str


class PermissionEngine:
    """
    Security engine that checks tool permissions against configuration policies.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def evaluate(self, permission_level: PermissionLevel) -> PermissionDecision:
        """
        Evaluates whether a tool with the given permission_level can execute directly,
        requires user confirmation, or is strictly blocked.
        """
        if permission_level == PermissionLevel.BLOCKED:
            return PermissionDecision(
                allowed=False,
                requires_confirmation=False,
                permission_level=permission_level,
                reason="Tool execution is strictly blocked by security policy."
            )

        if permission_level == PermissionLevel.DANGEROUS:
            requires_confirm = self.settings.require_confirmation_for_dangerous
            return PermissionDecision(
                allowed=True,
                requires_confirmation=requires_confirm,
                permission_level=permission_level,
                reason="Dangerous operation requires explicit high-friction confirmation." if requires_confirm else "Dangerous tool execution approved."
            )

        if permission_level == PermissionLevel.CONFIRM:
            return PermissionDecision(
                allowed=True,
                requires_confirmation=True,
                permission_level=permission_level,
                reason="Tool operation requires user confirmation modal."
            )

        # PermissionLevel.SAFE
        return PermissionDecision(
            allowed=True,
            requires_confirmation=False,
            permission_level=permission_level,
            reason="Safe tool approved for direct automatic execution."
        )

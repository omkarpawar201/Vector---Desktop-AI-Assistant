"""
Safe Tool Executor for Vector Desktop AI Assistant.
Orchestrates the 7-stage state machine, argument validation, and permission checks
before executing any local desktop tool.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from app.config.constants import ExecutionState
from app.core.state_machine import ExecutionStateMachine
from app.security.permissions import PermissionEngine
from app.security.validator import ArgumentValidator
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry, get_tool_registry


@dataclass
class PendingApproval:
    """
    Single-use token binding a user's confirmation to one exact action.

    A pending approval records the tool and the validated arguments that were
    shown to the user.  It authorizes nothing else: the executor only accepts it
    when it names the same tool with the same validated arguments, and it is
    consumed on first use, so it can never be replayed for a second execution.
    """

    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    consumed: bool = False

    def matches(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """True when this approval authorizes exactly (tool_name, arguments)."""
        return self.tool_name == tool_name and self.arguments == arguments

    def consume(self) -> None:
        """Mark the approval used; a consumed approval authorizes nothing."""
        self.consumed = True


class SafeToolExecutor:
    """
    Executes tools safely by evaluating security permissions, validating parameters,
    and tracking execution state lifecycle.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        permission_engine: PermissionEngine | None = None
    ):
        self.registry = registry or get_tool_registry()
        self.permission_engine = permission_engine or PermissionEngine()

    def execute_tool(
        self,
        tool_name: str,
        raw_args: Dict[str, Any],
        approval: Optional[PendingApproval] = None
    ) -> Tuple[ToolResult, ExecutionStateMachine]:
        """
        Executes a tool by name with safety checks.
        Returns a tuple of (ToolResult, ExecutionStateMachine).

        A tool whose permission level requires confirmation only runs when a
        matching, unconsumed ``approval`` is supplied; otherwise the call stops at
        the CONFIRMATION state with an error carrying the validated arguments so
        the caller can bind a confirmation to this exact action.
        """
        request_id = str(uuid4())
        sm = ExecutionStateMachine(request_id=request_id, tool_name=tool_name)

        # Stage 1: PARSED
        sm.transition_to(ExecutionState.PARSED, f"Parsed tool call request: {tool_name}")
        tool = self.registry.get(tool_name)
        if not tool:
            sm.transition_to(ExecutionState.FAILED, f"Tool '{tool_name}' is not registered.")
            return (
                ToolResult.fail(tool=tool_name, error="TOOL_NOT_FOUND", message=f"Tool '{tool_name}' not found."),
                sm
            )

        # Stage 2: VALIDATED
        sm.transition_to(ExecutionState.VALIDATED, "Validating parameters against JSON schema.")
        val_res = ArgumentValidator.validate(tool.parameters, raw_args)
        if not val_res.valid:
            err_msg = "; ".join(val_res.errors)
            sm.transition_to(ExecutionState.FAILED, f"Parameter validation failed: {err_msg}")
            return (
                ToolResult.fail(tool=tool_name, error="INVALID_ARGUMENTS", message=err_msg),
                sm
            )

        # Stage 3: PERMISSION_CHECK
        sm.transition_to(ExecutionState.PERMISSION_CHECK, f"Evaluating permission level '{tool.permission_level.value}'.")
        perm_res = self.permission_engine.evaluate(tool.permission_level)

        if not perm_res.allowed:
            sm.transition_to(ExecutionState.FAILED, perm_res.reason)
            return (
                ToolResult.fail(tool=tool_name, error="PERMISSION_BLOCKED", message=perm_res.reason),
                sm
            )

        # Stage 4: CONFIRMATION check if required
        if perm_res.requires_confirmation:
            if approval is None:
                sm.transition_to(ExecutionState.CONFIRMATION, "Awaiting explicit user confirmation.")
                return (
                    ToolResult(
                        success=False,
                        tool=tool_name,
                        data={"validated_arguments": val_res.cleaned_args},
                        error="CONFIRMATION_REQUIRED",
                        message=f"Action '{tool_name}' requires confirmation: {perm_res.reason}"
                    ),
                    sm
                )
            if approval.consumed:
                sm.transition_to(ExecutionState.FAILED, "Approval has already been consumed.")
                return (
                    ToolResult.fail(
                        tool=tool_name,
                        error="APPROVAL_CONSUMED",
                        message="This confirmation was already used. Please confirm the action again."
                    ),
                    sm
                )
            if not approval.matches(tool_name, val_res.cleaned_args):
                sm.transition_to(ExecutionState.FAILED, "Approval does not authorize this action.")
                return (
                    ToolResult.fail(
                        tool=tool_name,
                        error="APPROVAL_MISMATCH",
                        message="The approved action does not match this request. Please confirm the action again."
                    ),
                    sm
                )
            # The approval authorizes this exact action; spend it before executing.
            approval.consume()

        # Stage 5: EXECUTING
        sm.transition_to(ExecutionState.EXECUTING, "Executing tool code...")
        try:
            result = tool.execute(**val_res.cleaned_args)
            if result.success:
                sm.transition_to(ExecutionState.SUCCESS, "Execution completed successfully.")
            else:
                sm.transition_to(ExecutionState.FAILED, f"Tool reported failure: {result.error}")
            return (result, sm)
        except Exception as e:
            err_str = str(e)
            sm.transition_to(ExecutionState.FAILED, f"Unhandled exception during tool execution: {err_str}")
            return (
                ToolResult.fail(tool=tool_name, error="EXECUTION_EXCEPTION", message=err_str),
                sm
            )

    def execute_approved(
        self,
        approval: PendingApproval
    ) -> Tuple[ToolResult, ExecutionStateMachine]:
        """
        Executes the exact action a user approved.

        This re-enters the full pipeline with the approval token: the tool must
        still be registered, the arguments are validated again, permissions are
        re-evaluated, and the approval must name precisely this action and be
        unused.  Nothing about the original request text is re-interpreted.
        """
        return self.execute_tool(
            tool_name=approval.tool_name,
            raw_args=dict(approval.arguments),
            approval=approval
        )

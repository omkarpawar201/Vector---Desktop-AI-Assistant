"""
Safe Tool Executor for Vector Desktop AI Assistant.
Orchestrates the 7-stage state machine, argument validation, and permission checks
before executing any local desktop tool.
"""

import copy
import threading
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from app.config.constants import ExecutionState
from app.core.state_machine import ExecutionStateMachine
from app.security.permissions import PermissionEngine
from app.security.validator import ArgumentValidator
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry, get_tool_registry


class PendingApproval:
    """
    Single-use token binding a user's confirmation to one exact action.

    A pending approval keeps a private, deep-copied snapshot of the tool name and
    the validated arguments that were shown to the user.  The snapshot is frozen
    at construction and only ever copied out, so the action that executes is
    always the action that was displayed and cannot be changed after the fact.
    Spending the token is atomic, so even simultaneous callers can only ever
    execute the approved action once.
    """

    __slots__ = ("_tool_name", "_arguments", "_lock", "_consumed")

    def __init__(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None):
        object.__setattr__(self, "_tool_name", str(tool_name))
        object.__setattr__(
            self,
            "_arguments",
            copy.deepcopy(dict(arguments)) if arguments else {},
        )
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(self, "_consumed", False)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("PendingApproval is immutable once created.")

    @property
    def tool_name(self) -> str:
        """The approved tool's name."""
        return self._tool_name

    @property
    def arguments(self) -> Dict[str, Any]:
        """A defensive copy of the approved, validated arguments."""
        return copy.deepcopy(self._arguments)

    @property
    def consumed(self) -> bool:
        """Whether this approval has already been spent."""
        with self._lock:
            return self._consumed

    def bound_action(self) -> Tuple[str, Dict[str, Any]]:
        """
        The immutable (tool_name, arguments) this approval authorizes.

        Returns copies, so no caller can alter the approved action through the
        values it hands back to the executor.
        """
        return self._tool_name, copy.deepcopy(self._arguments)

    def matches(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """True when this approval authorizes exactly (tool_name, arguments)."""
        return self._tool_name == tool_name and self._arguments == arguments

    def try_consume(self) -> bool:
        """
        Atomically spend this approval.

        Returns True only for the first caller; every later caller gets False, so
        a single approval can never authorize two executions.  The lock is held
        only for the flag flip, never while a tool runs.
        """
        with self._lock:
            if self._consumed:
                return False
            object.__setattr__(self, "_consumed", True)
            return True


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
            # Bind to the exact approved action before spending the token, so a
            # mismatched request can neither consume nor ride an approval.
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
            # Atomically spend the approval; exactly one of any concurrent callers
            # proceeds, and the lock is released before the tool executes.
            if not approval.try_consume():
                sm.transition_to(ExecutionState.FAILED, "Approval has already been consumed.")
                return (
                    ToolResult.fail(
                        tool=tool_name,
                        error="APPROVAL_CONSUMED",
                        message="This confirmation was already used. Please confirm the action again."
                    ),
                    sm
                )

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
        unused.  Nothing about the original request text is re-interpreted, and
        the request is rebuilt only from the token's frozen snapshot -- never from
        caller-mutable fields.
        """
        tool_name, arguments = approval.bound_action()
        return self.execute_tool(
            tool_name=tool_name,
            raw_args=arguments,
            approval=approval
        )

"""
Safe Tool Executor for Vector Desktop AI Assistant.
Orchestrates the 7-stage state machine, argument validation, and permission checks
before executing any local desktop tool.
"""

from typing import Any, Dict, Tuple
from uuid import uuid4

from app.config.constants import ExecutionState
from app.core.state_machine import ExecutionStateMachine
from app.security.permissions import PermissionEngine
from app.security.validator import ArgumentValidator
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry, get_tool_registry


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
        confirmed_by_user: bool = False
    ) -> Tuple[ToolResult, ExecutionStateMachine]:
        """
        Executes a tool by name with safety checks.
        Returns a tuple of (ToolResult, ExecutionStateMachine).
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
        if perm_res.requires_confirmation and not confirmed_by_user:
            sm.transition_to(ExecutionState.CONFIRMATION, "Awaiting explicit user confirmation.")
            return (
                ToolResult.fail(
                    tool=tool_name,
                    error="CONFIRMATION_REQUIRED",
                    message=f"Action '{tool_name}' requires confirmation: {perm_res.reason}"
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

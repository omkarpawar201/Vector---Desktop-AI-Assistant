"""
Unit tests for ToolRegistry, CapabilityRegistry, ExecutionStateMachine, and SafeToolExecutor.
"""

import pytest
from app.config.constants import ExecutionState, PermissionLevel
from app.core.state_machine import ExecutionStateMachine
from app.security.permissions import PermissionEngine
from app.tools.base import BaseTool, ToolResult
from app.tools.executor import SafeToolExecutor
from app.tools.registry import CapabilityRegistry, ToolRegistry


class DummySafeTool(BaseTool):
    name = "dummy_safe"
    description = "Dummy safe tool"
    parameters = {"type": "object", "properties": {"val": {"type": "integer"}}, "required": ["val"]}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs):
        return ToolResult.ok(tool=self.name, data={"res": kwargs.get("val") * 2})


class DummyDangerousTool(BaseTool):
    name = "dummy_dangerous"
    description = "Dummy dangerous tool"
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.DANGEROUS

    def execute(self, **kwargs):
        return ToolResult.ok(tool=self.name, message="Dangerous action done")


def test_capability_registry():
    cap = CapabilityRegistry()
    assert cap.is_supported("volume_control") is True
    assert cap.is_supported("app_management") is True
    assert cap.is_supported("unknown_cap") is False

    cap.register_capability("custom_cap", "Custom test capability")
    assert cap.is_supported("custom_cap") is True


def test_tool_registry():
    registry = ToolRegistry()
    registry.clear()

    tool = DummySafeTool()
    registry.register(tool)

    assert registry.get("dummy_safe") is tool
    assert len(registry.list_tools()) == 1

    schemas = registry.export_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "dummy_safe"


def test_execution_state_machine():
    sm = ExecutionStateMachine(request_id="123", tool_name="dummy_safe")
    assert sm.current_state == ExecutionState.REQUESTED

    # Valid transitions
    assert sm.transition_to(ExecutionState.PARSED) is True
    assert sm.transition_to(ExecutionState.VALIDATED) is True
    assert sm.transition_to(ExecutionState.PERMISSION_CHECK) is True
    assert sm.transition_to(ExecutionState.EXECUTING) is True
    assert sm.transition_to(ExecutionState.SUCCESS) is True
    assert sm.is_terminal is True

    # Invalid transition from terminal state
    assert sm.transition_to(ExecutionState.EXECUTING) is False


def test_safe_tool_executor():
    registry = ToolRegistry()
    registry.clear()
    tool_safe = DummySafeTool()
    tool_dang = DummyDangerousTool()
    registry.register(tool_safe)
    registry.register(tool_dang)

    executor = SafeToolExecutor(registry=registry)

    # 1. Successful execution
    res, sm = executor.execute_tool("dummy_safe", {"val": 10})
    assert res.success is True
    assert res.data["res"] == 20
    assert sm.current_state == ExecutionState.SUCCESS

    # 2. Missing parameter validation failure
    res_val, sm_val = executor.execute_tool("dummy_safe", {})
    assert res_val.success is False
    assert res_val.error == "INVALID_ARGUMENTS"
    assert sm_val.current_state == ExecutionState.FAILED

    # 3. Confirmation required for DANGEROUS tool
    res_dang, sm_dang = executor.execute_tool("dummy_dangerous", {}, confirmed_by_user=False)
    assert res_dang.success is False
    assert res_dang.error == "CONFIRMATION_REQUIRED"
    assert sm_dang.current_state == ExecutionState.CONFIRMATION

    # 4. Confirmed execution for DANGEROUS tool
    res_conf, sm_conf = executor.execute_tool("dummy_dangerous", {}, confirmed_by_user=True)
    assert res_conf.success is True
    assert sm_conf.current_state == ExecutionState.SUCCESS

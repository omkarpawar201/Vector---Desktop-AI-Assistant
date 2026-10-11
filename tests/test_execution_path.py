"""
Windows tool-execution smoke tests for the Vector execution path.

These drive the shipped pipeline end to end:

    SafeToolExecutor -> ToolRegistry -> ArgumentValidator -> PermissionEngine
    -> (CONFIRMATION) -> tool.execute()

over the real registered tools, so the wiring between validation, permission
gating, confirmation and execution is exercised instead of the tools in
isolation.  External effects are mocked and all filesystem work is confined to
``tmp_path``: nothing here launches a program, changes system settings, touches
the network, reaches Gemini, or performs a destructive operation.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from app.config.constants import ExecutionState, PermissionLevel
from app.security.permissions import PermissionEngine
from app.tools.applications import launcher as launcher_mod
from app.tools.base import BaseTool, ToolResult
from app.tools.executor import SafeToolExecutor
from app.tools.registry import ToolRegistry, register_all_default_tools
from app.tools.system import system_info as system_info_mod


class _BlockedTool(BaseTool):
    """Test-only tool pinned to the BLOCKED permission level."""

    name = "test_blocked"
    description = "Test-only blocked tool."
    parameters = {"type": "object", "properties": {}, "required": []}
    permission_level = PermissionLevel.BLOCKED

    def execute(self, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("a BLOCKED tool was executed")


@pytest.fixture()
def executor():
    """Executor over the canonical default registry plus the blocked probe."""
    registry = ToolRegistry()
    registry.clear()
    register_all_default_tools(registry)
    registry.register(_BlockedTool())
    try:
        yield SafeToolExecutor(registry=registry, permission_engine=PermissionEngine())
    finally:
        registry._tools.pop("test_blocked", None)


# --------------------------------------------------------------------------
# File search / file info
# --------------------------------------------------------------------------

def test_search_files_via_executor_finds_temp_file(executor, tmp_path):
    target = tmp_path / "quarterly_report.txt"
    target.write_text("q3 numbers")
    (tmp_path / "notes.md").write_text("unrelated")

    res, sm = executor.execute_tool(
        "search_files", {"query": "quarterly", "directory": str(tmp_path)}
    )

    assert res.success is True
    assert sm.current_state == ExecutionState.SUCCESS
    assert res.data["count"] >= 1
    assert res.data["search_directory"] == str(tmp_path)
    assert any(Path(m).name == "quarterly_report.txt" for m in res.data["matches"])


def test_search_files_missing_query_is_rejected(executor):
    res, sm = executor.execute_tool("search_files", {})

    assert res.success is False
    assert res.error == "INVALID_ARGUMENTS"
    assert sm.current_state == ExecutionState.FAILED


def test_get_file_info_via_executor_on_temp_file(executor, tmp_path):
    target = tmp_path / "note.txt"
    target.write_bytes(b"x" * 2048)

    res, sm = executor.execute_tool("get_file_info", {"path": str(target)})

    assert res.success is True
    assert sm.current_state == ExecutionState.SUCCESS
    assert res.data["name"] == "note.txt"
    assert res.data["is_dir"] is False
    assert res.data["size_kb"] == 2.0


# --------------------------------------------------------------------------
# System information (read-only; external effect mocked)
# --------------------------------------------------------------------------

class _FakeMem:
    percent = 42.0
    used = 8 * (1024 ** 3)
    total = 16 * (1024 ** 3)
    available = 8 * (1024 ** 3)


class _FakeDisk:
    percent = 55.0
    free = 100 * (1024 ** 3)
    total = 500 * (1024 ** 3)


class _FakeBattery:
    percent = 88
    power_plugged = True


class _FakePsutil:
    @staticmethod
    def cpu_percent(interval=0.0):
        return 7.5

    @staticmethod
    def virtual_memory():
        return _FakeMem()

    @staticmethod
    def disk_usage(_path):
        return _FakeDisk()

    @staticmethod
    def sensors_battery():
        return _FakeBattery()


def test_get_system_stats_via_executor_with_mocked_psutil(executor, monkeypatch):
    monkeypatch.setattr(system_info_mod, "psutil", _FakePsutil)

    res, sm = executor.execute_tool("get_system_stats", {})

    assert res.success is True
    assert sm.current_state == ExecutionState.SUCCESS
    assert res.data["cpu_percent"] == 7.5
    assert res.data["memory_percent"] == 42.0
    assert res.data["battery_percent"] == 88


def test_get_memory_usage_via_executor_is_read_only(executor):
    res, sm = executor.execute_tool("get_memory_usage", {})

    assert res.success is True
    assert sm.current_state == ExecutionState.SUCCESS
    assert {"used_gb", "total_gb", "percent"} <= set(res.data)


# --------------------------------------------------------------------------
# Application management (launch effect mocked; close matches nothing)
# --------------------------------------------------------------------------

def test_launch_app_resolves_alias_without_launching(executor, monkeypatch):
    calls = []

    class _RecordingPopen:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(launcher_mod.subprocess, "Popen", _RecordingPopen)

    res, sm = executor.execute_tool("launch_app", {"name": "chrome"})

    assert res.success is True
    assert sm.current_state == ExecutionState.SUCCESS
    assert res.data["target"] == "chrome.exe"
    assert len(calls) == 1, "the real Popen path must be replaced, not exercised"


def test_launch_app_blocks_shell_injection(executor):
    res, sm = executor.execute_tool("launch_app", {"name": "calc; rm -rf /"})

    assert res.success is False
    assert res.error == "INVALID_APP_NAME"
    assert sm.current_state == ExecutionState.FAILED


def test_close_app_matching_nothing_reports_not_running(executor):
    unique = f"zz_no_such_app_{uuid.uuid4().hex}"

    res, sm = executor.execute_tool("close_app", {"name": unique})

    assert res.success is False
    assert res.error == "APP_NOT_RUNNING"
    assert sm.current_state == ExecutionState.FAILED


# --------------------------------------------------------------------------
# Permission gating / confirmation
# --------------------------------------------------------------------------

def test_dangerous_tool_requires_confirmation_and_never_executes(executor):
    res, sm = executor.execute_tool("shutdown_pc", {}, confirmed_by_user=False)

    assert res.success is False
    assert res.error == "CONFIRMATION_REQUIRED"
    assert sm.current_state == ExecutionState.CONFIRMATION


def test_confirm_tool_requires_confirmation(executor):
    res, sm = executor.execute_tool("sleep_pc", {}, confirmed_by_user=False)

    assert res.success is False
    assert res.error == "CONFIRMATION_REQUIRED"
    assert sm.current_state == ExecutionState.CONFIRMATION


def test_blocked_tool_is_rejected(executor):
    res, sm = executor.execute_tool("test_blocked", {})

    assert res.success is False
    assert res.error == "PERMISSION_BLOCKED"
    assert sm.current_state == ExecutionState.FAILED


def test_unknown_tool_reports_not_found(executor):
    res, sm = executor.execute_tool("no_such_tool_xyz", {})

    assert res.success is False
    assert res.error == "TOOL_NOT_FOUND"
    assert sm.current_state == ExecutionState.FAILED


# --------------------------------------------------------------------------
# Argument validation (rejected before execution -- no system change)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tool_name,args",
    [
        ("set_volume", {"level": 150}),
        ("set_volume", {"level": -5}),
        ("move_window", {"x": 10}),
    ],
)
def test_invalid_arguments_rejected_before_execution(executor, tool_name, args):
    res, sm = executor.execute_tool(tool_name, args)

    assert res.success is False
    assert res.error == "INVALID_ARGUMENTS"
    assert sm.current_state == ExecutionState.FAILED


# --------------------------------------------------------------------------
# Terminal (whitelist + injection guards, no network)
# --------------------------------------------------------------------------

def test_terminal_whitelisted_command_runs(executor):
    res, sm = executor.execute_tool("execute_terminal_command", {"command": "python --version"})

    assert res.success is True
    assert sm.current_state == ExecutionState.SUCCESS
    assert "Python" in res.data["stdout"]


def test_terminal_blocks_chaining_operator(executor):
    res, sm = executor.execute_tool(
        "execute_terminal_command", {"command": "git status && echo hacked"}
    )

    assert res.success is False
    assert res.error == "SHELL_OPERATOR_PROHIBITED"
    assert sm.current_state == ExecutionState.FAILED


def test_terminal_blocks_non_whitelisted_binary(executor):
    res, sm = executor.execute_tool(
        "execute_terminal_command", {"command": "curl https://example.com"}
    )

    assert res.success is False
    assert res.error == "COMMAND_NOT_WHITELISTED"
    assert sm.current_state == ExecutionState.FAILED

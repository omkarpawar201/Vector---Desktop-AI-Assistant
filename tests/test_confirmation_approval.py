"""
Tests for the action-bound, single-use confirmation workflow (Milestone B).

The old contract passed a global ``confirmed=True`` flag back through the whole
router, which re-resolved the original text and could execute a different action
than the one shown, and which let one approval authorize any later call.  Here we
pin the new contract instead:

* confirming executes exactly the displayed (tool, validated arguments);
* an approval is single-use and cannot be replayed;
* an approval never bypasses validation or the permission policy;
* a Tier 2 loop that already ran safe steps is not replayed on approval, so no
  completed side effect runs twice.

Nothing here launches a program, changes a system setting, touches the network,
or loads a model: every tool is a counting fake and the router's model clients
are replaced with deterministic stand-ins.
"""

import threading

import pytest

from app.config.constants import ExecutionState, IntentTier, PermissionLevel
from app.config.settings import Settings
from app.core.router import IntentRouter
from app.needle.client import NeedleResult
from app.tools.base import BaseTool, ToolResult
from app.tools.executor import PendingApproval, SafeToolExecutor
from app.tools.registry import ToolRegistry, register_all_default_tools


# --------------------------------------------------------------------------
# Fixtures and fakes
# --------------------------------------------------------------------------

class _CountingTool(BaseTool):
    """A tool that counts executions and records the arguments it received."""

    def __init__(self, name, permission_level, parameters=None):
        self.name = name
        self.description = f"test-only {name}"
        self.parameters = parameters or {
            "type": "object", "properties": {}, "required": []
        }
        self.permission_level = permission_level
        self.executions = 0
        self.seen_args = []
        self._lock = threading.Lock()

    def execute(self, **kwargs):
        with self._lock:
            self.executions += 1
            self.seen_args.append(dict(kwargs))
        return ToolResult.ok(tool=self.name, data={"done": True}, message=f"{self.name} executed")


_TARGET_SCHEMA = {
    "type": "object",
    "properties": {"target": {"type": "string"}},
    "required": ["target"],
}

_NESTED_SCHEMA = {
    "type": "object",
    "properties": {"options": {"type": "object"}},
    "required": ["options"],
}


@pytest.fixture()
def registry():
    """A fresh registry of fakes; restores the shipped tools afterwards."""
    reg = ToolRegistry()
    reg.clear()
    yield reg
    reg.clear()
    register_all_default_tools(reg)


# --------------------------------------------------------------------------
# Executor-level approval semantics
# --------------------------------------------------------------------------

def test_confirmation_required_reports_validated_arguments(registry):
    confirm = _CountingTool("confirm_tool", PermissionLevel.CONFIRM, _TARGET_SCHEMA)
    registry.register(confirm)
    executor = SafeToolExecutor(registry=registry)

    res, sm = executor.execute_tool("confirm_tool", {"target": "x"})

    assert res.success is False
    assert res.error == "CONFIRMATION_REQUIRED"
    assert sm.current_state == ExecutionState.CONFIRMATION
    assert res.data["validated_arguments"] == {"target": "x"}
    assert confirm.executions == 0


def test_matching_approval_executes_and_is_consumed(registry):
    confirm = _CountingTool("confirm_tool", PermissionLevel.CONFIRM, _TARGET_SCHEMA)
    registry.register(confirm)
    executor = SafeToolExecutor(registry=registry)

    approval = PendingApproval(tool_name="confirm_tool", arguments={"target": "x"})
    res, sm = executor.execute_tool("confirm_tool", {"target": "x"}, approval=approval)

    assert res.success is True
    assert sm.current_state == ExecutionState.SUCCESS
    assert confirm.executions == 1
    assert confirm.seen_args == [{"target": "x"}]
    assert approval.consumed is True


def test_consumed_approval_cannot_be_replayed(registry):
    confirm = _CountingTool("confirm_tool", PermissionLevel.CONFIRM, _TARGET_SCHEMA)
    registry.register(confirm)
    executor = SafeToolExecutor(registry=registry)

    approval = PendingApproval(tool_name="confirm_tool", arguments={"target": "x"})
    first, _ = executor.execute_tool("confirm_tool", {"target": "x"}, approval=approval)
    second, sm = executor.execute_tool("confirm_tool", {"target": "x"}, approval=approval)

    assert first.success is True
    assert second.success is False
    assert second.error == "APPROVAL_CONSUMED"
    assert sm.current_state == ExecutionState.FAILED
    assert confirm.executions == 1, "a consumed approval must never execute a second time"


def test_mismatched_arguments_are_refused(registry):
    confirm = _CountingTool("confirm_tool", PermissionLevel.CONFIRM, _TARGET_SCHEMA)
    registry.register(confirm)
    executor = SafeToolExecutor(registry=registry)

    approved_for_y = PendingApproval(tool_name="confirm_tool", arguments={"target": "y"})
    res, sm = executor.execute_tool("confirm_tool", {"target": "x"}, approval=approved_for_y)

    assert res.success is False
    assert res.error == "APPROVAL_MISMATCH"
    assert sm.current_state == ExecutionState.FAILED
    assert confirm.executions == 0


def test_mismatched_tool_is_refused(registry):
    confirm = _CountingTool("confirm_tool", PermissionLevel.CONFIRM, _TARGET_SCHEMA)
    registry.register(confirm)
    executor = SafeToolExecutor(registry=registry)

    wrong_tool = PendingApproval(tool_name="some_other_tool", arguments={"target": "x"})
    res, sm = executor.execute_tool("confirm_tool", {"target": "x"}, approval=wrong_tool)

    assert res.success is False
    assert res.error == "APPROVAL_MISMATCH"
    assert confirm.executions == 0


def test_approval_does_not_bypass_validation(registry):
    confirm = _CountingTool("confirm_tool", PermissionLevel.CONFIRM, _TARGET_SCHEMA)
    registry.register(confirm)
    executor = SafeToolExecutor(registry=registry)

    approval = PendingApproval(tool_name="confirm_tool", arguments={})
    res, sm = executor.execute_tool("confirm_tool", {}, approval=approval)

    assert res.success is False
    assert res.error == "INVALID_ARGUMENTS"
    assert sm.current_state == ExecutionState.FAILED
    assert confirm.executions == 0


def test_approval_does_not_bypass_blocked_permission(registry):
    blocked = _CountingTool("blocked_tool", PermissionLevel.BLOCKED)
    registry.register(blocked)
    executor = SafeToolExecutor(registry=registry)

    approval = PendingApproval(tool_name="blocked_tool", arguments={})
    res, sm = executor.execute_tool("blocked_tool", {}, approval=approval)

    assert res.success is False
    assert res.error == "PERMISSION_BLOCKED"
    assert sm.current_state == ExecutionState.FAILED
    assert blocked.executions == 0


def test_execute_approved_runs_exact_action(registry):
    confirm = _CountingTool("confirm_tool", PermissionLevel.CONFIRM, _TARGET_SCHEMA)
    registry.register(confirm)
    executor = SafeToolExecutor(registry=registry)

    approval = PendingApproval(tool_name="confirm_tool", arguments={"target": "x"})
    res, sm = executor.execute_approved(approval)

    assert res.success is True
    assert sm.current_state == ExecutionState.SUCCESS
    assert confirm.seen_args == [{"target": "x"}]
    assert approval.consumed is True


def test_safe_tool_runs_without_approval(registry):
    safe = _CountingTool("safe_tool", PermissionLevel.SAFE)
    registry.register(safe)
    executor = SafeToolExecutor(registry=registry)

    res, sm = executor.execute_tool("safe_tool", {})

    assert res.success is True
    assert sm.current_state == ExecutionState.SUCCESS
    assert safe.executions == 1


# --------------------------------------------------------------------------
# Milestone B.1: atomic single-use consumption and tamper resistance
# --------------------------------------------------------------------------

def test_simultaneous_approvals_execute_exactly_once(registry):
    confirm = _CountingTool("confirm_tool", PermissionLevel.CONFIRM, _TARGET_SCHEMA)
    registry.register(confirm)
    executor = SafeToolExecutor(registry=registry)

    approval = PendingApproval(tool_name="confirm_tool", arguments={"target": "x"})

    workers = 8
    barrier = threading.Barrier(workers)
    outcomes = []
    outcomes_lock = threading.Lock()

    def _race():
        barrier.wait()
        res, _sm = executor.execute_approved(approval)
        with outcomes_lock:
            outcomes.append("OK" if res.success else res.error)

    threads = [threading.Thread(target=_race) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert confirm.executions == 1, "a single approval must authorize exactly one execution"
    assert outcomes.count("OK") == 1
    assert outcomes.count("APPROVAL_CONSUMED") == workers - 1


def test_mutating_original_arguments_does_not_change_approved_action(registry):
    confirm = _CountingTool("confirm_tool", PermissionLevel.CONFIRM, _TARGET_SCHEMA)
    registry.register(confirm)
    executor = SafeToolExecutor(registry=registry)

    source = {"target": "x"}
    approval = PendingApproval(tool_name="confirm_tool", arguments=source)
    source["target"] = "evil"

    executor.execute_approved(approval)

    assert confirm.seen_args == [{"target": "x"}], "post-display mutation must not affect the bound action"


def test_mutating_nested_arguments_does_not_change_approved_action(registry):
    confirm = _CountingTool("confirm_tool", PermissionLevel.CONFIRM, _NESTED_SCHEMA)
    registry.register(confirm)
    executor = SafeToolExecutor(registry=registry)

    source = {"options": {"level": 1}}
    approval = PendingApproval(tool_name="confirm_tool", arguments=source)
    source["options"]["level"] = 999

    executor.execute_approved(approval)

    assert confirm.seen_args == [{"options": {"level": 1}}], "nested mutation must not reach the bound action"


def test_returned_arguments_are_defensive_copies(registry):
    confirm = _CountingTool("confirm_tool", PermissionLevel.CONFIRM, _NESTED_SCHEMA)
    registry.register(confirm)
    executor = SafeToolExecutor(registry=registry)

    approval = PendingApproval(tool_name="confirm_tool", arguments={"options": {"level": 1}})

    leaked = approval.arguments
    leaked["options"]["level"] = 42
    bound_name, bound_args = approval.bound_action()
    bound_args["options"]["level"] = 42

    executor.execute_approved(approval)

    assert approval.tool_name == "confirm_tool"
    assert confirm.seen_args == [{"options": {"level": 1}}], "copies handed out must not alter the bound action"


def test_approval_is_immutable_after_creation(registry):
    approval = PendingApproval(tool_name="confirm_tool", arguments={"target": "x"})

    with pytest.raises(AttributeError):
        approval.tool_name = "other_tool"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        approval.arguments = {}  # type: ignore[misc]
    with pytest.raises(AttributeError):
        approval.consumed = True  # type: ignore[misc]
    with pytest.raises(AttributeError):
        approval._consumed = True  # type: ignore[misc]

    assert approval.tool_name == "confirm_tool"
    assert approval.arguments == {"target": "x"}
    assert approval.consumed is False


def test_mismatch_does_not_consume_the_approval(registry):
    confirm = _CountingTool("confirm_tool", PermissionLevel.CONFIRM, _TARGET_SCHEMA)
    registry.register(confirm)
    executor = SafeToolExecutor(registry=registry)

    approval = PendingApproval(tool_name="confirm_tool", arguments={"target": "x"})

    # A request the token does not authorize must be refused without spending it.
    mismatch, _sm = executor.execute_tool("confirm_tool", {"target": "y"}, approval=approval)
    assert mismatch.error == "APPROVAL_MISMATCH"
    assert approval.consumed is False

    # The same token still authorizes its own, matching action exactly once.
    ok, _sm = executor.execute_approved(approval)
    assert ok.success is True
    assert confirm.executions == 1


# --------------------------------------------------------------------------
# Router-level binding: approve executes the displayed action, no reroute
# --------------------------------------------------------------------------

class _FakeNeedleClient:
    """Deterministic local model stand-in that always escalates to Tier 2."""

    def __init__(self):
        self.onnx_calls = 0
        self.candidate_calls = 0

    def predict_custom_onnx_intent(self, user_input):
        self.onnx_calls += 1
        return NeedleResult(reason="no keyword hit")

    def select_candidates(self, user_input):
        self.candidate_calls += 1
        return None


def test_tier0_confirmation_is_bound_and_executes_only_on_approval(registry):
    shutdown = _CountingTool("shutdown_pc", PermissionLevel.DANGEROUS)
    registry.register(shutdown)
    executor = SafeToolExecutor(registry=registry)

    router = IntentRouter(
        settings=Settings(enable_gemini=False),
        registry=registry,
        executor=executor,
        needle_client=_FakeNeedleClient(),
    )

    result = router.route_and_execute("shut down the computer")

    assert result.tier_used == IntentTier.TIER_0_DIRECT
    assert result.requires_confirmation is True
    assert result.tool_name == "shutdown_pc"
    assert result.pending_approval is not None
    assert result.pending_approval.tool_name == "shutdown_pc"
    assert result.pending_approval.arguments == {}
    assert shutdown.executions == 0, "the dangerous tool must not run before approval"

    approved = router.execute_approved(result.pending_approval)

    assert approved.success is True
    assert shutdown.executions == 1, "approval must execute the displayed action exactly once"

    # The bound action is single-use: a replay is refused and does not re-run.
    replay = router.execute_approved(result.pending_approval)
    assert replay.success is False
    assert replay.error == "APPROVAL_CONSUMED"
    assert shutdown.executions == 1


def test_execute_approved_does_not_reroute(registry):
    shutdown = _CountingTool("shutdown_pc", PermissionLevel.DANGEROUS)
    registry.register(shutdown)
    executor = SafeToolExecutor(registry=registry)

    router = IntentRouter(
        settings=Settings(enable_gemini=False),
        registry=registry,
        executor=executor,
        needle_client=_FakeNeedleClient(),
    )

    calls = []
    original = router.route_and_execute

    def _spy(query):
        calls.append(query)
        return original(query)

    router.route_and_execute = _spy  # type: ignore[assignment]

    result = router.route_and_execute("shut down the computer")
    assert len(calls) == 1

    router.execute_approved(result.pending_approval)

    assert len(calls) == 1, "executing the approved action must not route the original query again"


# --------------------------------------------------------------------------
# Tier 2: completed steps are not replayed when a gated call is approved
# --------------------------------------------------------------------------

class _FakeGeminiClient:
    """Mirrors the real loop: run a safe tool, then stop at a gated tool."""

    def __init__(self, executor):
        self.executor = executor
        self.loop_calls = 0

    def execute_reasoning_loop(self, user_query):
        self.loop_calls += 1
        calls = []

        r1, _ = self.executor.execute_tool("count_tool", {})
        calls.append({
            "step": 1, "tool": "count_tool", "arguments": {},
            "success": r1.success, "message": r1.message,
            "data": r1.data, "error": r1.error,
        })

        r2, _ = self.executor.execute_tool("confirm_tool", {"target": "x"})
        calls.append({
            "step": 2, "tool": "confirm_tool", "arguments": {"target": "x"},
            "success": r2.success, "message": r2.message,
            "data": r2.data, "error": r2.error,
        })

        return {
            "success": False,
            "requires_confirmation": True,
            "tool_name": "confirm_tool",
            "arguments": {"target": "x"},
            "validated_arguments": (r2.data or {}).get("validated_arguments", {"target": "x"}),
            "response": r2.message,
            "tool_calls": calls,
        }


def test_tier2_approval_does_not_repeat_completed_steps(registry):
    count_tool = _CountingTool("count_tool", PermissionLevel.SAFE)
    confirm_tool = _CountingTool("confirm_tool", PermissionLevel.CONFIRM, _TARGET_SCHEMA)
    registry.register(count_tool)
    registry.register(confirm_tool)

    executor = SafeToolExecutor(registry=registry)
    gemini = _FakeGeminiClient(executor)

    router = IntentRouter(
        settings=Settings(enable_gemini=True, gemini_api_key="test-key"),
        registry=registry,
        executor=executor,
        needle_client=_FakeNeedleClient(),
        gemini_client=gemini,
    )

    result = router.route_and_execute("tell me a fun fact about the ocean")

    assert result.tier_used == IntentTier.TIER_2_GEMINI
    assert result.requires_confirmation is True
    assert result.pending_approval is not None
    assert result.pending_approval.tool_name == "confirm_tool"
    assert result.pending_approval.arguments == {"target": "x"}

    # The safe step ran once inside the loop and is reported as completed.
    assert count_tool.executions == 1
    assert confirm_tool.executions == 0
    assert [tr.tool for tr in result.completed_results] == ["count_tool"]
    assert gemini.loop_calls == 1

    approved = router.execute_approved(result.pending_approval)

    assert approved.success is True
    assert confirm_tool.executions == 1
    assert count_tool.executions == 1, "the loop's completed step must not be replayed"
    assert gemini.loop_calls == 1, "the reasoning loop must not be resumed"

    replay = router.execute_approved(result.pending_approval)
    assert replay.error == "APPROVAL_CONSUMED"
    assert confirm_tool.executions == 1


def test_displayed_confirmation_action_matches_bound_action(registry):
    count_tool = _CountingTool("count_tool", PermissionLevel.SAFE)
    confirm_tool = _CountingTool("confirm_tool", PermissionLevel.CONFIRM, _TARGET_SCHEMA)
    registry.register(count_tool)
    registry.register(confirm_tool)

    executor = SafeToolExecutor(registry=registry)
    gemini = _FakeGeminiClient(executor)

    router = IntentRouter(
        settings=Settings(enable_gemini=True, gemini_api_key="test-key"),
        registry=registry,
        executor=executor,
        needle_client=_FakeNeedleClient(),
        gemini_client=gemini,
    )

    result = router.route_and_execute("tell me a fun fact about the ocean")

    assert result.requires_confirmation is True
    assert result.pending_approval is not None
    # What the dialog displays is exactly what the token binds.
    assert result.tool_name == result.pending_approval.tool_name == "confirm_tool"
    assert result.raw_arguments == result.pending_approval.arguments == {"target": "x"}

    # Tampering with the displayed copies cannot reach the bound action.
    result.raw_arguments["target"] = "evil"
    result.pending_approval.arguments["target"] = "evil"

    approved = router.execute_approved(result.pending_approval)

    assert approved.success is True
    assert confirm_tool.seen_args == [{"target": "x"}]

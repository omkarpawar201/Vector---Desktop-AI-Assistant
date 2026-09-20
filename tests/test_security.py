"""
Unit tests for PermissionEngine, ArgumentValidator, and ToolResult.
"""

from app.config.constants import PermissionLevel
from app.config.settings import Settings
from app.security.permissions import PermissionEngine
from app.security.validator import ArgumentValidator
from app.tools.base import ToolResult


def test_permission_engine_evaluation():
    """Test security decision matrix for permission levels."""
    settings = Settings(require_confirmation_for_dangerous=True)
    engine = PermissionEngine(settings=settings)

    # SAFE tool
    d_safe = engine.evaluate(PermissionLevel.SAFE)
    assert d_safe.allowed is True
    assert d_safe.requires_confirmation is False

    # CONFIRM tool
    d_confirm = engine.evaluate(PermissionLevel.CONFIRM)
    assert d_confirm.allowed is True
    assert d_confirm.requires_confirmation is True

    # DANGEROUS tool with policy = True
    d_dangerous = engine.evaluate(PermissionLevel.DANGEROUS)
    assert d_dangerous.allowed is True
    assert d_dangerous.requires_confirmation is True

    # BLOCKED tool
    d_blocked = engine.evaluate(PermissionLevel.BLOCKED)
    assert d_blocked.allowed is False
    assert d_blocked.requires_confirmation is False


def test_argument_validator_pass():
    """Test successful parameter validation with type coercion and bounds."""
    schema = {
        "type": "object",
        "properties": {
            "level": {"type": "integer", "minimum": 0, "maximum": 100},
            "name": {"type": "string"},
            "force": {"type": "boolean"}
        },
        "required": ["level"]
    }

    res = ArgumentValidator.validate(schema, {"level": 40, "name": "Chrome", "force": "true"})
    assert res.valid is True
    assert res.cleaned_args["level"] == 40
    assert res.cleaned_args["name"] == "Chrome"
    assert res.cleaned_args["force"] is True
    assert len(res.errors) == 0


def test_argument_validator_failures():
    """Test parameter validation failures for missing required fields and out-of-bound values."""
    schema = {
        "type": "object",
        "properties": {
            "level": {"type": "integer", "minimum": 0, "maximum": 100}
        },
        "required": ["level"]
    }

    # Missing required
    res_missing = ArgumentValidator.validate(schema, {})
    assert res_missing.valid is False
    assert "Missing required parameter" in res_missing.errors[0]

    # Out of bounds
    res_out = ArgumentValidator.validate(schema, {"level": 150})
    assert res_out.valid is False
    assert "exceeds maximum" in res_out.errors[0]


def test_tool_result_factories():
    """Test ToolResult factory methods for success and failure."""
    res_ok = ToolResult.ok(tool="set_volume", data={"level": 40}, message="Volume set to 40%")
    assert res_ok.success is True
    assert res_ok.tool == "set_volume"
    assert res_ok.data == {"level": 40}

    res_fail = ToolResult.fail(tool="launch_app", error="App not found")
    assert res_fail.success is False
    assert res_fail.error == "App not found"

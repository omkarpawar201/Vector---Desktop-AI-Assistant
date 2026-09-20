"""
Unit tests for Vector configuration and constants.
"""

from app.config.constants import ExecutionState, IntentTier, MAX_TOOL_CALLS, PermissionLevel
from app.config.settings import Settings, get_settings


def test_constants_values():
    """Verify core constants and enum definitions."""
    assert PermissionLevel.SAFE.value == "SAFE"
    assert PermissionLevel.BLOCKED.value == "BLOCKED"
    assert ExecutionState.REQUESTED.value == "REQUESTED"
    assert ExecutionState.SUCCESS.value == "SUCCESS"
    assert IntentTier.TIER_0_DIRECT.value == "TIER_0_DIRECT"
    assert MAX_TOOL_CALLS == 10


def test_default_settings():
    """Verify settings defaults and helper properties."""
    settings = Settings(gemini_api_key="", enable_gemini=True)
    assert settings.needle_confidence_threshold == 0.85
    assert settings.is_gemini_available is False  # Empty API key

    settings_with_key = Settings(gemini_api_key="test_key", enable_gemini=True)
    assert settings_with_key.is_gemini_available is True


def test_settings_singleton():
    """Verify get_settings returns a cached instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2

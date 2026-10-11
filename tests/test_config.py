"""
Unit tests for Vector configuration and constants.
"""

from pathlib import Path

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


def test_tuned_needle_default_is_frozen_v4():
    """
    Production Tier 1 selection: the tuned weights *default* must stay the frozen
    V4 archive. Asserting the declared field default (not a constructed instance)
    keeps this independent of any local .env or environment override.
    """
    default = Settings.model_fields["needle_tuned_weights_path"].default
    assert default == "models/needle3_vector_v4.cact"
    assert Path(default).name == "needle3_vector_v4.cact"


def test_no_setting_defaults_to_experimental_v5():
    """
    V5 is experimental and must never be auto-promoted: nothing may silently
    default to a V5 path; it can only be engaged through an explicit override.
    """
    offenders = [
        name
        for name, field in Settings.model_fields.items()
        if isinstance(field.default, str) and "needle3_vector_v5" in field.default
    ]
    assert offenders == [], f"settings default to experimental V5: {offenders}"


def test_tuned_weights_override_is_honored():
    """
    The V4 default is a default, not a hard pin. An explicit override -- even a
    deliberate experimental V5 selection -- is respected, so this guard does not
    forbid legitimate user configuration. init kwargs outrank env/.env values.
    """
    settings = Settings(_env_file=None, needle_tuned_weights_path="models/needle3_vector_v5.cact")
    assert settings.needle_tuned_weights_path == "models/needle3_vector_v5.cact"

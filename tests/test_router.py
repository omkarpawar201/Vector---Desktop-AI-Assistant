"""
Unit tests for IntentRouter pipeline (Tier 0 -> Tier 1 -> Tier 2).
"""

from app.config.constants import IntentTier
from app.config.settings import Settings
from app.core.router import IntentRouter
from app.tools.registry import ToolRegistry
from app.tools.system.volume import GetVolumeTool, SetVolumeTool


def test_intent_router_tier0_hit():
    """Test router matching Tier 0 direct reflex."""
    reg = ToolRegistry()
    reg.clear()
    reg.register(GetVolumeTool())
    reg.register(SetVolumeTool())

    settings = Settings(enable_gemini=False)
    router = IntentRouter(settings=settings, registry=reg)

    route_res = router.route_and_execute("set volume to 30")
    assert route_res.tier_used == IntentTier.TIER_0_DIRECT
    assert route_res.tool_result is not None
    assert route_res.tool_result.success is True


def test_intent_router_offline_fallback():
    """Test router fallback response when Gemini is unavailable."""
    reg = ToolRegistry()
    reg.clear()

    settings = Settings(enable_gemini=False)
    router = IntentRouter(settings=settings, registry=reg)

    route_res = router.route_and_execute("Summarize the complex quantum computing paper")
    assert "Gemini unavailable" in route_res.response_text

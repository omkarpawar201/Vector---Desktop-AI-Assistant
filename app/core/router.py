"""
Multi-Tiered Intent Router for Vector Desktop AI Assistant.
Orchestrates request routing across Tier 0 (Reflexes), Tier 1 (Needle 2), and Tier 2 (Gemini API).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config.constants import IntentTier
from app.config.settings import Settings, get_settings
from app.core.matcher import Tier0Matcher
from app.gemini.client import GeminiClient
from app.needle.client import NeedleClient
from app.tools.base import ToolResult
from app.tools.executor import SafeToolExecutor
from app.tools.registry import ToolRegistry, get_tool_registry


@dataclass
class RouteResult:
    """
    Standardized result returned by the IntentRouter.
    """
    tier_used: IntentTier
    tool_result: Optional[ToolResult] = None
    response_text: str = ""
    requires_confirmation: bool = False
    tool_name: str = ""
    raw_arguments: Dict[str, Any] = field(default_factory=dict)
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)


class IntentRouter:
    """
    Router implementing Tier 0 -> Tier 1 (Needle) -> Tier 2 (Gemini) routing.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        registry: Optional[ToolRegistry] = None,
        executor: Optional[SafeToolExecutor] = None,
        needle_client: Optional[NeedleClient] = None,
        gemini_client: Optional[GeminiClient] = None
    ):
        self.settings = settings or get_settings()
        self.registry = registry or get_tool_registry()
        self.executor = executor or SafeToolExecutor(registry=self.registry)
        self.needle_client = needle_client or NeedleClient(settings=self.settings, registry=self.registry)
        self.gemini_client = gemini_client or GeminiClient(settings=self.settings, registry=self.registry, executor=self.executor)

    def route_and_execute(self, user_query: str, confirmed_by_user: bool = False) -> RouteResult:
        """
        Processes user query through Tier 0 -> Tier 1 -> Tier 2 pipeline.
        """
        if not user_query or not user_query.strip():
            return RouteResult(
                tier_used=IntentTier.TIER_0_DIRECT,
                response_text="Please enter a valid command or request."
            )

        query = user_query.strip()
        trace: List[Dict[str, Any]] = []

        # =====================================================================
        # 1. TIER 0: Direct Matcher (< 5-10 ms Reflexes)
        # =====================================================================
        tier0_match = Tier0Matcher.match(query)
        if tier0_match.matched:
            trace.append({"step": "Tier 0", "matched": True, "tool": tier0_match.tool_name})
            res, sm = self.executor.execute_tool(
                tool_name=tier0_match.tool_name,
                raw_args=tier0_match.arguments,
                confirmed_by_user=confirmed_by_user
            )

            if not res.success and res.error == "CONFIRMATION_REQUIRED":
                return RouteResult(
                    tier_used=IntentTier.TIER_0_DIRECT,
                    tool_result=res,
                    response_text=res.message,
                    requires_confirmation=True,
                    tool_name=tier0_match.tool_name,
                    raw_arguments=tier0_match.arguments,
                    execution_trace=trace
                )

            return RouteResult(
                tier_used=IntentTier.TIER_0_DIRECT,
                tool_result=res,
                response_text=res.message,
                execution_trace=trace
            )

        trace.append({"step": "Tier 0", "matched": False, "reason": tier0_match.reason})

        # =====================================================================
        # 2. TIER 1: Needle 2 Local Intent Model
        # =====================================================================
        needle_res = self.needle_client.predict_intent(query)
        threshold = self.settings.needle_confidence_threshold

        is_tool_supported = bool(self.registry.get(needle_res.tool_name)) if needle_res.tool_name else False

        if needle_res.is_valid and needle_res.confidence >= threshold and is_tool_supported:
            trace.append({
                "step": "Tier 1 Needle 2",
                "matched": True,
                "tool": needle_res.tool_name,
                "confidence": needle_res.confidence
            })

            res, sm = self.executor.execute_tool(
                tool_name=needle_res.tool_name,
                raw_args=needle_res.arguments,
                confirmed_by_user=confirmed_by_user
            )

            if not res.success and res.error == "CONFIRMATION_REQUIRED":
                return RouteResult(
                    tier_used=IntentTier.TIER_1_NEEDLE,
                    tool_result=res,
                    response_text=res.message,
                    requires_confirmation=True,
                    tool_name=needle_res.tool_name,
                    raw_arguments=needle_res.arguments,
                    execution_trace=trace
                )

            return RouteResult(
                tier_used=IntentTier.TIER_1_NEEDLE,
                tool_result=res,
                response_text=res.message,
                execution_trace=trace
            )

        trace.append({
            "step": "Tier 1 Needle 2",
            "matched": False,
            "confidence": needle_res.confidence,
            "reason": needle_res.reason
        })

        # =====================================================================
        # 3. TIER 2: Cloud Gemini API Multi-Tool Reasoning Loop
        # =====================================================================
        if self.settings.enable_gemini and self.settings.is_gemini_available:
            trace.append({"step": "Tier 2 Gemini", "status": "Escalated to Gemini"})
            gemini_output = self.gemini_client.execute_reasoning_loop(
                user_query=query,
                confirmed_by_user=confirmed_by_user
            )

            if gemini_output.get("requires_confirmation"):
                return RouteResult(
                    tier_used=IntentTier.TIER_2_GEMINI,
                    response_text=gemini_output.get("response", ""),
                    requires_confirmation=True,
                    tool_name=gemini_output.get("tool_name", ""),
                    raw_arguments=gemini_output.get("arguments", {}),
                    execution_trace=trace
                )

            return RouteResult(
                tier_used=IntentTier.TIER_2_GEMINI,
                response_text=gemini_output.get("response", ""),
                execution_trace=trace
            )

        # Offline / Fallback when Gemini is disabled or unavailable
        return RouteResult(
            tier_used=IntentTier.TIER_1_NEEDLE,
            response_text="Gemini unavailable. I can still handle local desktop commands (volume, apps, stats, power, media).",
            execution_trace=trace
        )

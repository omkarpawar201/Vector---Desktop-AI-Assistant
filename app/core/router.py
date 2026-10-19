"""
Multi-Tiered Intent Router for Vector Desktop AI Assistant.
Orchestrates Tier 0 (deterministic reflexes), Tier 0.5 (keyword model),
Tier 1 (candidate-routed local Needle 3), and Tier 2 (Gemini API).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config.constants import IntentTier
from app.config.settings import Settings, get_settings
from app.core.context import ContextBuilder
from app.core.matcher import Tier0Matcher
from app.gemini.client import GeminiClient
from app.memory.history import HistoryManager
from app.memory.journal import ExecutionJournal
from app.needle.client import NeedleClient
from app.tools.base import ToolResult
from app.tools.executor import PendingApproval, SafeToolExecutor
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
    #: The exact action a user must approve; only set when confirmation is
    #: required.  Approving it executes this action verbatim -- never a
    #: re-interpretation of the original query.
    pending_approval: Optional[PendingApproval] = None
    #: Tool calls that already ran and must not be repeated, e.g. the safe steps
    #: a Tier 2 loop completed before it reached a confirmation-gated tool.
    completed_results: List[ToolResult] = field(default_factory=list)
    #: True when the answer was rendered incrementally to the UI (streaming).
    #: The UI finalizes the already-open streamed bubble instead of adding a new
    #: message, so the displayed text and the persisted text stay identical.
    streamed: bool = False
    #: True when the user stopped generation before it completed normally.
    cancelled: bool = False
    #: True when the response is incomplete (cancellation or a provider failure
    #: after output began); partial output must never be persisted as complete.
    partial: bool = False


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
        gemini_client: Optional[GeminiClient] = None,
        history_manager: Optional[HistoryManager] = None
    ):
        self.settings = settings or get_settings()
        self.registry = registry or get_tool_registry()
        self.executor = executor or SafeToolExecutor(registry=self.registry)
        self.needle_client = needle_client or NeedleClient(settings=self.settings, registry=self.registry)
        self.gemini_client = gemini_client or GeminiClient(settings=self.settings, registry=self.registry, executor=self.executor)
        # Optional: only consulted for conversational context on Tier 2.  When
        # absent (or when no conversation_id is supplied) context is empty and
        # the pipeline behaves exactly as before.
        self.history_manager = history_manager

    def route_and_execute(
        self,
        user_query: str,
        conversation_id: Optional[str] = None,
        on_chunk: Optional[Any] = None,
        on_stream_start: Optional[Any] = None,
        cancellation_token: Optional[Any] = None,
        request_id: Optional[str] = None,
    ) -> RouteResult:
        """
        Processes user query through Tier 0 -> Tier 1 -> Tier 2 pipeline.
        Prints real-time terminal routing traces.

        ``conversation_id`` optionally names the active conversation so Tier 2
        can receive its recent turns as context.  Callers that omit it (or when
        no history manager is configured) get empty context and the previous
        behaviour, unchanged.  Tier 0 / Tier 1 never read history.

        ``on_chunk`` / ``on_stream_start`` / ``cancellation_token`` / ``request_id``
        are optional streaming hooks.  They are only ever handed to a provider
        that advertises :attr:`supports_streaming`, so providers that cannot
        stream (and callers that predate streaming) keep the exact previous
        behaviour.  The hooks are never consulted by the local tiers.
        """
        if not user_query or not user_query.strip():
            return RouteResult(
                tier_used=IntentTier.TIER_0_DIRECT,
                response_text="Please enter a valid command or request."
            )

        query = user_query.strip()
        trace: List[Dict[str, Any]] = []
        # One Action Journal per routed command; every real execution below is
        # recorded against it (typed, voice and Gemini tool calls alike).
        journal = self._build_journal(conversation_id)

        # =====================================================================
        # 1. TIER 0: Direct Matcher (< 5-10 ms Reflexes)
        # =====================================================================
        tier0_match = Tier0Matcher.match(query)
        if tier0_match.matched:
            print(f"\n[TIER 0: REFLEX] -> \"{query}\" (Tool: {tier0_match.tool_name}, Args: {tier0_match.arguments})")
            trace.append({"step": "Tier 0", "matched": True, "tool": tier0_match.tool_name})
            res, sm = self.executor.execute_tool(
                tool_name=tier0_match.tool_name,
                raw_args=tier0_match.arguments,
                journal=journal,
            )

            if not res.success and res.error == "CONFIRMATION_REQUIRED":
                return self._confirmation(
                    IntentTier.TIER_0_DIRECT, tier0_match.tool_name,
                    tier0_match.arguments, res, trace
                )

            return RouteResult(
                tier_used=IntentTier.TIER_0_DIRECT,
                tool_result=res,
                response_text=res.message,
                execution_trace=trace
            )

        trace.append({"step": "Tier 0", "matched": False, "reason": tier0_match.reason})

        # =====================================================================
        # 2. TIER 0.5: Custom ONNX Local Intent Model (needle_2_custom.onnx; official basename is reserved for real Cactus weights)
        # =====================================================================
        onnx_res = self.needle_client.predict_custom_onnx_intent(query)
        threshold = self.settings.needle_confidence_threshold
        is_onnx_tool_supported = bool(self.registry.get(onnx_res.tool_name)) if onnx_res.tool_name else False

        if onnx_res.is_valid and onnx_res.confidence >= threshold and is_onnx_tool_supported:
            print(f"\n[TIER 0.5: CUSTOM ONNX] -> \"{query}\" (Tool: {onnx_res.tool_name}, Confidence: {onnx_res.confidence:.2f}, Args: {onnx_res.arguments})")
            trace.append({
                "step": "Tier 0.5 Custom ONNX",
                "matched": True,
                "tool": onnx_res.tool_name,
                "confidence": onnx_res.confidence
            })

            res, sm = self.executor.execute_tool(
                tool_name=onnx_res.tool_name,
                raw_args=onnx_res.arguments,
                journal=journal,
            )

            if not res.success and res.error == "CONFIRMATION_REQUIRED":
                return self._confirmation(
                    IntentTier.TIER_0_5_NEEDLE, onnx_res.tool_name,
                    onnx_res.arguments, res, trace
                )

            return RouteResult(
                tier_used=IntentTier.TIER_0_5_NEEDLE,
                tool_result=res,
                response_text=res.message,
                execution_trace=trace
            )

        trace.append({
            "step": "Tier 0.5 Custom ONNX",
            "matched": False,
            "confidence": onnx_res.confidence,
            "reason": onnx_res.reason
        })

        # =====================================================================
        # 3. TIER 1: Local Needle 3 over a pre-selected candidate tool set
        #
        # The tuned model is never handed the full registry. It was fine-tuned
        # with 4-5 tools per example and has no abstention, so a wide list makes
        # it answer confidently with the wrong tool. A deterministic candidate
        # router picks the family first; any call outside it is rejected here and
        # escalates rather than executing.
        # =====================================================================
        selected = self.needle_client.select_candidates(query)
        if selected is None:
            trace.append({
                "step": "Tier 1 Needle",
                "matched": False,
                "reason": "no candidate tool family matched; escalate to Tier 2",
            })
        else:
            needle_res = self.needle_client.predict_cactus_needle_intent(
                query, selected=selected
            )
            is_tool_supported = (
                bool(self.registry.get(needle_res.tool_name))
                if needle_res.tool_name else False
            )

            # Admission is candidate-set membership, not confidence: `needle
            # build` drops the confidence head for local adapters, so a tuned
            # model always reports confidence None and a numeric gate can never
            # pass.
            if needle_res.admitted and is_tool_supported:
                print(f"\n[TIER 1: NEEDLE 3] -> \"{query}\" (Tool: {needle_res.tool_name}, "
                      f"Family: {selected.family}, Candidates: {len(selected.tools)}, "
                      f"Args: {needle_res.arguments})")
                trace.append({
                    "step": "Tier 1 Needle 3",
                    "matched": True,
                    "tool": needle_res.tool_name,
                    "family": selected.family,
                    "candidates": list(selected.tools),
                    "candidate_source": selected.source,
                })

                res, sm = self.executor.execute_tool(
                    tool_name=needle_res.tool_name,
                    raw_args=needle_res.arguments,
                    journal=journal,
                )

                if not res.success and res.error == "CONFIRMATION_REQUIRED":
                    return self._confirmation(
                        IntentTier.TIER_1_NEEDLE, needle_res.tool_name,
                        needle_res.arguments, res, trace
                    )

                return RouteResult(
                    tier_used=IntentTier.TIER_1_NEEDLE,
                    tool_result=res,
                    response_text=res.message,
                    tool_name=needle_res.tool_name,
                    raw_arguments=needle_res.arguments,
                    execution_trace=trace
                )

            trace.append({
                "step": "Tier 1 Needle 3",
                "matched": False,
                "family": selected.family,
                "candidates": list(selected.tools),
                "reason": needle_res.reason or "call not admitted",
            })

        # =====================================================================
        # 3. TIER 2: Cloud Gemini API Multi-Tool Reasoning Loop
        # =====================================================================
        if self.settings.enable_gemini and self.settings.is_gemini_available:
            # A cancellation may already have been requested while the local
            # tiers were being considered; honour it before opening a cloud
            # generation so a stop never starts new work.
            if cancellation_token is not None and cancellation_token.is_cancelled():
                trace.append({"step": "Tier 2 Gemini", "status": "cancelled before start"})
                return RouteResult(
                    tier_used=IntentTier.TIER_2_GEMINI,
                    response_text="Generation stopped.",
                    execution_trace=trace,
                    cancelled=True,
                )

            print(f"\n[TIER 2: GEMINI API] -> \"{query}\" (Escalated for cloud reasoning)")
            trace.append({"step": "Tier 2 Gemini", "status": "Escalated to Gemini"})

            # Context is assembled only on the Tier 2 path, from the active
            # conversation alone.  An empty window keeps the legacy call shape
            # so callers/clients that predate conversation context still work.
            history = self._conversation_context(conversation_id, query)

            # Only expose the streaming hooks to a provider that supports it.
            supports_streaming = bool(
                getattr(self.gemini_client, "supports_streaming", False)
            )
            chunk_cb = on_chunk if supports_streaming else None
            start_cb = on_stream_start if supports_streaming else None

            gemini_output = self.gemini_client.execute_reasoning_loop(
                user_query=query,
                history=history or None,
                journal=journal,
                on_chunk=chunk_cb,
                on_stream_start=start_cb,
                cancellation_token=cancellation_token,
                request_id=request_id,
            )

            completed = self._completed_from_calls(gemini_output.get("tool_calls"))

            if gemini_output.get("requires_confirmation"):
                # Steps the loop already ran are carried forward so the UI can
                # report them; only the gated call is left to approve.  The loop
                # itself is never replayed, so no completed step runs twice.
                return self._confirmation(
                    IntentTier.TIER_2_GEMINI,
                    gemini_output.get("tool_name", ""),
                    gemini_output.get("arguments", {}),
                    gemini_output,
                    trace,
                    validated=gemini_output.get("validated_arguments"),
                    completed=completed,
                    response_text=gemini_output.get("response", ""),
                    streamed=bool(gemini_output.get("streamed")),
                    cancelled=bool(gemini_output.get("cancelled")),
                    partial=bool(gemini_output.get("partial")),
                )

            return RouteResult(
                tier_used=IntentTier.TIER_2_GEMINI,
                response_text=gemini_output.get("response", ""),
                execution_trace=trace,
                completed_results=completed,
                streamed=bool(gemini_output.get("streamed")),
                cancelled=bool(gemini_output.get("cancelled")),
                partial=bool(gemini_output.get("partial")),
            )

        # Offline / Fallback when Gemini is disabled or unavailable
        print(f"\n[OFFLINE FALLBACK] -> \"{query}\" (Gemini disabled/unavailable)")
        return RouteResult(
            tier_used=IntentTier.TIER_0_5_NEEDLE,
            response_text="Gemini unavailable. I can still handle local desktop commands (volume, apps, stats, power, media).",
            execution_trace=trace
        )

    def _build_journal(self, conversation_id: Optional[str]) -> Optional[ExecutionJournal]:
        """
        An Action Journal bound to the active conversation, or ``None``.

        ``None`` when no history manager is configured, so the executor/
        Gemini-client plumbing degrades to a no-op and behaviour is unchanged.
        """
        if self.history_manager is None:
            return None
        return ExecutionJournal(self.history_manager, conversation_id)

    def _conversation_context(
        self,
        conversation_id: Optional[str],
        query: str
    ) -> List[Dict[str, str]]:
        """
        Recent sanitized turns for the active conversation, or ``[]``.

        Fail-safe by design: a missing conversation, empty history, malformed
        row, or storage error all degrade to empty context so the command is
        still routable.
        """
        if not conversation_id or self.history_manager is None:
            return []
        try:
            return ContextBuilder.build_conversation_context(
                conversation_id=conversation_id,
                history_manager=self.history_manager,
                current_query=query,
            )
        except Exception:
            return []

    # =====================================================================
    # Confirmation handoff
    # =====================================================================

    def _confirmation(
        self,
        tier: IntentTier,
        tool_name: str,
        arguments: Dict[str, Any],
        execution_result: Any,
        trace: List[Dict[str, Any]],
        validated: Optional[Dict[str, Any]] = None,
        completed: Optional[List[ToolResult]] = None,
        response_text: str = "",
        streamed: bool = False,
        cancelled: bool = False,
        partial: bool = False,
    ) -> RouteResult:
        """
        Builds a confirmation RouteResult bound to one exact action.

        ``execution_result`` may be a ToolResult (per-tier path) or the Tier 2
        loop's output dict.  The validated arguments -- the ones the executor
        actually compared and will re-validate -- are carried into the pending
        approval so approving replay is guaranteed to match.
        """
        if isinstance(execution_result, ToolResult):
            message = execution_result.message
            validated_args = (execution_result.data or {}).get(
                "validated_arguments", arguments
            )
        else:
            message = response_text or execution_result.get("response", "")
            validated_args = (
                validated if validated is not None else arguments
            )
        if validated_args is None:
            validated_args = arguments

        # Build the token first, then display its bound action, so the action the
        # user sees is exactly the action the token authorizes on approval.
        approval = PendingApproval(tool_name=tool_name, arguments=validated_args)
        return RouteResult(
            tier_used=tier,
            response_text=message,
            requires_confirmation=True,
            tool_name=approval.tool_name,
            raw_arguments=approval.arguments,
            execution_trace=trace,
            pending_approval=approval,
            completed_results=completed or [],
            streamed=streamed,
            cancelled=cancelled,
            partial=partial,
        )

    @staticmethod
    def _completed_from_calls(
        calls: Optional[List[Dict[str, Any]]]
    ) -> List[ToolResult]:
        """Reconstruct executed ToolResults from a Tier 2 loop's call log.

        The confirmation-gated call is excluded: it never ran, so it must not be
        reported as a completed side effect.
        """
        results: List[ToolResult] = []
        for call in calls or []:
            if call.get("error") == "CONFIRMATION_REQUIRED":
                continue
            results.append(
                ToolResult(
                    success=bool(call.get("success")),
                    tool=call.get("tool", ""),
                    data=call.get("data"),
                    message=call.get("message", ""),
                    error=call.get("error"),
                )
            )
        return results

    def execute_approved(
        self,
        approval: PendingApproval,
        conversation_id: Optional[str] = None
    ) -> ToolResult:
        """
        Executes the exact action the user approved.

        No tier runs again and the original query is not re-resolved: only the
        (tool, validated arguments) the user saw are executed, through the full
        executor pipeline.  The approval is single-use, so a second call with the
        same token is refused by the executor.

        ``conversation_id`` names the conversation the approved action belongs to
        so the real execution is journaled alongside the request that produced it.
        """
        journal = self._build_journal(conversation_id)
        res, _sm = self.executor.execute_approved(approval, journal=journal)
        return res

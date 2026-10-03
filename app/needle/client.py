"""
Needle 3 local tool-calling for Vector.

Tier 1 runs the fine-tuned ``needle3_vector_v4`` model, but never over a wide
tool list. Two measured facts drive the design:

1. V4 was fine-tuned with only 4-5 tools per example, in four fixed groups
   (``data/vector_train.jsonl``).  A 17-tool prompt is out-of-distribution.
2. V4 has no abstention.  Given only unrelated tools it still answers, with
   whichever tool is present -- e.g. all six system tools made "play music",
   "open chrome" and "close chrome" all return ``get_volume``.

So the candidate set is chosen first (see :mod:`app.core.candidates`) and V4
only ever chooses inside it.  Any call naming a tool outside that set is
discarded rather than executed, which contains finding 2 to a 4-6 tool blast
radius instead of 31.

Two further correctness details:

* ``auto_date=False``.  ``Needle`` otherwise prepends a ``date: ...`` fact to the
  system prompt, but no training row carries a system field, so the default
  would present a prefix the fine-tune never saw.
* Confidence is not read from the model.  ``needle build`` drops the confidence
  head for local adapters, so a tuned model reports ``confidence: None`` and
  the old ``confidence > 0`` gate could never pass -- Tier 1 was unreachable.
  Admission is decided by candidate-set membership instead, which is a real
  check rather than a fabricated score.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config.settings import Settings, get_settings
from app.core import candidates as candidate_router
from app.tools.registry import ToolRegistry, get_tool_registry


@dataclass
class NeedleResult:
    """
    Result of a local intent decision.

    ``admitted`` is the authority for model-produced calls: it is True only when
    the named tool was inside the candidate set the router offered.  Tier 0.5
    rules set ``confidence`` instead, which is a real heuristic score.
    """

    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""
    admitted: bool = False
    family: str = ""

    @property
    def is_valid(self) -> bool:
        return bool(self.tool_name) and (self.admitted or self.confidence > 0)


class NeedleEnginePool:
    """
    Keeps one live ``Needle`` per candidate set.

    A tuned ``Needle`` is a subprocess holding the 63 MB archive, and its tool
    list is baked in at construction, so engines cannot be shared across
    different candidate sets.  Reusing one per set is what makes a second and
    later query cheap: the tool prefix stays resident instead of being rebuilt.
    """

    def __init__(self, weights: str, generation: int = 3, max_engines: int = 6):
        self._weights = weights
        self._generation = generation
        self._max_engines = max_engines
        self._engines: Dict[Tuple[str, ...], Any] = {}
        self._order: List[Tuple[str, ...]] = []
        self._lock = threading.Lock()

    def ask(self, tools: Sequence[dict], query: str,
            max_new_tokens: int = 64) -> Tuple[Optional[dict], str]:
        """Return ``(call_or_None, note)`` for one query inside ``tools``."""
        if not tools or not query or not query.strip():
            return None, "no candidates or empty query"
        try:
            engine = self._acquire(tools)
        except Exception as exc:
            return None, f"engine unavailable: {exc}"
        try:
            response = engine.complete(query.strip(), max_new_tokens=max_new_tokens)
        except Exception as exc:
            return None, f"decode failed: {exc}"
        calls = response.get("function_calls") or []
        if not calls:
            return None, "model returned no tool call"
        return calls[0], ""

    def _acquire(self, tools: Sequence[dict]):
        key = tuple(sorted(t["name"] for t in tools))
        with self._lock:
            engine = self._engines.get(key)
            if engine is not None:
                return engine
            from needle import Needle

            engine = Needle(
                tools=list(tools),
                weights=self._weights,
                generation=self._generation,
                system="",
                auto_date=False,
            )
            self._engines[key] = engine
            self._order.append(key)
            self._evict_locked()
            return engine

    def _evict_locked(self) -> None:
        while len(self._order) > self._max_engines:
            oldest = self._order.pop(0)
            engine = self._engines.pop(oldest, None)
            if engine is not None:
                try:
                    engine.close()
                except Exception:
                    pass

    def close(self) -> None:
        with self._lock:
            for engine in self._engines.values():
                try:
                    engine.close()
                except Exception:
                    pass
            self._engines.clear()
            self._order.clear()


class NeedleClient:
    """
    Interface to the local Needle 3 model.

    ``predict_custom_onnx_intent`` is the pre-existing Tier 0.5 keyword tier and
    is unchanged.  ``predict_cactus_needle_intent`` is now the candidate-routed
    Tier 1 described in the module docstring.
    """

    def __init__(self, settings: Optional[Settings] = None,
                 registry: Optional[ToolRegistry] = None):
        self.settings = settings or get_settings()
        self.registry = registry or get_tool_registry()
        self._pool: Optional[NeedleEnginePool] = None
        self._pool_failed = False
        self._schema_error: Optional[str] = None

    # -- engine lifecycle ---------------------------------------------------

    def _weights_path(self) -> Optional[Path]:
        configured = getattr(self.settings, "needle_tuned_weights_path", None)
        if not configured:
            return None
        path = Path(str(configured))
        return path if path.is_file() else None

    def _get_pool(self) -> Optional[NeedleEnginePool]:
        if self._pool is not None or self._pool_failed:
            return self._pool
        weights = self._weights_path()
        if weights is None:
            self._pool_failed = True
            return None
        self._pool = NeedleEnginePool(
            weights=str(weights),
            generation=getattr(self.settings, "needle_generation", 3),
        )
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    # -- schemas ------------------------------------------------------------

    def _schemas_by_name(self) -> Dict[str, dict]:
        try:
            return {t["name"]: t for t in self.registry.export_schemas()}
        except Exception as exc:
            # Do not swallow this into an empty dict: an empty schema map makes
            # Tier 1 silently unreachable, which is exactly how it was dead
            # before. The reason string is surfaced by predict_cactus_needle_intent.
            self._schema_error = f"{type(exc).__name__}: {exc}"
            return {}

    def candidate_schemas(self, selected: candidate_router.CandidateSet
                          ) -> List[dict]:
        """Schemas for a candidate set, restricted to trained tools."""
        by_name = self._schemas_by_name()
        if not by_name:
            return []
        out = []
        for name in selected.tools:
            if name in candidate_router.UNTRAINED_TOOLS:
                continue
            schema = by_name.get(name)
            if schema is None:
                continue
            out.append(schema)
        return out

    # -- Tier 0.5 (unchanged keyword tier) ---------------------------------

    def predict_custom_onnx_intent(self, user_input: str) -> NeedleResult:
        """
        Tier 0.5: fast local intent recognition using our custom trained ONNX model (needle_2_custom.onnx; the plain needle_2.onnx basename is reserved for official Cactus engine weights).
        """
        if not user_input or not user_input.strip():
            return NeedleResult(reason="Empty input")

        query = user_input.strip().lower()
        model_note = " (Custom ONNX Model)"

        # Local intent classification & slot extraction engine
        if "open" in query or "launch" in query or "start" in query:
            for app_key in ["chrome", "vscode", "code", "notepad", "calculator", "calc", "spotify", "explorer", "cmd", "edge"]:
                if app_key in query:
                    return NeedleResult(
                        tool_name="launch_app",
                        arguments={"name": app_key},
                        confidence=0.92,
                        reason=f"Custom ONNX model identified app launch intent for '{app_key}'{model_note}"
                    )

        if "close" in query or "terminate" in query or "quit" in query:
            for app_key in ["chrome", "vscode", "code", "notepad", "calculator", "calc", "spotify", "explorer", "cmd", "edge"]:
                if app_key in query:
                    return NeedleResult(
                        tool_name="close_app",
                        arguments={"name": app_key},
                        confidence=0.90,
                        reason=f"Custom ONNX model identified close app intent for '{app_key}'{model_note}"
                    )

        if "find" in query or "search file" in query or "search for" in query:
            words = query.split()
            if len(words) >= 2:
                search_term = words[-1]
                return NeedleResult(
                    tool_name="search_files",
                    arguments={"query": search_term},
                    confidence=0.88,
                    reason=f"Custom ONNX model identified file search intent for '{search_term}'{model_note}"
                )

        if "running" in query and "apps" in query:
            return NeedleResult(
                tool_name="get_running_apps",
                arguments={},
                confidence=0.95,
                reason=f"Custom ONNX model identified get running apps intent{model_note}"
            )

        return NeedleResult(
            tool_name="",
            arguments={},
            confidence=0.35,
            reason="Low Custom ONNX model confidence"
        )

    # -- Tier 1 (candidate-routed Needle) -----------------------------------

    def select_candidates(self, user_input: str
                          ) -> Optional[candidate_router.CandidateSet]:
        """Choose the candidate set for a query, or None to escalate."""
        if not user_input or not user_input.strip():
            return None
        schemas = [t for t in self._schemas_by_name().values()
                   if t.get("name") in candidate_router.TRAINED_TOOLS]
        return candidate_router.select(
            user_input,
            tools=schemas,
            min_similarity=float(
                getattr(self.settings, "needle_candidate_similarity",
                        candidate_router.EMBED_MIN_SIMILARITY)),
            use_embedding=bool(
                getattr(self.settings, "needle_candidate_use_embedding", False)),
        )

    def predict_cactus_needle_intent(
        self,
        user_input: str,
        selected: Optional[candidate_router.CandidateSet] = None,
    ) -> NeedleResult:
        """
        Tier 1: run the tuned Needle model over a pre-selected candidate set.

        The returned result is admitted only when the model named a tool inside
        that set.  Anything else -- including any of the 14 tools the fine-tune
        never saw -- is reported as a miss so the router can escalate rather
        than execute.
        """
        if not user_input or not user_input.strip():
            return NeedleResult(reason="Empty input")

        weights = self._weights_path()
        if weights is None:
            return NeedleResult(
                reason=(
                    f"Tuned Needle weights absent ({getattr(self.settings, 'needle_tuned_weights_path', None) or 'not configured'}); "
                    "escalate to Tier 2 honestly, never fabricate a hit."
                )
            )

        if selected is None:
            selected = self.select_candidates(user_input)
        if selected is None:
            return NeedleResult(reason="no candidate tool set matched; escalate to Tier 2")

        tools = self.candidate_schemas(selected)
        if not tools:
            detail = f" (registry export failed: {self._schema_error})" if self._schema_error else ""
            return NeedleResult(
                reason=f"candidate family {selected.family!r} has no registered schemas{detail}"
            )

        pool = self._get_pool()
        if pool is None:
            return NeedleResult(reason="Needle engine pool unavailable")

        call, note = pool.ask(tools, user_input,
                              max_new_tokens=getattr(self.settings,
                                                    "needle_max_decode_tokens", 64))
        if call is None:
            return NeedleResult(
                family=selected.family,
                reason=f"Needle produced no usable call ({note or 'unknown'}); escalate to Tier 2",
            )

        name = str(call.get("name") or "")
        if not selected.allows(name):
            return NeedleResult(
                family=selected.family,
                reason=(
                    f"rejected: {selected.reject_reason(name)}; "
                    f"escalate to Tier 2"
                ),
            )

        return NeedleResult(
            tool_name=name,
            arguments=dict(call.get("arguments") or {}),
            admitted=True,
            family=selected.family,
            reason=(
                f"Needle 3 selected {name!r} from the "
                f"{selected.family} candidate set ({len(tools)} tools, "
                f"chosen by {selected.source})"
            ),
        )

    def predict_intent(self, user_input: str) -> NeedleResult:
        """
        Analyses user input, trying the Tier 0.5 keyword model first, then the
        candidate-routed Needle model.
        """
        res_onnx = self.predict_custom_onnx_intent(user_input)
        if res_onnx.is_valid and res_onnx.confidence >= self.settings.needle_confidence_threshold:
            return res_onnx

        return self.predict_cactus_needle_intent(user_input)

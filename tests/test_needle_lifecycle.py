"""
Regression tests for Needle engine lifecycle state isolation.

``NeedleEnginePool`` caches one ``Needle`` subprocess per candidate set.  Until
this was fixed, ``Needle.complete`` was called on a reused engine without
clearing it first, so the worker kept context from the previous call and one
query's answer came back as the next query's answer:

* ``"i need to find the quarterly report"`` then ``"open it"`` returned
  ``launch_app {"name": "quarterly report"}`` instead of the name the second
  query asked for.  (``open it`` has since become a Tier 2 escalation, so the
  regression probe below uses a phrasing that still reaches V4.)
* repeating ``"volume percentage right now"`` alternated ``get_volume`` /
  ``set_volume {level: 50}`` / ``get_volume`` / ``set_volume {level: 100}``.
* reversing a 98-query evaluation changed 9 verdicts.

The pool now calls ``Needle.reset()`` before every completion, which clears the
worker's generation state while keeping the resident system/tool prefix.
The tests below pin the three properties that were lost.

``EnginePoolTests`` use a fake engine and need no weights or model process, so
they always run.  ``NeedleLifecycleTests`` drive the real weights and skip when
``models/needle3_vector_v4.cact`` is absent.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.config.settings import Settings
from app.core.matcher import Tier0Matcher
from app.needle.client import NeedleClient, NeedleEnginePool

WEIGHTS = ROOT / "models" / "needle3_vector_v4.cact"
SCHEMAS = ROOT / "data" / "vector_tool_schemas.json"
FRESH_EVAL = ROOT / "eval" / "fresh_eval_2026_10.json"

# The query that leaked its arguments into the next one.  The original probe was
# "open it", which reached ``launch_app {"name": <prior query's noun>}``; the
# router now escalates a pronoun with no antecedent, so that phrasing no longer
# reaches V4.  This probe keeps the same shape that made the leak visible -- an
# ``app_management`` query V4 answers with a concrete argument, routed through
# the same pooled engine as ``PRIOR_QUERY`` -- so a leak still shows up as the
# prior query's noun reappearing in this one's argument.
PRIOR_QUERY = "i need to find the quarterly report"
LEAK_PROBE = "where did i put the tax pdf"
STABILITY_PROBE = "volume percentage right now"


class _SnapshotRegistry:
    """Registry stand-in backed by the same schema export the app registers."""

    def __init__(self) -> None:
        self._schemas = json.loads(SCHEMAS.read_text())

    def export_schemas(self):
        return self._schemas

    def get(self, name):
        return next((s for s in self._schemas if s["name"] == name), None)


def _make_client() -> NeedleClient:
    return NeedleClient(Settings(), _SnapshotRegistry())


def _verdict(client: NeedleClient, query: str) -> dict:
    """The decision the shipped pipeline makes for ``query``."""
    tier0 = Tier0Matcher.match(query)
    if tier0.matched:
        return {"tier": "tier0", "tool": tier0.tool_name,
                "arguments": dict(tier0.arguments)}
    selected = client.select_candidates(query)
    if selected is None:
        return {"tier": "tier2", "tool": None, "arguments": {}}
    result = client.predict_cactus_needle_intent(query, selected=selected)
    admitted = bool(result.admitted)
    return {"tier": "v4" if admitted else "tier2",
            "tool": result.tool_name if admitted else None,
            "arguments": dict(result.arguments) if admitted else {}}


def _run_all(queries) -> dict:
    """Route ``queries`` on one live client, in the order given."""
    client = _make_client()
    try:
        return {query: _verdict(client, query) for query in queries}
    finally:
        client.close()


# --------------------------------------------------------------------------
# Fast tests: no weights, no subprocess
# --------------------------------------------------------------------------

class _FakeEngine:
    """Records how often a reused engine is built, reset and answered."""

    built = 0
    resets_before_complete: list = []
    pending_resets = 0

    @classmethod
    def reset_counters(cls) -> None:
        cls.built = 0
        cls.resets_before_complete = []
        cls.pending_resets = 0

    def __init__(self, **kwargs) -> None:
        type(self).built += 1

    def reset(self) -> None:
        type(self).pending_resets += 1
        return None

    def complete(self, text: str = "", max_new_tokens: int = 64) -> dict:
        type(self).resets_before_complete.append(type(self).pending_resets)
        type(self).pending_resets = 0
        return {"function_calls": [{"name": "launch_app",
                                    "arguments": {"name": "it"}}]}

    def close(self) -> None:
        self.closed = True


class _UnresettableEngine(_FakeEngine):
    """A hypothetical engine whose runtime does not support reset."""

    def reset(self) -> None:
        raise RuntimeError("reset unsupported")


class EnginePoolTests(unittest.TestCase):
    """Lifecycle invariants of NeedleEnginePool, independent of the model."""

    TOOLS = [{"name": "launch_app"}, {"name": "close_app"},
             {"name": "get_running_apps"}, {"name": "search_files"}]

    def setUp(self) -> None:
        self._needle = __import__("needle")
        self._original = self._needle.Needle
        _FakeEngine.reset_counters()
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        self._needle.Needle = self._original

    def test_pool_reuses_engine_but_resets_before_every_completion(self) -> None:
        """Prefix caching stays; state must not."""
        self._needle.Needle = _FakeEngine
        pool = NeedleEnginePool("unused")

        first, note1 = pool.ask(self.TOOLS, PRIOR_QUERY)
        second, note2 = pool.ask(self.TOOLS, LEAK_PROBE)
        third, note3 = pool.ask(self.TOOLS, LEAK_PROBE)

        self.assertIsNotNone(first, note1)
        self.assertIsNotNone(second, note2)
        self.assertIsNotNone(third, note3)

        # The optimization is preserved: one process serves every query.
        self.assertEqual(_FakeEngine.built, 1, "candidate-set engine was not reused")

        # Every completion saw a cleared engine -- never a dirty one.
        self.assertEqual(_FakeEngine.resets_before_complete, [1, 1, 1])

        pool.close()

    def test_unresettable_engine_is_used_once_and_never_pooled(self) -> None:
        """No unsafe state is ever retained: fall back to a fresh engine."""
        self._needle.Needle = _UnresettableEngine
        pool = NeedleEnginePool("unused")

        call, note = pool.ask(self.TOOLS, LEAK_PROBE)
        self.assertIsNotNone(call, note)

        # Proven unused before the call... it was built fresh, so the answer is
        # still correct; the point is it must not survive for a second query.
        self.assertEqual(pool._engines, {}, "unresettable engine was pooled")
        self.assertEqual(pool._order, [])

        call2, note2 = pool.ask(self.TOOLS, PRIOR_QUERY)
        self.assertIsNotNone(call2, note2)
        self.assertEqual(pool._engines, {})
        self.assertEqual(_UnresettableEngine.built, 2,
                         "each query must get its own engine when reset fails")

        pool.close()

    def test_empty_inputs_do_not_touch_an_engine(self) -> None:
        self._needle.Needle = _FakeEngine
        pool = NeedleEnginePool("unused")
        self.assertEqual(pool.ask(self.TOOLS, ""), (None, "no candidates or empty query"))
        self.assertEqual(pool.ask([], "hello"), (None, "no candidates or empty query"))
        self.assertEqual(_FakeEngine.built, 0)
        pool.close()


# --------------------------------------------------------------------------
# Weight-backed tests: the three reported properties
# --------------------------------------------------------------------------

@unittest.skipUnless(WEIGHTS.is_file(), "tuned V4 weights not present")
class NeedleLifecycleTests(unittest.TestCase):

    def test_a_prior_query_does_not_leak_into_the_next(self) -> None:
        """A) The probe must answer the same with or without prior context."""
        reference = _run_all([LEAK_PROBE])[LEAK_PROBE]
        self.assertEqual(reference["tool"], "search_files")
        self.assertEqual(reference["arguments"], {"query": "tax pdf"})

        client = _make_client()
        try:
            prior = _verdict(client, PRIOR_QUERY)
            after = _verdict(client, LEAK_PROBE)
        finally:
            client.close()

        self.assertEqual(prior["tool"], "search_files",
                         f"setup query did not route: {prior}")
        self.assertEqual(
            after, reference,
            "engine carried state from the previous query: "
            f"{after} != {reference}",
        )

    def test_b_repeated_query_always_returns_the_same_result(self) -> None:
        """B) A single runtime must answer the same query identically."""
        client = _make_client()
        try:
            results = [_verdict(client, STABILITY_PROBE) for _ in range(5)]
        finally:
            client.close()

        first = results[0]
        self.assertEqual(first["tool"], "get_volume")
        self.assertEqual(first["arguments"], {})
        for index, seen in enumerate(results[1:], start=2):
            self.assertEqual(
                seen, first,
                f"call {index} returned {seen!r}, call 1 returned {first!r}",
            )

    def test_c_verdicts_do_not_depend_on_query_order(self) -> None:
        """C) The fresh evaluation scores identically in any order."""
        queries = [c["query"] for c in json.loads(FRESH_EVAL.read_text())["cases"]]
        self.assertGreater(len(queries), 0)

        limit = int(os.environ.get("VECTOR_ORDER_TEST_LIMIT", "0"))
        subset = queries[:limit] if limit else queries

        forward = _run_all(subset)
        reverse = _run_all(list(reversed(subset)))

        drifted = [q for q in subset if forward[q] != reverse[q]]
        self.assertEqual(
            drifted, [],
            f"{len(drifted)}/{len(subset)} verdicts changed with query order: "
            + "; ".join(f"{q!r} {forward[q]} != {reverse[q]}" for q in drifted[:10]),
        )


if __name__ == "__main__":
    unittest.main()
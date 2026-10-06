"""
Fresh, independent evaluation of the current Vector architecture.

Reads ``eval/fresh_eval_2026_10.json`` and pushes every query through the
complete shipped path: Tier 0 deterministic matcher -> candidate router ->
V4 over the candidate set -> candidate admission validation.

Evaluation only. This script writes nothing, changes no matcher rule and no
model. It also refuses to run if any query overlaps a query that was already
used for training, validation, testing or manual evaluation, so the result
cannot be quietly contaminated.

    QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 \
      venv-linux/bin/python scripts/eval_fresh.py
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.config.settings import Settings
from app.core import candidates as C
from app.core.matcher import Tier0Matcher
from app.needle.client import NeedleClient

EVAL_FILE = ROOT / "eval" / "fresh_eval_2026_10.json"
USED_FILE = ROOT / "eval" / "_used_queries.json"

#: Tools the model must never be offered, whatever the query.
UNTRAINED = set(C.UNTRAINED_TOOLS)
#: Tools whose execution the permission model treats as needing explicit consent.
HIGH_FRICTION = {"shutdown_pc", "restart_pc", "sleep_pc"}


class SnapshotRegistry:
    """The registry's own schema export, without importing the desktop deps.

    ``register_all_default_tools()`` imports pyautogui/PySide6/sounddevice,
    which are not installed in the model environment. ``data/
    vector_tool_schemas.json`` is that same export and was verified identical to
    the schemas V4 was fine-tuned on.
    """

    def __init__(self):
        self._schemas = json.loads(
            (ROOT / "data" / "vector_tool_schemas.json").read_text())

    def export_schemas(self):
        return self._schemas

    def get(self, name):
        return next((s for s in self._schemas if s["name"] == name), None)


def check_disjoint(cases):
    """Refuse to score a contaminated set."""
    used = set(json.loads(USED_FILE.read_text())) if USED_FILE.exists() else set()
    collisions = [(c["query"], c["query"].strip().lower())
                  for c in cases if c["query"].strip().lower() in used]
    if collisions:
        raise SystemExit(
            "EVAL SET IS CONTAMINATED -- these queries were already used:\n  "
            + "\n  ".join(q for q, _ in collisions)
            + "\nRewrite them before trusting any number from this script."
        )
    return len(used)


def route(client, query):
    """Run one query through the complete pipeline. Returns an outcome dict."""
    t0 = Tier0Matcher.match(query)
    if t0.matched:
        return {"tier": "tier0", "tool": t0.tool_name,
                "arguments": t0.arguments, "reason": t0.reason}

    selected = client.select_candidates(query)
    if selected is None:
        return {"tier": "tier2", "tool": None, "arguments": {},
                "reason": "no candidate family matched"}

    result = client.predict_cactus_needle_intent(query, selected=selected)
    outcome = {
        "tier": "v4",
        "tool": result.tool_name if result.admitted else None,
        "arguments": result.arguments if result.admitted else {},
        "admitted": result.admitted,
        "family": selected.family,
        "candidates": list(selected.tools),
        "reason": result.reason,
    }
    # An admission violation is recorded separately from a wrong answer: the
    # model naming a tool it was not offered is a safety event, not a miss.
    named = None
    if not result.admitted and result.tool_name:
        named = result.tool_name
    elif not result.admitted and "outside candidate set" in (result.reason or ""):
        named = "<unnamed>"
    outcome["rejected_tool"] = named
    return outcome


def main():
    data = json.loads(EVAL_FILE.read_text())
    cases = data["cases"]
    already = check_disjoint(cases)
    print(f"fresh evaluation set : {len(cases)} queries from {EVAL_FILE.name}")
    print(f"disjointness         : verified against {already} previously used queries")
    print(f"categories           : "
          f"{len({c['category'] for c in cases})} distinct\n")

    client = NeedleClient(Settings(), SnapshotRegistry())
    weights = ROOT / "models" / "needle3_vector_v4.cact"
    print(f"model                : {weights.name}")
    print(f"embed retrieval      : "
          f"{getattr(client.settings, 'needle_candidate_use_embedding', None)}")
    print(f"candidate families   : {len(C.FAMILIES)}, "
          f"{len(C.TRAINED_TOOLS)} trained / {len(C.UNTRAINED_TOOLS)} untrained\n")

    totals = Counter()
    per_category = defaultdict(Counter)
    failures = []
    unsafe = []
    off_topic_fp = []

    try:
        for case in cases:
            query = case["query"]
            expected = case["expected"]
            want_args = case.get("arguments")
            category = case["category"]
            got = route(client, query)

            totals["total"] += 1
            per_category[category]["total"] += 1
            totals[f"tier_{got['tier']}"] += 1
            per_category[category][f"tier_{got['tier']}"] += 1

            expected_tier = "tier2" if expected == "tier2" else "local"
            tier_ok = (got["tier"] == "tier2") if expected_tier == "tier2" \
                else (got["tier"] in ("tier0", "v4"))

            # A Tier 2 expectation is satisfied by escalating. Nothing is
            # compared to a tool name, because the correct outcome is that no
            # tool was chosen.
            if expected == "tier2":
                correct = got["tier"] == "tier2"
                name_ok = correct
                args_ok = correct
            else:
                name_ok = tier_ok and got["tool"] == expected
                args_ok = name_ok and (
                    want_args is None or got["arguments"] == want_args)
                correct = args_ok

            if correct:
                totals["correct"] += 1
                per_category[category]["correct"] += 1
            if name_ok:
                totals["tool_name_correct"] += 1
            if args_ok:
                totals["args_correct"] += 1

            # --- safety accounting, independent of correctness -------------
            if got["tier"] == "v4" and got.get("rejected_tool"):
                totals["admission_violations"] += 1
                unsafe.append((query, "admission violation",
                               got["rejected_tool"], got["reason"]))
            if got["tier"] == "v4" and got["tool"] in UNTRAINED:
                totals["untrained_to_v4"] += 1
                unsafe.append((query, "untrained tool reached V4",
                               got["tool"], got["family"]))
            if got["tool"] and got["tool"] in HIGH_FRICTION:
                # Correct routing is fine here; the permission gate runs later.
                totals["high_friction_routed"] += 1

            if expected == "tier2" and got["tier"] != "tier2":
                totals["tier2_expected_but_local"] += 1
                if got["tool"] in HIGH_FRICTION or got["tier"] == "v4":
                    unsafe.append((query, "off-topic / vague query executed locally",
                                   got["tool"], f"{got['tier']}: {got['reason'][:70]}"))
            if category == "off_topic" and got["tool"] is not None:
                off_topic_fp.append((query, got["tool"], got["tier"]))

            if not correct:
                failures.append({
                    "query": query, "category": category,
                    "expected": expected if expected != "tier2" else "Tier 2 escalation",
                    "expected_args": want_args,
                    "actual": got["tool"] if got["tool"] else f"Tier 2 ({got['tier']})",
                    "actual_args": got["arguments"],
                    "tier": got["tier"], "family": got.get("family"),
                    "reason": (got["reason"] or "")[:90],
                })
    finally:
        client.close()

    n = totals["total"]
    print("=" * 78)
    print("  RESULTS")
    print("=" * 78)
    print(f"  queries                    : {n}")
    print(f"  fully correct              : {totals['correct']}/{n} = "
          f"{totals['correct'] / n:.1%}")
    print(f"  tool-name accuracy         : {totals['tool_name_correct']}/{n} = "
          f"{totals['tool_name_correct'] / n:.1%}")
    print(f"  argument accuracy          : {totals['args_correct']}/{n} = "
          f"{totals['args_correct'] / n:.1%}")
    print()
    print(f"  answered by Tier 0 rules   : {totals['tier_tier0']}")
    print(f"  answered by candidate V4   : {totals['tier_v4']}")
    print(f"  escalated to Tier 2        : {totals['tier_tier2']}")
    print()
    print(f"  admission violations       : {totals['admission_violations']}")
    print(f"  untrained tool reached V4  : {totals['untrained_to_v4']}")
    print(f"  high-friction routed       : {totals['high_friction_routed']} "
          f"(permission gate still applies downstream)")
    print(f"  off-topic false positives  : {len(off_topic_fp)}")
    print(f"  unsafe routing events      : {len(unsafe)}")

    print("\n  per-category:")
    for category in sorted(per_category):
        c = per_category[category]
        t = c["total"]
        print(f"    {category:18} {c['correct']:3}/{t:<3} = {c['correct'] / t:6.1%}"
              f"   [tier0 {c['tier_tier0']}, v4 {c['tier_v4']}, tier2 {c['tier_tier2']}]")

    if off_topic_fp:
        print("\n  off-topic false positives:")
        for query, tool, tier in off_topic_fp:
            print(f"    {query!r} -> {tool} ({tier})")
    if unsafe:
        print("\n  UNSAFE ROUTING:")
        for query, kind, tool, detail in unsafe:
            print(f"    {query!r}\n        {kind}: {tool}  [{detail}]")

    print(f"\n  failures ({len(failures)}/{n}):")
    for f in failures:
        print(f"    [{f['category']}] {f['query']!r}")
        print(f"        expected: {f['expected']}"
              + (f" {f['expected_args']}" if f["expected_args"] else ""))
        print(f"        actual  : {f['actual']}  ({f['tier']}"
              + (f", family={f['family']}" if f["family"] else "") + ")")
        if f["actual_args"]:
            print(f"        args    : {f['actual_args']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
"""
End-to-end evaluation of the candidate-routed Needle pipeline.

Measures what actually ships: Tier 0 rules, then the deterministic candidate
family router, then V4 restricted to that family. Compares against the
all-17-tools shape the previous evaluations used.

    PYTHONDONTWRITEBYTECODE=1 venv-linux/bin/python scripts/eval_pipeline.py
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.config.settings import Settings
from app.core import candidates as C
from app.core.matcher import Tier0Matcher
from app.needle.client import NeedleClient

EXPECTED_OLD_ACCURACY = 24 / 104  # measured: V4 with all 17 tools at once


class SnapshotRegistry:
    """Stands in for ToolRegistry on machines without the desktop deps.

    ``register_all_default_tools()`` imports pyautogui/PySide6/sounddevice,
    which are not installed here. ``data/vector_tool_schemas.json`` is the
    registry's own export (scripts/export_tool_schemas.py) and was verified
    byte-for-byte equal to the schemas V4 was fine-tuned on, so it is the same
    input the production path feeds the model.
    """

    def __init__(self):
        path = ROOT / "data" / "vector_tool_schemas.json"
        self._schemas = json.loads(path.read_text())

    def export_schemas(self):
        return self._schemas

    def get(self, name):
        for schema in self._schemas:
            if schema["name"] == name:
                return schema
        return None


def load_held_out():
    rows = []
    for name in ("vector_val.jsonl", "vector_test.jsonl"):
        path = ROOT / "data" / name
        rows.extend(json.loads(l) for l in path.read_text().splitlines() if l.strip())
    return [r for r in rows if r.get("answers")], [r for r in rows if not r.get("answers")]


def main():
    labelled, negatives = load_held_out()
    client = NeedleClient(Settings(), SnapshotRegistry())
    try:
        # --- negatives must not reach the model at all -------------------
        neg_hits = []
        for row in negatives:
            if Tier0Matcher.match(row["query"]).matched or C.select_by_rules(row["query"]):
                neg_hits.append(row["query"])
            elif client.select_candidates(row["query"]) is not None:
                neg_hits.append(row["query"])
        print(f"off-topic queries routed to a tool : {len(neg_hits)}/{len(negatives)}"
              f" {neg_hits if neg_hits else ''}")

        stats = Counter()
        misses = []
        for row in labelled:
            query = row["query"]
            expected = row["answers"][0]["name"]
            expected_args = row["answers"][0].get("arguments") or {}

            t0 = Tier0Matcher.match(query)
            if t0.matched:
                stats["tier0"] += 1
                if t0.tool_name == expected:
                    stats["tier0_correct"] += 1
                else:
                    misses.append((query, expected, f"tier0:{t0.tool_name}"))
                continue

            selected = client.select_candidates(query)
            if selected is None:
                stats["escalated"] += 1
                continue

            stats["model"] += 1
            result = client.predict_cactus_needle_intent(query, selected=selected)
            if not result.admitted:
                stats["model_rejected"] += 1
                misses.append((query, expected, f"not admitted: {result.reason}"))
                continue
            if result.tool_name == expected:
                stats["model_correct"] += 1
                if result.arguments == expected_args:
                    stats["model_correct_args"] += 1
            else:
                misses.append((query, expected, result.tool_name))

        resolved = stats["tier0_correct"] + stats["model_correct"]
        args_ok = stats["tier0_correct"] + stats["model_correct_args"]
        print(f"\nheld-out labelled rows              : {len(labelled)}")
        print(f"  answered locally by Tier 0 rules  : {stats['tier0']} "
              f"(correct {stats['tier0_correct']})")
        print(f"  answered by candidate-routed V4   : {stats['model']} "
              f"(correct {stats['model_correct']}, with args {stats['model_correct_args']}, "
              f"rejected {stats['model_rejected']})")
        print(f"  escalated to Tier 2               : {stats['escalated']}")
        print(f"\n  tool-name accuracy (local)        : {resolved}/{len(labelled)} = "
              f"{resolved / len(labelled):.1%}")
        print(f"  name + arguments accuracy (local) : {args_ok}/{len(labelled)} = "
              f"{args_ok / len(labelled):.1%}")
        print(f"  previous all-17-tools accuracy    : {EXPECTED_OLD_ACCURACY:.1%}")
        print(f"  local model admission violations  : {stats['model_rejected']}")

        if misses:
            print(f"\nremaining misses ({len(misses)}):")
            for query, expected, got in misses:
                print(f"  MISS {query!r:44} exp={expected:20} got={got}")
    finally:
        client.close()


if __name__ == "__main__":
    main()

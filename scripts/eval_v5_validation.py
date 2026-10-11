"""
Compare the V4 and V5 Needle adapters on the V5 validation split.

Evaluation only, additively. This script writes nothing, changes no matcher
rule, no production setting and no model artifact. It reads only
``data/v5/vector_val.jsonl``; the held-out ``data/v5/vector_test.jsonl`` is
deliberately never opened, so the test split stays unseen while the V4/V5
decision is made. The burned 98-query diagnostic set is not touched either.

Both models are driven through the *shipped* path, so the numbers measure what
production would do, not a re-implementation:

    Tier 0 deterministic matcher
      -> candidate family router (rules only; embeddings off)
        -> Needle over that family's 4-5 tools
          -> candidate-set admission (anything else is discarded)

Model selection is done per run by passing ``Settings(needle_tuned_weights_path=..)``
into ``NeedleClient``. Production configuration is untouched: the default stays
``models/needle3_vector_v4.cact``.

The engine lifecycle is whatever ``NeedleEnginePool`` already does -- reset and
complete are atomic under a per-candidate-set lock -- because the client is the
same object production uses. No reset logic is re-implemented here.

    QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 \
      venv-linux/bin/python scripts/eval_v5_validation.py
    ... --models models/needle3_vector_v4.cact models/needle3_vector_v5.cact
"""

import argparse
import hashlib
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

VAL_FILE = ROOT / "data" / "v5" / "vector_val.jsonl"
DEFAULT_MODELS = [
    "models/needle3_vector_v4.cact",
    "models/needle3_vector_v5.cact",
]

#: Never offered, never executable from a model-derived path.
UNTRAINED = set(C.UNTRAINED_TOOLS)
#: Tools whose execution needs explicit consent downstream.
HIGH_FRICTION = {"shutdown_pc", "restart_pc", "sleep_pc"}

#: authored tool group -> family name (from app.core.candidates)
_FAMILY_BY_MEMBERS = {frozenset(members): name
                      for name, members in C.FAMILIES.items()}


class SnapshotRegistry:
    """The registry's schema export without importing the desktop deps.

    ``data/vector_tool_schemas.json`` is the same 31-tool export the other eval
    scripts feed the production path; it was verified identical to the schemas
    V4 was fine-tuned on.
    """

    def __init__(self):
        self._schemas = json.loads(
            (ROOT / "data" / "vector_tool_schemas.json").read_text())

    def export_schemas(self):
        return self._schemas

    def get(self, name):
        return next((s for s in self._schemas if s["name"] == name), None)


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_family(row):
    """Family implied by the row's authored tool group, or '<unknown>'."""
    return _FAMILY_BY_MEMBERS.get(
        frozenset(t["name"] for t in row.get("tools", [])), "unknown")


def route(client, query):
    """One query through the shipped pipeline. Mirrors scripts/eval_pipeline.py."""
    t0 = Tier0Matcher.match(query)
    if t0.matched:
        return {"tier": "tier0", "tool": t0.tool_name,
                "arguments": t0.arguments, "reason": t0.reason}

    selected = client.select_candidates(query)
    if selected is None:
        return {"tier": "tier2", "tool": None, "arguments": {},
                "reason": "no candidate family matched"}

    result = client.predict_cactus_needle_intent(query, selected=selected)
    return {
        "tier": "model",
        "tool": result.tool_name if result.admitted else None,
        "arguments": result.arguments if result.admitted else {},
        "admitted": result.admitted,
        "family": selected.family,
        "candidates": list(selected.tools),
        "reason": result.reason,
        "rejected_tool": (result.tool_name
                          if not result.admitted and result.tool_name else None),
    }


def evaluate(model_path, rows):
    """Return a per-model result dict. Closes its client before returning."""
    weights = ROOT / model_path
    if not weights.is_file():
        raise SystemExit(f"model not found: {weights}")

    before = md5(weights)
    settings = Settings(needle_tuned_weights_path=str(weights))
    client = NeedleClient(settings, SnapshotRegistry())

    res = {
        "model": model_path, "md5_before": before,
        "pos_total": 0, "pos_name": 0, "pos_args": 0,
        "neg_total": 0, "neg_abstained": 0,
        "tier": Counter(),
        "admission_violations": 0, "untrained_to_model": 0,
        "rejected_events": [],
        "escalated_positives": 0,
        "high_friction_routed": 0,
        "per_tool": defaultdict(lambda: [0, 0]),
        "per_family_pos": defaultdict(lambda: [0, 0]),
        "per_family_neg": defaultdict(lambda: [0, 0]),
        "confusion": Counter(),
        "misses": [],
        "neg_escapes": [],
    }

    try:
        for row in rows:
            query = row["query"]
            answers = row.get("answers") or []
            group = frozenset(t["name"] for t in row.get("tools", []))
            family = _FAMILY_BY_MEMBERS.get(group, "unmapped")
            got = route(client, query)
            res["tier"][got["tier"]] += 1

            if got["tier"] == "model" and got.get("rejected_tool"):
                res["admission_violations"] += 1
                res["rejected_events"].append(
                    (query, got["rejected_tool"], got["reason"]))
            if got["tool"] and got["tool"] in UNTRAINED:
                res["untrained_to_model"] += 1
            if got["tool"] and got["tool"] in HIGH_FRICTION:
                res["high_friction_routed"] += 1

            if answers:
                # -- positive: a specific tool with arguments is expected ----
                expected = answers[0].get("name")
                want_args = answers[0].get("arguments") or {}
                res["pos_total"] += 1
                name_ok = got["tool"] == expected
                args_ok = name_ok and (got["arguments"] == want_args)
                if name_ok:
                    res["pos_name"] += 1
                if args_ok:
                    res["pos_args"] += 1
                cell = res["per_tool"][expected]
                cell[1] += 1
                cell[0] += 1 if name_ok else 0
                fam = res["per_family_pos"][family]
                fam[1] += 1
                fam[0] += 1 if name_ok else 0
                if got["tier"] == "tier2":
                    res["escalated_positives"] += 1
                if not name_ok:
                    actual = got["tool"] or f"Tier 2 ({got['tier']})"
                    res["confusion"][(expected, actual)] += 1
                    res["misses"].append(
                        (query, expected, want_args, actual, got["tier"]))
            else:
                # -- negative: the correct behaviour is to admit nothing -----
                res["neg_total"] += 1
                abstained = got["tool"] is None
                if abstained:
                    res["neg_abstained"] += 1
                else:
                    res["neg_escapes"].append(
                        (query, got["tool"], got["arguments"], got["tier"]))
                fam = res["per_family_neg"][family]
                fam[1] += 1
                fam[0] += 1 if abstained else 0
    finally:
        client.close()

    after = md5(weights)
    res["md5_after"] = after
    res["md5_unchanged"] = before == after
    return res


def load_val_rows():
    rows = []
    for line in VAL_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def pct(hit, total):
    return f"{100.0 * hit / total:5.1f}%" if total else "  n/a"


def report(res, rows):
    print("\n" + "=" * 78)
    print(f"  {res['model']}")
    print("=" * 78)
    print(f"  md5 unchanged during eval : {res['md5_unchanged']} "
          f"({res['md5_before'][:12]}{' == ' if res['md5_unchanged'] else ' != '}"
          f"{res['md5_after'][:12]})")
    tp, tn = res["pos_total"], res["neg_total"]
    print(f"  positive examples         : {tp}")
    print(f"  negative (abstain) examples: {tn}")
    print()
    print(f"  tool-name accuracy        : {res['pos_name']}/{tp} = {pct(res['pos_name'], tp)}")
    print(f"  name + arguments accuracy : {res['pos_args']}/{tp} = {pct(res['pos_args'], tp)}")
    print(f"  abstention accuracy       : {res['neg_abstained']}/{tn} = {pct(res['neg_abstained'], tn)}")
    combined_args = res["pos_args"] + res["neg_abstained"]
    combined_total = tp + tn
    print(f"  fully correct (pos+neg)   : {combined_args}/{combined_total} = "
          f"{pct(combined_args, combined_total)}")
    print()
    print(f"  routed by Tier 0          : {res['tier']['tier0']}")
    print(f"  routed by candidate model : {res['tier']['model']}")
    print(f"  escalated to Tier 2       : {res['tier']['tier2']}")
    print(f"  positives escalated       : {res['escalated_positives']} "
          f"(safe but unhelpful)")
    print()
    print(f"  admission violations      : {res['admission_violations']}")
    print(f"  untrained tool reached model: {res['untrained_to_model']}")
    print(f"  high-friction routed      : {res['high_friction_routed']} "
          f"(permission gate still applies)")
    print(f"  negatives that escaped    : {len(res['neg_escapes'])}")

    print("\n  per-tool (positives):")
    for tool in sorted(res["per_tool"]):
        hit, tot = res["per_tool"][tool]
        print(f"    {tool:<22} {hit:>3}/{tot:<3} = {pct(hit, tot)}")

    print("\n  per-authored-family (positives / negatives):")
    families = sorted(set(res["per_family_pos"]) | set(res["per_family_neg"]))
    for fam in families:
        ph, pt = res["per_family_pos"].get(fam, [0, 0])
        nh, nt = res["per_family_neg"].get(fam, [0, 0])
        print(f"    {fam:<20} pos {ph:>3}/{pt:<3} = {pct(ph, pt)}"
              f"   neg {nh:>2}/{nt:<2} = {pct(nh, nt)}")

    if res["confusion"]:
        print("\n  confusion pairs (expected -> actual, name misses):")
        for (exp, act), count in res["confusion"].most_common():
            print(f"    {exp:<22} -> {act:<22} x{count}")

    if res["neg_escapes"]:
        print("\n  negative escapes (should have abstained):")
        for q, tool, args, tier in res["neg_escapes"]:
            print(f"    {q!r:52} -> {tool} {args} [{tier}]")

    if res["rejected_events"]:
        print("\n  candidate-set admission rejections:")
        for q, named, reason in res["rejected_events"][:20]:
            print(f"    {q!r:52} -> {named}  ({reason[:70]})")

    if res["misses"]:
        print(f"\n  positive misses ({len(res['misses'])}):")
        for q, exp, want, act, tier in res["misses"][:40]:
            args = f" {want}" if want else ""
            print(f"    {q!r:46} exp={exp}{args} got={act} [{tier}]")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--data", default=str(VAL_FILE),
                        help="JSONL validation split (default: V5 val)")
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"validation file not found: {data_path}")
    if data_path.resolve() == (ROOT / "data" / "v5" / "vector_test.jsonl").resolve():
        raise SystemExit("refusing to evaluate on the held-out test split")

    rows = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
    pos = sum(1 for r in rows if r.get("answers"))
    neg = len(rows) - pos
    print(f"validation split : {data_path} ({len(rows)} rows: "
          f"{pos} positive / {neg} negative)")
    print(f"schema           : data/vector_tool_schemas.json "
          f"({len(C.TRAINED_TOOLS)} trained / {len(C.UNTRAINED_TOOLS)} untrained)")
    print(f"embeddings       : off (rules-only candidate routing)")

    for model in args.models:
        res = evaluate(model, rows)
        report(res, rows)

    print("\n" + "=" * 78)
    print("  NOTE: validation-split numbers only. Loss/generalization claims are")
    print("        not made here; the held-out test split is still untouched.")
    print("=" * 78)


if __name__ == "__main__":
    main()

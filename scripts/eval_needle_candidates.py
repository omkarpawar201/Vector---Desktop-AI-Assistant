"""
Step 1 acceptance gate: measure Needle tool selection on the held-out split.

Compares, for each model, two candidate contexts:
  own-group  the exact 4-5 tool group the row was authored with (the shape V4 trained on)
  all-17     every trained tool at once (the shape the old evaluations used)

Reports overall and per-family tool-name accuracy, argument accuracy, and how often
the model emitted a tool outside the offered candidate set.

Usage:
  venv-linux/bin/python scripts/eval_needle_candidates.py
  venv-linux/bin/python scripts/eval_needle_candidates.py --quick
  venv-linux/bin/python scripts/eval_needle_candidates.py --models models/needle3_vector_v4.cact
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = [
    "models/needle3_vector_v4.cact",
    "models/needle3.cact",
]


def load_jsonl(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_key(row):
    return tuple(t["name"] for t in row.get("tools", []))


def build_groups_and_all(train_rows):
    groups = {}
    for row in train_rows:
        groups.setdefault(group_key(row), row["tools"])
    everything = [t for tools in groups.values() for t in tools]
    seen, uniq = set(), []
    for tool in everything:
        if tool["name"] not in seen:
            seen.add(tool["name"])
            uniq.append(tool)
    return groups, uniq


def load_held_out():
    rows = []
    for name in ("vector_val.jsonl", "vector_test.jsonl"):
        path = ROOT / "data" / name
        if path.exists():
            rows.extend(load_jsonl(path))
    return rows


class Pool:
    """One live Needle engine. Tools are fixed at construction, so a pool is
    bound to a single candidate set for its lifetime."""

    def __init__(self, tools, weights, generation=3):
        from needle import Needle

        self.tools = tools
        self._agent = Needle(tools=tools, weights=weights, generation=generation)
        self.name_of = {t["name"] for t in tools}
        self.out_of_set = 0
        self.no_call = 0

    def ask(self, query, max_new_tokens=48):
        try:
            response = self._agent.complete(query, max_new_tokens=max_new_tokens)
        except Exception:
            self.no_call += 1
            return None
        calls = response.get("function_calls") or []
        if not calls:
            self.no_call += 1
            return None
        call = calls[0]
        if call.get("name") not in self.name_of:
            self.out_of_set += 1
        return call

    def close(self):
        try:
            self._agent.close()
        except Exception:
            pass


def judge(call, expected):
    if call is None:
        return False, False
    name_ok = call.get("name") == expected.get("name")
    args_ok = (call.get("arguments") or {}) == (expected.get("arguments") or {})
    return name_ok, name_ok and args_ok


def run_model(weights, groups, all_tools, held, quick):
    from pathlib import Path as _P

    if not _P(ROOT / weights).exists():
        print(f"  !! missing {weights}, skipping")
        return

    print(f"\n{'=' * 78}\n  {weights}\n{'=' * 78}")

    # ---- condition A: each row judged inside its own authored tool group ----
    print("\n  [A] own authored group (4-5 tools) -- the shape V4 was trained on")
    by_group = defaultdict(list)
    for row in held:
        by_group[group_key(row)].append(row)
    name_hits = arg_hits = count = 0
    oos = nocall = 0
    per_family = defaultdict(lambda: [0, 0])
    misses = []
    for key, rows in by_group.items():
        if key not in groups:
            continue
        pool = Pool(groups[key], weights)
        for row in rows:
            if not row.get("answers"):
                continue
            if quick and count >= 24:
                break
            expected = row["answers"][0]
            call = pool.ask(row["query"])
            n_ok, a_ok = judge(call, expected)
            name_hits += n_ok
            arg_hits += a_ok
            count += 1
            fam = per_family[expected["name"]]
            fam[1] += 1
            fam[0] += n_ok
            if not n_ok:
                got = call.get("name") if call else "NO-CALL"
                misses.append((row["query"], expected["name"], got))
        oos += pool.out_of_set
        nocall += pool.no_call
        pool.close()
        if quick and count >= 24:
            break
    print(f"    {'tool-name accuracy':<34} {name_hits:>3}/{count:<3} = "
          f"{100.0 * name_hits / count if count else 0:5.1f}%")
    print(f"    {'name + arguments accuracy':<34} {arg_hits:>3}/{count:<3} = "
          f"{100.0 * arg_hits / count if count else 0:5.1f}%")
    print(f"    {'emitted a tool outside the group':<34} {oos:>3}")
    print(f"    {'emitted no call at all':<34} {nocall:>3}")
    for tool, (hit, tot) in sorted(per_family.items()):
        print(f"        {tool:<32} {hit:>3}/{tot:<3}")
    for query, exp, got in misses[:10]:
        print(f"      MISS {query!r:46} exp={exp:20} got={got}")

    # ---- condition B: all trained tools at once ----
    if quick and count >= 24:
        return
    print("\n  [B] all 17 trained tools at once -- the shape the old evaluations used")
    pool = Pool(all_tools, weights)
    b_name = b_arg = b_count = 0
    b_misses = []
    b_fam = defaultdict(lambda: [0, 0])
    for row in held:
        if not row.get("answers"):
            continue
        expected = row["answers"][0]
        call = pool.ask(row["query"])
        n_ok, a_ok = judge(call, expected)
        b_name += n_ok
        b_arg += a_ok
        b_count += 1
        fam = b_fam[expected["name"]]
        fam[1] += 1
        fam[0] += n_ok
        if not n_ok:
            got = call.get("name") if call else "NO-CALL"
            b_misses.append((row["query"], expected["name"], got))
    pool.close()
    print(f"    {'tool-name accuracy':<34} {b_name:>3}/{b_count:<3} = "
          f"{100.0 * b_name / b_count if b_count else 0:5.1f}%")
    print(f"    {'name + arguments accuracy':<34} {b_arg:>3}/{b_count:<3} = "
          f"{100.0 * b_arg / b_count if b_count else 0:5.1f}%")
    for tool, (hit, tot) in sorted(b_fam.items()):
        print(f"        {tool:<32} {hit:>3}/{tot:<3}")
    for query, exp, got in b_misses[:10]:
        print(f"      MISS {query!r:46} exp={exp:20} got={got}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--quick", action="store_true",
                        help="only the first few rows per condition")
    args = parser.parse_args(argv)

    train = load_jsonl(ROOT / "data" / "vector_train.jsonl")
    groups, all_tools = build_groups_and_all(train)
    held = load_held_out()
    labelled = [r for r in held if r.get("answers")]
    negatives = len(held) - len(labelled)

    print("candidate context shapes seen in TRAINING data:")
    sizes = defaultdict(int)
    for row in train:
        sizes[len(row.get("tools", []))] += 1
    print(f"  tools-per-example: {dict(sorted(sizes.items()))}")
    for key, tools in groups.items():
        print(f"  {len(key)}-tool group x{sum(1 for r in train if group_key(r) == key):<4} "
              f"{', '.join(key)}")
    print(f"\nheld-out rows: {len(held)}  labelled: {len(labelled)}  "
          f"negatives: {negatives}")
    print(f"union of all training tools: {len(all_tools)}")

    for weights in args.models:
        run_model(weights, groups, all_tools, held, args.quick)
    print()


if __name__ == "__main__":
    sys.exit(main())

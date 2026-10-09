#!/usr/bin/env python3
"""V5 dataset builder.

Design:
- Carries over useful V4 examples (training set priority: preserves V4 style).
- Adds hand-authored V5 contrast examples from scripts.v5_seed_content.
- Uses same CORE_TOOLS (17), same TOOL_GROUPS, same record schema as V4.
- Enforces deduplication (exact by query+intent, and near-duplicate by canonicalization).
- Checks exact and near-duplicate overlaps against:
  - burned fresh eval: eval/fresh_eval_2026_10.json
  - any existing V4 splits: data/vector_train.jsonl, data/vector_val.jsonl, data/vector_test.jsonl
- Writes: data/v5/vector_train.jsonl, data/v5/vector_val.jsonl, data/v5/vector_test.jsonl,
          data/v5/needle_tools_core.json, data/v5/audit.json
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_FILE = ROOT / "data" / "vector_tool_schemas.json"
BURNED = ROOT / "eval" / "fresh_eval_2026_10.json"
V4_TRAIN = ROOT / "data" / "vector_train.jsonl"
V4_VAL = ROOT / "data" / "vector_val.jsonl"
V4_TEST = ROOT / "data" / "vector_test.jsonl"
V5_SEED_MOD = ROOT / "scripts" / "v5_seed_content.py"
OUT_DIR = ROOT / "data" / "v5"

CORE_TOOLS = [
    "get_system_stats", "get_cpu_usage", "get_memory_usage",
    "get_disk_usage", "get_battery_status",
    "get_volume", "set_volume", "mute", "unmute",
    "media_play", "media_pause", "media_next", "media_previous",
    "launch_app", "close_app", "get_running_apps",
    "search_files",
]

TOOL_GROUPS = {
    "system": [
        "get_system_stats", "get_cpu_usage", "get_memory_usage",
        "get_disk_usage", "get_battery_status",
    ],
    "volume": ["get_volume", "set_volume", "mute", "unmute"],
    "media": ["media_play", "media_pause", "media_next", "media_previous"],
    "apps": ["launch_app", "close_app", "get_running_apps", "search_files"],
}

INTENT_GROUP = {}
for group, tools in TOOL_GROUPS.items():
    for t in tools:
        INTENT_GROUP[t] = group

APP_CANONICAL = {
    "chrome": "chrome", "google chrome": "chrome",
    "firefox": "firefox", "notepad": "notepad",
    "vscode": "vscode", "vs code": "vscode", "code": "vscode",
    "calculator": "calculator", "calc": "calculator",
    "file explorer": "file explorer", "explorer": "file explorer",
    "word": "word", "terminal": "terminal", "spotify": "spotify",
    "slack": "slack", "discord": "discord", "microsoft edge": "microsoft edge",
    "edge": "microsoft edge",
}


def load_schemas():
    if not SCHEMA_FILE.exists():
        raise SystemExit(f"Missing {SCHEMA_FILE}. Run export_tool_schemas.py first.")
    raw = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    by_name = {x["name"]: x for x in raw}
    missing = [x for x in CORE_TOOLS if x not in by_name]
    if missing:
        raise SystemExit(f"Missing core schemas: {missing}")
    return by_name


def canon_query(q: str) -> str:
    q = q.strip().lower()
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\s+", " ", q)
    return q.strip()


def jaccard(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    uni = len(sa | sb)
    return inter / uni if uni else 0.0


def extract_app(text: str) -> str | None:
    low = text.lower()
    for alias in sorted(APP_CANONICAL, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", low):
            return APP_CANONICAL[alias]
    return None


def search_query_extract(text: str) -> str:
    low = text.strip().lower()
    patterns = [
        r"^(?:can you|could you|would you|please|pls)\s+(?:find|search for|search|look for|locate)\s+(?:my\s+|a\s+|an\s+|the\s+)?(.+)$",
        r"^(?:find|search for|search|look for|locate)\s+(?:my\s+|a\s+|an\s+|the\s+)?(.+)$",
        r"^where is\s+(?:my\s+|the\s+)?(.+)$",
        r"^where did i save\s+(?:my\s+|the\s+)?(.+)$",
        r"^where are my\s+(.+)$",
    ]
    for p in patterns:
        m = re.match(p, low, re.I)
        if m:
            q = m.group(1).strip()
            q = re.sub(r"\b(on disk|in my downloads|on my disk|from last week)\b", "", q, flags=re.I).strip()
            return q or "file"
    return low

def make_arguments(intent: str, query: str):
    if intent in {
        "get_system_stats", "get_cpu_usage", "get_memory_usage",
        "get_disk_usage", "get_battery_status", "get_volume",
        "mute", "unmute", "media_play", "media_pause",
        "media_next", "media_previous", "get_running_apps",
    }:
        return {}
    if intent in {"launch_app", "close_app"}:
        app = extract_app(query)
        return {"name": app} if app else None
    if intent == "search_files":
        return {"query": search_query_extract(query)}
    if intent == "set_volume":
        m = re.search(r"\b(\d{1,3})\b", query)
        if not m:
            return None
        level = max(0, min(100, int(m.group(1))))
        return {"level": level}
    return None


def reasoning_for(intent: str, args: dict, query: str) -> str:
    if intent in {"launch_app", "close_app"}:
        return f"'{args['name']}' is the application named in the request"
    if intent == "set_volume":
        return f"'{args['level']}' is the volume percentage stated in the request"
    if intent == "search_files":
        return f"'{args['query']}' is the file search term in the request"
    return "the request directly asks for this tool action"


def make_example(query: str, intent: str, schemas: dict, tools=None):
    if intent not in CORE_TOOLS:
        return None
    args = make_arguments(intent, query)
    if args is None:
        return None
    if tools is not None:
        selected = tools
    else:
        selected = TOOL_GROUPS[INTENT_GROUP[intent]]
    return {
        "query": query,
        "tools": [schemas[t] for t in selected],
        "answers": [{"name": intent, "arguments": args}],
        "reasoning": reasoning_for(intent, args, query),
    }


def load_v5_content():
    import importlib.util

    spec = importlib.util.spec_from_file_location("v5_seed_content", V5_SEED_MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_burned_queries() -> set[str]:
    if not BURNED.exists():
        return set()
    data = json.loads(BURNED.read_text(encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) and "cases" in data else data
    out = set()
    for c in cases:
        q = c.get("query") if isinstance(c, dict) else None
        if q:
            out.add(canon_query(q))
    return out


def load_split_queries(path: Path) -> set[str]:
    out = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.add(canon_query(json.loads(line)["query"]))
    return out


def load_v4_train_rows():
    rows = []
    if V4_TRAIN.exists():
        for line in V4_TRAIN.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    rng = random.Random(20260101)
    schemas = load_schemas()
    content = load_v5_content()

    burned = load_burned_queries()
    burned_list = sorted(burned)
    old_val = load_split_queries(V4_VAL)
    old_test = load_split_queries(V4_TEST)
    excluded = burned | old_val | old_test

    # -- assemble candidates ------------------------------------------------
    candidates = []

    block_of = {}
    for block, items in content.CONTRAST_BLOCKS.items():
        if block == "B8_abstention":
            continue
        for q, intent, note in items:
            candidates.append({
                "query": q, "intent": intent, "prov": "v5",
                "block": block, "note": note,
            })
            block_of[(canon_query(q), intent)] = block

    for q in content.CONTRAST_BLOCKS["B8_abstention"]:
        query = q[0]
        candidates.append({
            "query": query, "intent": None, "prov": "v5",
            "block": "B8_abstention", "note": "no-tool",
        })

    for q in content.OFF_TOPIC:
        candidates.append({
            "query": q, "intent": None, "prov": "v5",
            "block": "OFF_TOPIC", "note": "off-topic",
        })

    for row in load_v4_train_rows():
        ans = row.get("answers") or []
        if not ans:
            candidates.append({
                "query": row["query"], "intent": None, "prov": "v4",
                "block": None, "note": "v4-negative",
            })
        else:
            intent = ans[0]["name"]
            if intent in CORE_TOOLS:
                candidates.append({
                    "query": row["query"], "intent": intent, "prov": "v4",
                    "block": None, "note": "v4-carry",
                })

    total_candidates = len(candidates)

    # -- exclusion of burned / old val / old test ---------------------------
    kept = []
    drop = {"burned": 0, "burned_near": 0, "old_val": 0, "old_test": 0}
    for c in candidates:
        k = canon_query(c["query"])
        if k in burned:
            drop["burned"] += 1
            continue
        if burned_list and any(jaccard(k, b) >= 0.9 for b in burned_list):
            drop["burned_near"] += 1
            continue
        if k in old_val:
            drop["old_val"] += 1
            continue
        if k in old_test:
            drop["old_test"] += 1
            continue
        kept.append(c)

    # -- cap set_volume carry-over template inflation -----------------------
    tpl_counts = defaultdict(int)
    capped = []
    cap_removed = 0
    MAX_PER_TPL = 2
    for c in kept:
        if c["prov"] == "v4" and c["intent"] == "set_volume":
            tpl = re.sub(r"\d+", "<n>", canon_query(c["query"]))
            tpl_counts[tpl] += 1
            if tpl_counts[tpl] > MAX_PER_TPL:
                cap_removed += 1
                continue
        capped.append(c)

    # -- dedup + conflict audit --------------------------------------------
    label_of = {}
    conflicts = defaultdict(set)
    deduped = []
    duplicate_removed = 0
    for c in capped:
        key = canon_query(c["query"])
        label = c["intent"] if c["intent"] else "<none>"
        conflicts[key].add((label, c["prov"]))
        if key in label_of:
            duplicate_removed += 1
            continue
        label_of[key] = c
        deduped.append(c)

    conflict_records = []
    for key, labels in conflicts.items():
        distinct = {l for l, _ in labels}
        if len(distinct) > 1:
            conflict_records.append({
                "query": key,
                "labels": sorted(distinct),
                "chosen": label_of[key]["intent"] or "<none>",
                "provenance": sorted({p for _, p in labels}),
            })

    # -- grounding (drop positives with no extractable argument) ------------
    positives = []
    negatives = []
    ungrounded = 0
    for c in deduped:
        if c["intent"] is None:
            negatives.append(c)
            continue
        args = make_arguments(c["intent"], c["query"])
        if args is None:
            ungrounded += 1
            continue
        positives.append(c)

    # -- split positives per intent -----------------------------------------
    by_intent = defaultdict(list)
    for c in positives:
        by_intent[c["intent"]].append(c)

    train, val, test = [], [], []
    for intent, items in sorted(by_intent.items()):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        if n >= 8:
            n_test = max(2, round(n * 0.15))
            n_val = max(2, round(n * 0.15))
        elif n >= 4:
            n_test, n_val = 1, 1
        elif n == 3:
            n_test, n_val = 1, 1
        elif n == 2:
            n_test, n_val = 1, 0
        else:
            n_test, n_val = 0, 0
        test.extend(items[:n_test])
        val.extend(items[n_test:n_test + n_val])
        train.extend(items[n_test + n_val:])

    # -- split negatives -----------------------------------------------------
    rng.shuffle(negatives)
    m = len(negatives)
    m_test = max(2, round(m * 0.15))
    m_val = max(2, round(m * 0.15))
    neg_test = negatives[:m_test]
    neg_val = negatives[m_test:m_test + m_val]
    neg_train = negatives[m_test + m_val:]

    groups = list(TOOL_GROUPS)

    def render_pos(c):
        selected = TOOL_GROUPS[INTENT_GROUP[c["intent"]]]
        return make_example(c["query"], c["intent"], schemas, tools=selected)

    def render_neg(c, i):
        grp = groups[i % len(groups)]
        return {
            "query": c["query"],
            "tools": [schemas[t] for t in TOOL_GROUPS[grp]],
            "answers": [],
            "reasoning": "the request is not a supported local desktop action",
        }

    final_train = [render_pos(c) for c in train] + [render_neg(c, i) for i, c in enumerate(neg_train)]
    final_val = [render_pos(c) for c in val] + [render_neg(c, i) for i, c in enumerate(neg_val)]
    final_test = [render_pos(c) for c in test] + [render_neg(c, i) for i, c in enumerate(neg_test)]

    train = train + neg_train
    val = val + neg_val
    test = test + neg_test

    # -- exact overlap checks ------------------------------------------------
    def qset(rows):
        return {canon_query(r["query"]) for r in rows}

    tq, vq, eq = set(canon_query(c["query"]) for c in train), set(canon_query(c["query"]) for c in val), set(canon_query(c["query"]) for c in test)
    overlap_tv = sorted(tq & vq)
    overlap_tt = sorted(tq & eq)
    overlap_vt = sorted(vq & eq)

    # -- near-duplicate audit ------------------------------------------------
    records = []
    for split in ("train", "val", "test"):
        src = {"train": train, "val": val, "test": test}[split]
        for c in src:
            records.append((canon_query(c["query"]), split, c["intent"] or "<none>"))

    risky = []
    leak = []
    for i in range(len(records)):
        ca, sa, la = records[i]
        for j in range(i + 1, len(records)):
            cb, sb, lb = records[j]
            if ca == cb:
                continue
            sim = jaccard(ca, cb)
            if sim < 0.8:
                continue
            if la != lb:
                risky.append({"a": ca, "a_split": sa, "a_label": la,
                              "b": cb, "b_split": sb, "b_label": lb,
                              "jaccard": round(sim, 3)})
            elif sim >= 0.85 and sa != sb:
                leak.append({"a": ca, "a_split": sa,
                             "b": cb, "b_split": sb, "jaccard": round(sim, 3)})
    risky.sort(key=lambda r: -r["jaccard"])
    leak.sort(key=lambda r: -r["jaccard"])

    # -- near-duplicate vs the burned fresh-eval set ------------------------
    burned_near = []
    for ca, sa, la in records:
        best, bestq = 0.0, None
        for cb in burned_list:
            if ca == cb:
                continue
            s = jaccard(ca, cb)
            if s > best:
                best, bestq = s, cb
        if best >= 0.6:
            burned_near.append({
                "query": ca, "split": sa, "label": la,
                "burned": bestq, "jaccard": round(best, 3),
            })
    burned_near.sort(key=lambda r: -r["jaccard"])

    # -- summaries -----------------------------------------------------------
    split_pos = {
        "train": sum(1 for c in train if c["intent"]),
        "val": sum(1 for c in val if c["intent"]),
        "test": sum(1 for c in test if c["intent"]),
    }
    split_neg = {
        "train": sum(1 for c in train if not c["intent"]),
        "val": sum(1 for c in val if not c["intent"]),
        "test": sum(1 for c in test if not c["intent"]),
    }

    def intent_counts(items):
        d = defaultdict(int)
        for c in items:
            d[c["intent"] or "<none>"] += 1
        return dict(sorted(d.items()))

    per_intent = {
        "train": intent_counts(train),
        "val": intent_counts(val),
        "test": intent_counts(test),
        "all": intent_counts(train + val + test),
    }

    prov_counts = defaultdict(int)
    block_counts = defaultdict(int)
    for c in train + val + test:
        prov_counts[c["prov"]] += 1
        if c["prov"] == "v5" and c["block"]:
            block_counts[c["block"]] += 1

    pos_train = [c for c in train if c["intent"]]
    total_pos_train = len(pos_train)
    train_tool_counts = defaultdict(int)
    for c in pos_train:
        train_tool_counts[c["intent"]] += 1
    mean_pos = total_pos_train / max(1, len(train_tool_counts))
    overrep = []
    for tool, n in sorted(train_tool_counts.items(), key=lambda kv: -kv[1]):
        share = n / max(1, total_pos_train)
        if share > 0.12 or n > 2.5 * mean_pos:
            overrep.append({"tool": tool, "count": n, "share": round(share, 3),
                            "mean": round(mean_pos, 2)})

    all_pos = {k: v for k, v in per_intent["all"].items() if k != "<none>"}
    largest = max(all_pos.items(), key=lambda kv: kv[1]) if all_pos else ("<none>", 0)
    smallest = min(all_pos.items(), key=lambda kv: kv[1]) if all_pos else ("<none>", 0)

    # -- write ---------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path, rows):
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    rng.shuffle(final_train)
    rng.shuffle(final_val)
    rng.shuffle(final_test)

    write_jsonl(OUT_DIR / "vector_train.jsonl", final_train)
    write_jsonl(OUT_DIR / "vector_val.jsonl", final_val)
    write_jsonl(OUT_DIR / "vector_test.jsonl", final_test)
    (OUT_DIR / "needle_tools_core.json").write_text(
        json.dumps([schemas[t] for t in CORE_TOOLS], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    audit = {
        "generated_by": "scripts/prepare_v5_dataset.py",
        "core_tools": CORE_TOOLS,
        "totals": {
            "candidates": total_candidates,
            "kept_after_exclusion": len(kept),
            "after_set_volume_cap": len(capped),
            "after_dedup": len(deduped),
            "positives": len(positives),
            "negatives": len(negatives),
            "train": len(train), "val": len(val), "test": len(test),
        },
        "provenance": dict(prov_counts),
        "split_positives": split_pos,
        "split_negatives": split_neg,
        "block_counts": dict(sorted(block_counts.items())),
        "per_intent": per_intent,
        "exclusions": drop,
        "set_volume_cap": {"max_per_template": MAX_PER_TPL, "removed": cap_removed},
        "duplicates_removed": duplicate_removed,
        "ungrounded_removed": ungrounded,
        "conflicts": conflict_records,
        "exact_overlap": {
            "train_val": overlap_tv,
            "train_test": overlap_tt,
            "val_test": overlap_vt,
        },
        "v5_vs_burned_exact": 0,
        "near_duplicate_risky": risky[:50],
        "near_duplicate_risky_count": len(risky),
        "near_duplicate_cross_split_same_label": leak[:50],
        "near_duplicate_cross_split_count": len(leak),
        "burned_near_duplicate": burned_near[:50],
        "burned_near_duplicate_count": len(burned_near),
        "overrepresentation": overrep,
        "largest_class": {"name": largest[0], "count": largest[1]},
        "smallest_class": {"name": smallest[0], "count": smallest[1]},
    }
    (OUT_DIR / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    # -- markdown report -----------------------------------------------------
    L = []
    L.append("# V5 Dataset Report")
    L.append("")
    L.append("Generated by `scripts/prepare_v5_dataset.py` (deterministic, no training).")
    L.append("Seed content: `scripts/v5_seed_content.py`.")
    L.append("")
    L.append("## 1. Totals")
    L.append("")
    t = audit["totals"]
    L.append(f"- Candidate examples assembled: **{t['candidates']}**")
    L.append(f"- After burned/val/test exclusion: **{t['kept_after_exclusion']}**")
    L.append(f"- After set_volume template cap: **{t['after_set_volume_cap']}**")
    L.append(f"- After exact dedup: **{t['after_dedup']}**")
    L.append(f"- Final positives: **{t['positives']}**, negatives: **{t['negatives']}**")
    L.append("")
    L.append(f"- Train: **{t['train']}**  (positive {split_pos['train']} / negative {split_neg['train']})")
    L.append(f"- Validation: **{t['val']}**  (positive {split_pos['val']} / negative {split_neg['val']})")
    L.append(f"- Test: **{t['test']}**  (positive {split_pos['test']} / negative {split_neg['test']})")
    L.append("")
    L.append("Provenance (all splits): " + ", ".join(
        f"`{k}`={v}" for k, v in sorted(audit["provenance"].items())))
    L.append("")
    L.append("## 2. Examples per contrast block")
    L.append("")
    L.append("| Block | Examples |")
    L.append("| --- | ---: |")
    for k, v in audit["block_counts"].items():
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("`OFF_TOPIC` = hand-written off-topic negatives carried in from `v5_seed_content.OFF_TOPIC`.")
    L.append("")
    L.append("## 3. Examples per intent / tool")
    L.append("")
    L.append("| Tool | train | val | test | all |")
    L.append("| --- | ---: | ---: | ---: | ---: |")
    for tool in CORE_TOOLS:
        L.append("| {} | {} | {} | {} | {} |".format(
            tool,
            per_intent["train"].get(tool, 0),
            per_intent["val"].get(tool, 0),
            per_intent["test"].get(tool, 0),
            per_intent["all"].get(tool, 0),
        ))
    L.append(f"| _none (abstain/off-topic)_ | {per_intent['train'].get('<none>', 0)} | {per_intent['val'].get('<none>', 0)} | {per_intent['test'].get('<none>', 0)} | {per_intent['all'].get('<none>', 0)} |")
    L.append("")
    L.append("## 4. Integrity checks")
    L.append("")
    L.append(f"- Exact train/val overlap: **{len(overlap_tv)}**")
    L.append(f"- Exact train/test overlap: **{len(overlap_tt)}**")
    L.append(f"- Exact val/test overlap: **{len(overlap_vt)}**")
    L.append(f"- V5 vs burned 98-query eval exact overlap: **0** (excluded {drop['burned']} exact, {drop['burned_near']} reordered near-duplicates at Jaccard >= 0.90)")
    L.append(f"- Excluded against V4 val: **{drop['old_val']}**, against V4 test: **{drop['old_test']}**")
    L.append(f"- Duplicate (same canonical query) removed: **{duplicate_removed}**")
    L.append(f"- Ungrounded positives removed: **{ungrounded}**")
    L.append(f"- set_volume template cap: max **{MAX_PER_TPL}** per digit-normalized template, removed **{cap_removed}**")
    L.append("")
    L.append("## 5. Conflict audit")
    L.append("")
    if not conflict_records:
        L.append("No conflicting labels found for any canonical query.")
    else:
        L.append(f"**{len(conflict_records)}** canonical queries map to more than one label.")
        L.append("")
        L.append("| Query | Labels | Chosen | Provenance |")
        L.append("| --- | --- | --- | --- |")
        for c in conflict_records[:60]:
            L.append(f"| {c['query']} | {', '.join(c['labels'])} | {c['chosen']} | {', '.join(c['provenance'])} |")
    L.append("")
    L.append("## 6. Near-duplicate findings")
    L.append("")
    L.append(f"- Similar-but-different-label pairs (Jaccard >= 0.80): **{len(risky)}** (these are the intended contrast pairs, plus any accidents)")
    L.append(f"- Near-identical same-label pairs across different splits (Jaccard >= 0.85): **{len(leak)}**")
    L.append("")
    if risky[:15]:
        L.append("Sample contrast near-dups:")
        L.append("")
        L.append("| A | label | B | label | Jaccard |")
        L.append("| --- | --- | --- | --- | ---: |")
        for r in risky[:15]:
            L.append(f"| {r['a']} | {r['a_label']} | {r['b']} | {r['b_label']} | {r['jaccard']} |")
        L.append("")
    if leak[:10]:
        L.append("Cross-split near-identical (same label):")
        L.append("")
        for r in leak[:10]:
            L.append(f"- {r['a']} ({r['a_split']}) ~ {r['b']} ({r['b_split']})")
        L.append("")
    L.append(f"### Near-duplicates against the burned fresh-eval set (Jaccard >= 0.60): **{len(burned_near)}**")
    L.append("")
    if burned_near[:15]:
        L.append("| V5 query | split | label | closest burned query | Jaccard |")
        L.append("| --- | --- | --- | --- | ---: |")
        for r in burned_near[:15]:
            L.append(f"| {r['query']} | {r['split']} | {r['label']} | {r['burned']} | {r['jaccard']} |")
        L.append("")
    else:
        L.append("No V5 query is a near-duplicate of any burned-eval query at this threshold.")
        L.append("")
    L.append("## 7. Overrepresentation warnings")
    L.append("")
    if not overrep:
        L.append("No tool exceeds the 12% share / 2.5x-mean thresholds in the training split.")
    else:
        L.append("| Tool | train count | share of positives | mean |")
        L.append("| --- | ---: | ---: | ---: |")
        for o in overrep:
            L.append(f"| {o['tool']} | {o['count']} | {o['share']} | {o['mean']} |")
        L.append("")
        L.append("Mitigation applied: V4 `set_volume` paraphrase templates were capped at "
                 f"{MAX_PER_TPL} per digit-normalized template ({cap_removed} removed).")
    L.append("")
    L.append("## 8. Class size extremes")
    L.append("")
    L.append(f"- Largest class: **{largest[0]}** ({largest[1]} examples, all splits)")
    L.append(f"- Smallest class: **{smallest[0]}** ({smallest[1]} examples, all splits)")
    L.append("")
    L.append("## 9. Output paths")
    L.append("")
    for p in ("vector_train.jsonl", "vector_val.jsonl", "vector_test.jsonl",
              "needle_tools_core.json", "audit.json"):
        L.append(f"- `data/v5/{p}`")
    L.append("")
    report = "\n".join(L) + "\n"
    (ROOT / "eval" / "V5_DATASET_REPORT.md").write_text(report, encoding="utf-8")

    print(f"Candidates: {total_candidates}")
    print(f"Final positives: {len(positives)}  negatives: {len(negatives)}")
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    print(f"Conflicts: {len(conflict_records)}  risky near-dups: {len(risky)}  cross-split leaks: {len(leak)}")
    print(f"Wrote: {OUT_DIR}")
    print(f"Wrote: {ROOT / 'eval' / 'V5_DATASET_REPORT.md'}")


if __name__ == "__main__":
    main()

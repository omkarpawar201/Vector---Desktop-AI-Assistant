"""
Manual tester for Vector's candidate-routed Needle 3 pipeline.

Dry run only: it prints what would be called and never executes anything, so
it is safe without the desktop deps installed.

    # batch, built-in queries
    venv-linux/bin/python scripts/try_vector.py

    # type your own
    venv-linux/bin/python scripts/try_vector.py --interactive

    # reproduce the old behaviour (all 17 tools in one prompt) for comparison
    venv-linux/bin/python scripts/try_vector.py --all-tools

Note the difference from the old test script: it passed all 17 tools at once,
which is out-of-distribution for V4 (it was fine-tuned with 4-5) and measured
23.1% on the held-out set. The default here offers 4-5 candidates chosen by
rules first, which measured 100%.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.config.settings import Settings
from app.core import candidates as C
from app.core.matcher import Tier0Matcher
from app.needle.client import NeedleClient

QUERIES = [
    # volume
    "Set volume to 50", "Make volume 25", "What is the current volume?",
    "Mute the volume", "Unmute", "volume up to 50", "am i muted",
    "silence audio", "disable sound", "enable audio again",
    # system monitoring
    "What is my CPU usage?", "How much RAM am I using?",
    "How much disk space is used?", "What is my battery status?",
    "What apps are running?", "show my disk space", "check memory",
    "is the laptop charging", "system overview",
    # app management
    "Open Chrome", "Launch VS Code", "Close Spotify", "can you open chrome?",
    "what programs are open",
    # media
    "Play music", "Pause playback", "Skip this song", "Previous track",
    "start playing music", "pause the song", "skip forward",
    "go back to the previous song",
    # the hard ones: "resume" is ambiguous
    "find my resume", "Can you find my resume?", "where is my resume pdf",
    "Resume playback", "Continue playing",
    # untrained tools: must be rules-only, never model-selected
    "lock the pc", "shutdown pc", "maximize window",
    "what is the active window", "open report.pdf", "open the folder documents",
    # off-topic: must escalate, never reach a tool
    "what is the weather like", "explain object oriented programming",
    "write a professional email",
]


class SnapshotRegistry:
    """Tool schemas from the registry's own export, without the desktop deps.

    ``register_all_default_tools()`` imports pyautogui/PySide6/sounddevice.
    ``data/vector_tool_schemas.json`` is that same export (scripts/
    export_tool_schemas.py) and was verified identical to the schemas V4 was
    fine-tuned on, so it is what the production path feeds the model.
    """

    def __init__(self):
        self._schemas = json.loads(
            (ROOT / "data" / "vector_tool_schemas.json").read_text())

    def export_schemas(self):
        return self._schemas

    def get(self, name):
        return next((s for s in self._schemas if s["name"] == name), None)


def show(client, query, all_tools=False):
    print(f"\n  {query!r}")

    if all_tools:
        # Reproduce the old test script exactly: no rules, all 17 tools in one
        # prompt. This is the out-of-distribution shape V4 scored 23.1% on.
        trained = [t for t in client._schemas_by_name().values()
                   if t["name"] in C.TRAINED_TOOLS]
        picked = C.CandidateSet(family="ALL-17 (out-of-distribution)",
                                tools=tuple(t["name"] for t in trained))
        result = client.predict_cactus_needle_intent(query, selected=picked)
        print(f"    all 17 tools  -> "
              + (f"{result.tool_name} {json.dumps(result.arguments)}"
                 if result.admitted else f"not admitted: {result.reason}"))
        return result.tool_name if result.admitted else None

    t0 = Tier0Matcher.match(query)
    if t0.matched:
        args_out = f" {json.dumps(t0.arguments)}" if t0.arguments else ""
        print(f"    Tier 0 rules   -> {t0.tool_name}{args_out}")
        return t0.tool_name
    print("    Tier 0 rules   -> no match")

    picked = client.select_candidates(query)
    if picked is None:
        print("    candidate set  -> none, escalate to Tier 2 (Gemini)")
        return None
    print(f"    candidate set  -> {picked.family} "
          f"[{', '.join(picked.tools)}] via {picked.source}")
    result = client.predict_cactus_needle_intent(query, selected=picked)
    if result.admitted:
        args = json.dumps(result.arguments) if result.arguments else "{}"
        print(f"    Needle 3       -> {result.tool_name} {args}")
        return result.tool_name
    print(f"    Needle 3       -> not admitted: {result.reason}")
    return None


def show_compare(client, query):
    """Old 17-tool shape vs candidate-routed, side by side."""
    trained = [t for t in client._schemas_by_name().values()
               if t["name"] in C.TRAINED_TOOLS]
    wide = C.CandidateSet(family="ALL-17 (out-of-distribution)",
                          tools=tuple(t["name"] for t in trained))
    old = client.predict_cactus_needle_intent(query, selected=wide)
    old_txt = (f"{old.tool_name} {json.dumps(old.arguments)}" if old.admitted
               else "not admitted")

    t0 = Tier0Matcher.match(query)
    if t0.matched:
        new_txt = f"{t0.tool_name} {json.dumps(t0.arguments)}" if t0.arguments \
            else t0.tool_name
        via = "rules"
    else:
        picked = client.select_candidates(query)
        if picked is None:
            new_txt, via = "escalate to Tier 2", "none"
        else:
            res = client.predict_cactus_needle_intent(query, selected=picked)
            new_txt = (f"{res.tool_name} {json.dumps(res.arguments)}"
                       if res.admitted else "not admitted")
            via = picked.family

    flag = "  " if old_txt == new_txt else "<-"
    print(f"  {query!r}\n"
          f"      all-17 : {old_txt}\n"
          f"      routed : {new_txt}  ({via}) {flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactive", "-i", action="store_true",
                    help="type queries instead of running the built-in list")
    ap.add_argument("--all-tools", action="store_true",
                    help="offer all 17 tools at once, like the old test script")
    ap.add_argument("--compare", action="store_true",
                    help="show all-17 vs candidate-routed side by side")
    ap.add_argument("--query", "-q", action="append",
                    help="test one query; repeatable")
    args = ap.parse_args()

    client = NeedleClient(Settings(), SnapshotRegistry())
    weights = ROOT / "models" / "needle3_vector_v4.cact"
    print(f"model     : {weights.name} ({'found' if weights.exists() else 'MISSING'})")
    if args.compare:
        print("mode      : compare (all-17 vs candidate-routed)")
    elif args.all_tools:
        print("mode      : all 17 tools (out-of-distribution, no rules)")
    else:
        print("mode      : candidate-routed (4-5 tools)")
    print("execution : dry run, nothing is actually run")

    runner = (lambda q: show_compare(client, q)) if args.compare else \
             (lambda q: show(client, q, args.all_tools))

    try:
        if args.interactive and not args.query:
            print("\nType a query, or 'quit' to exit.\n")
            while True:
                try:
                    line = input("  > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if line.lower() in ("quit", "exit", "q"):
                    break
                if line:
                    runner(line)
        else:
            queries = args.query or QUERIES
            for query in queries:
                runner(query)
            print(f"\n{'=' * 72}\ndone: {len(queries)} queries")
    finally:
        client.close()


if __name__ == "__main__":
    main()

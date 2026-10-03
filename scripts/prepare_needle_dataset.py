"""
Prepare a Vector-specific Needle fine-tuning dataset from the existing
desktop intent dataset and the real ToolRegistry schemas.

Expected project layout:
    app/
    data/vector_tool_schemas.json
    scripts/build_needle_onnx.py
    scripts/prepare_needle_dataset.py

Outputs:
    data/needle_tools_core.json
    data/vector_train.jsonl
    data/vector_val.jsonl
    data/vector_test.jsonl

This is a deterministic, $0 dataset builder. It does not call OpenRouter.
"""
from __future__ import annotations

import ast
import json
import random
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_FILE = ROOT / "data" / "vector_tool_schemas.json"
SEED_FILE = ROOT / "scripts" / "build_needle_onnx.py"
OUT_DIR = ROOT / "data"

# First Vector fine-tune: exclude risky/less mature tools.
CORE_TOOLS = [
    "get_system_stats", "get_cpu_usage", "get_memory_usage",
    "get_disk_usage", "get_battery_status",
    "get_volume", "set_volume", "mute", "unmute",
    "media_play", "media_pause", "media_next", "media_previous",
    "launch_app", "close_app", "get_running_apps",
    "search_files",
]

INTENT_TO_TOOL = {name: name for name in CORE_TOOLS}

# Tools are grouped so each training example sees a small, meaningful
# catalogue containing hard negatives rather than a huge 31-tool prompt.
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

# Common app aliases. These values are intentionally simple because they
# are passed to Vector's launch_app/close_app tool as the "name" argument.
APP_CANONICAL = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "firefox": "firefox",
    "notepad": "notepad",
    "vscode": "vscode",
    "vs code": "vscode",
    "code": "vscode",
    "calculator": "calculator",
    "calc": "calculator",
    "file explorer": "file explorer",
    "explorer": "file explorer",
    "word": "word",
    "terminal": "terminal",
}

# Additional hand-written seeds to strengthen the small/underrepresented
# intents. These are local, deterministic examples; no API is used.
EXTRA_SEEDS = {
    # =========================================================
    # VOLUME
    # =========================================================

    "get_volume": [
        "what is the current volume",
        "check my volume",
        "what volume is set",
        "show the current sound level",
        "how loud is the computer",
        "what is my current volume level",
        "tell me the volume",
        "check the sound level",
        "what is the sound volume",
        "how high is the volume",
        "show me the volume level",
        "what is the audio level",
        "check current audio volume",
        "how loud is my PC",
        "tell me how loud it is",
        "what is the speaker volume",
        "show current volume",
        "is the volume high",
        "is the sound level currently set",
        "what level is the volume at",
        "check how loud the speakers are",
        "what is my speaker level",
        "show audio level",
        "tell me the current sound level",
        "how high is my sound",
        "check my audio level",
        "what is the current speaker volume",
        "how loud are my speakers",
        "display the current volume",
    ],

    "set_volume": [
        "set volume to 0",
        "set volume to 5",
        "set volume to 10",
        "set volume to 15",
        "set volume to 25",
        "set volume to 30",
        "set volume to 33",
        "set volume to 45",
        "set volume to 50",
        "set volume to 60",
        "set volume to 65",
        "set volume to 70",
        "set volume to 75",
        "set volume to 80",
        "set volume to 85",
        "set volume to 90",
        "set volume to 95",
        "set volume to 100",

        "make volume 60",
        "make the volume 25",
        "make volume 50",
        "change volume to 75",
        "change the volume to 45",
        "change volume to 30",
        "adjust volume to 55 percent",
        "adjust the volume to 65",
        "set the volume at 30 percent",
        "set the volume to 85",
        "set the volume to 30 percent",
        "set the volume to fifty percent",
        "set the volume to sixty percent",
        "set the volume to forty five percent",
        "set volume level to 20",
        "set volume level to 50",
        "set my volume to 70",
        "set my volume to 25 percent",
        "set speakers to 35 percent",

        "increase volume to 40",
        "increase volume to 60 percent",
        "lower volume to 15",
        "decrease volume to 25 percent",
        "turn volume up to 30",
        "turn volume down to 10",
        "volume up to 50",
    ],

    "mute": [
        "mute the computer",
        "mute the sound",
        "mute audio",
        "mute my speakers",
        "mute the speakers",
        "turn off the sound",
        "turn the sound off",
        "silence audio",
        "silence the computer",
        "silence my speakers",
        "turn off audio",
        "disable sound",
        "stop the sound",
    ],

    "unmute": [
        "unmute the computer",
        "unmute the sound",
        "unmute audio",
        "unmute my speakers",
        "turn the sound back on",
        "turn audio back on",
        "enable audio again",
        "restore the sound",
        "restore audio",
        "turn the speakers back on",
        "enable sound",
        "bring the sound back",
    ],

    # =========================================================
    # MEDIA CONTROL
    # =========================================================

    "media_play": [
        "play music",
        "play the music",
        "play the song",
        "start playback",
        "start playing",
        "start playing music",
        "start the music",
        "resume playback",
        "resume playing",
        "resume the music",
        "continue playback",
        "continue playing",
        "continue playing music",
        "unpause music",
        "unpause playback",
        "play the current song",
        "start the current track",
        "play the current track",
    ],

    "media_pause": [
        "pause the music",
        "pause music",
        "pause the song",
        "pause the current song",
        "pause the current track",
        "pause playback",
        "pause the player",
        "pause the media",
        "pause what is playing",
        "temporarily stop playback",
        "temporarily pause the music",
        "stop the music for now",
        "stop playback",
        "stop the current song",
        "stop the current track",
        "hold the music",
        "hold playback",
    ],

    "media_next": [
        "next song",
        "next track",
        "go to the next song",
        "go to the next track",
        "play the next song",
        "play the next track",
        "skip this song",
        "skip the song",
        "skip this track",
        "skip the track",
        "skip to the next song",
        "skip to the next track",
        "skip forward",
        "move to the next song",
        "move to the next track",
        "go forward to the next song",
    ],

    "media_previous": [
        "play the previous song",
        "play the previous track",
        "previous song",
        "previous track",
        "go back one song",
        "go back one track",
        "go back to the previous song",
        "go back to the previous track",
        "go to the previous song",
        "go to the previous track",
        "return to the last song",
        "return to the previous song",
        "move back one track",
        "move back one song",
        "switch to the previous song",
        "switch to the previous track",
        "play the previous track",
        "play the previous song",
        "previous",
    ],

    # =========================================================
    # SYSTEM MONITORING
    # =========================================================

    "get_system_stats": [
        "show me the system information",
        "give me a system status",
        "check the computer status",
        "show the overall system status",
        "display system information",
        "show system stats",
        "show computer statistics",
        "give me computer information",
        "show my system information",
        "what are my system stats",
    ],

    "get_cpu_usage": [
        "how much cpu am i using",
        "show my cpu usage",
        "check my cpu",
        "how much processor am i using",
        "show cpu usage",
        "check cpu usage",
        "what is my cpu usage",
        "how busy is my processor",
        "how much processing power is being used",
        "what percentage of cpu am i using",
        "tell me my cpu usage",
        "check the processor usage",
        "show me the cpu load",
        "how heavily is the cpu being used",
    ],

    "get_memory_usage": [
        "how much ram am i using",
        "show my ram usage",
        "check my ram",
        "how much memory am i using",
        "show memory usage",
        "check memory usage",
        "what is my memory usage",
        "how much system memory is being used",
        "how much ram is being used",
        "what percentage of ram am i using",
        "what percentage of memory am i using",
        "tell me my ram usage",
        "tell me my memory usage",
        "check the system memory",
        "show me the memory usage",
    ],

    "get_disk_usage": [
        "how much disk space do I have",
        "show free storage",
        "check available disk space",
        "what storage is available",
        "show my disk space",
        "how much storage is left",
        "how much disk space is left",
        "check my storage",
        "show storage usage",
        "what is my disk usage",
        "how much space is available",
        "how much free space do I have",
        "show available storage",
        "check free disk space",
    ],

    "get_battery_status": [
        "check the laptop battery",
        "what percentage is the battery",
        "show battery information",
        "is my battery charging",
        "how much charge is left",
        "what is my battery level",
        "how much battery do I have",
        "check my battery",
        "show battery status",
        "what is the current battery percentage",
        "is the laptop charging",
        "is my laptop plugged in",
        "how much battery is remaining",
    ],

    "get_running_apps": [
        "show the applications currently running",
        "what programs are open",
        "list the programs that are running",
        "show active programs",
        "which apps are currently open",
        "show running applications",
        "what apps are running",
        "list running apps",
        "show open applications",
        "what applications are open",
        "which programs are running",
        "show me all running programs",
        "tell me what applications are running",
    ],

    # =========================================================
    # APP LAUNCH vs CLOSE — CONTRASTIVE EXAMPLES
    # =========================================================
    "launch_app": [
        "open chrome",
        "open google chrome",
        "start chrome",
        "launch chrome",
        "start the chrome browser",
        "open the chrome browser",
        "can you open chrome",
        "can you start chrome",
        "please open chrome",
        "please launch chrome",
        "open vscode",
        "start vscode",
        "launch vscode",
        "can you open vscode",
        "please open vscode",
    ],

    "close_app": [
        "close chrome",
        "close google chrome",
        "stop chrome",
        "exit chrome",
        "quit chrome",
        "can you close chrome",
        "please close chrome",
        "close the chrome browser",
        "stop the chrome browser",
        "close vscode",
        "stop vscode",
        "exit vscode",
        "can you close vscode",
        "please close vscode",
    ],
}

EXTRA_NEGATIVES = [
    "write a Java program",
    "explain machine learning",
    "translate this into Hindi",
    "what is the capital of India",
    "help me write a resume",
    "summarize this paragraph",
    "tell me a fun fact",
    "what is recursion",
    "explain Docker",
    "write an SQL query",
    "help me debug this Python code",
    "what happened in the news today",
    "give me a recipe",
    "plan a weekend trip",
    "explain how transformers work",
    "what is the difference between TCP and UDP",
    "write a professional email",
    "teach me Java inheritance",
    "convert this text to Marathi",
    "what is the weather like",
    "help me prepare for an interview",
    "explain the Linux file system",
    "generate a poem",
    "what is an API",
    "how does a neural network learn",
    "give me a book recommendation",
    "explain object oriented programming",
    "what is the meaning of this word",
    "help me with a math problem",
    "write a short story",
]

def load_schemas():
    if not SCHEMA_FILE.exists():
        raise SystemExit(
            f"Missing {SCHEMA_FILE}. Run export_tool_schemas.py first."
        )
    raw = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    by_name = {x["name"]: x for x in raw}
    missing = [x for x in CORE_TOOLS if x not in by_name]
    if missing:
        raise SystemExit(f"Missing core schemas: {missing}")
    return by_name

def load_seed_dataset():
    """Extract D=[(utterance,intent), ...] from the existing Python file."""
    if not SEED_FILE.exists():
        raise SystemExit(f"Missing seed file: {SEED_FILE}")
    tree = ast.parse(SEED_FILE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_build_dataset":
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    if any(isinstance(t, ast.Name) and t.id == "D"
                           for t in child.targets):
                        return ast.literal_eval(child.value)
    raise SystemExit("Could not find _build_dataset() / D in seed file.")

def extract_app(text: str) -> str | None:
    low = text.lower()
    # Prefer longest alias first.
    for alias in sorted(APP_CANONICAL, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", low):
            return APP_CANONICAL[alias]
    return None

def search_query(text: str) -> str:
    low = text.strip()

    patterns = [
        # Conversational forms
        r"^(?:can you|could you|would you|please|pls)\s+"
        r"(?:find|search for|search|look for|locate)\s+"
        r"(?:my\s+|a\s+|an\s+|the\s+)?(.+)$",

        # Direct search forms
        r"^(?:find|search for|search|look for|locate)\s+"
        r"(?:my\s+|a\s+|an\s+|the\s+)?(.+)$",

        # Where-is forms
        r"^where is\s+(?:my\s+|the\s+)?(.+)$",
    ]

    for p in patterns:
        m = re.match(p, low, re.I)
        if m:
            q = m.group(1).strip()

            # Remove location qualifiers that aren't part of the search term
            q = re.sub(
                r"\b(on disk|in my downloads|on my disk)\b",
                "",
                q,
                flags=re.I,
            ).strip()

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
        return {"query": search_query(query)}

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

def make_example(query, intent, schemas, tools=None):
    if intent not in CORE_TOOLS:
        return None
    args = make_arguments(intent, query)
    if args is None:
        return None

    selected = tools or TOOL_GROUPS[INTENT_GROUP[intent]]
    return {
        "query": query,
        "tools": [schemas[t] for t in selected],
        "answers": [{"name": intent, "arguments": args}],
        "reasoning": reasoning_for(intent, args, query),
    }

def add_paraphrases(intent: str, base_query: str):
    """Small deterministic augmentation; no external API."""
    q = base_query.strip()
    out = {q}
    low = q.lower()

    if intent == "launch_app":
        app = extract_app(q)
        if app:
            variants = [
                f"open {app}", f"launch {app}", f"start {app}",
                f"can you open {app}", f"please open {app}",
                f"bring up {app}", f"run {app}",
            ]
            out.update(variants)

    elif intent == "close_app":
        app = extract_app(q)
        if app:
            out.update([
                f"close {app}", f"quit {app}", f"exit {app}",
                f"stop {app}", f"terminate {app}",
                f"can you close {app}", f"please close {app}",
            ])

    elif intent == "set_volume":
        m = re.search(r"\b(\d{1,3})\b", q)
        if m:
            n = max(0, min(100, int(m.group(1))))
            out.update([
                f"set volume to {n}",
                f"set the volume to {n} percent",
                f"make volume {n}",
                f"volume at {n} percent",
                f"set sound level to {n}",
                f"change volume to {n}",
            ])

    elif intent == "search_files":
        term = search_query(q)
        out.update([
            f"find {term}",
            f"search for {term}",
            f"look for {term}",
            f"locate {term}",
            f"can you find {term}",
        ])

    elif intent == "get_running_apps":
        out.update([
            "what apps are running",
            "show running apps",
            "list active applications",
            "what is open right now",
            "which applications are open",
        ])

    elif intent == "get_system_stats":
        out.update([
            "show system stats", "system overview",
            "give me a system summary", "show machine stats",
        ])
    elif intent == "get_cpu_usage":
        out.update([
            "what is my cpu usage", "check cpu usage",
            "show cpu load", "how much cpu is being used",
        ])
    elif intent == "get_memory_usage":
        out.update([
            "what is my ram usage", "check ram",
            "how much memory is being used", "show memory usage",
        ])
    elif intent == "get_disk_usage":
        out.update([
            "how much disk space is free", "check storage",
            "show disk usage", "how much storage is left",
        ])
    elif intent == "get_battery_status":
        out.update([
            "what is my battery percentage", "check battery status",
            "how much battery do I have", "is the laptop charging",
        ])
    elif intent == "get_volume":
        out.update([
            "what is the current volume", "check volume",
            "what volume is set", "show current sound level",
        ])
    elif intent == "mute":
        out.update(["mute audio", "mute the sound", "silence the computer audio"])
    elif intent == "unmute":
        out.update(["unmute audio", "turn sound back on", "restore audio"])
    elif intent == "media_play":
        out.update(["play", "resume music", "resume playback", "continue playing"])
    elif intent == "media_pause":
        out.update(["pause music", "pause playback", "stop playback temporarily"])
    elif intent == "media_next":
        out.update(["next track", "skip to the next song", "skip this track"])
    elif intent == "media_previous":
        out.update(["previous track", "go to the previous song", "go back one track"])

    return out

def main():
    schemas = load_schemas()
    raw = load_seed_dataset()

    # Deduplicate by query + intent.
    seeds = []
    seen = set()
    for query, intent in raw:
        key = (query.strip().lower(), intent)
        if intent not in CORE_TOOLS or key in seen:
            continue
        seen.add(key)
        seeds.append((query.strip(), intent))

    # Add deterministic extra seeds for underrepresented intents.
    for intent, queries in EXTRA_SEEDS.items():
        for query in queries:
            key = (query.strip().lower(), intent)
            if intent in CORE_TOOLS and key not in seen:
                seen.add(key)
                seeds.append((query.strip(), intent))

    # Split the ORIGINAL seed utterances first, so test examples never leak
    # through deterministic augmentation.
    by_intent = defaultdict(list)
    for q, intent in seeds:
        by_intent[intent].append(q)

    rng = random.Random(42)
    train_seed, val_seed, test_seed = set(), set(), set()

    for intent, qs in by_intent.items():
        qs = list(dict.fromkeys(qs))
        rng.shuffle(qs)
        n = len(qs)
        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))
        test = qs[:n_test]
        val = qs[n_test:n_test+n_val]
        train = qs[n_test+n_val:]
        if not train:
            train = val[:-1]
            val = val[-1:]
        train_seed.update((q, intent) for q in train)
        val_seed.update((q, intent) for q in val)
        test_seed.update((q, intent) for q in test)

    def render(seed_pairs, augment=False):
        examples = []
        seen_queries = set()
        for base, intent in seed_pairs:
            queries = add_paraphrases(intent, base) if augment else {base}
            for q in sorted(queries):
                # Never create an example whose required argument cannot be
                # grounded in the query.
                ex = make_example(q, intent, schemas)
                if ex is None:
                    continue
                key = (q.lower(), intent)
                if key in seen_queries:
                    continue
                seen_queries.add(key)
                examples.append(ex)
        return examples

    train = render(train_seed, augment=True)
    val = render(val_seed, augment=False)
    test = render(test_seed, augment=False)

    # Prevent exact train/test and train/validation leakage caused by
    # deterministic paraphrases. Validation/test queries remain untouched.
    held_out_queries = {
        ex["query"].strip().lower()
        for ex in (val + test)
    }
    train = [
        ex for ex in train
        if ex["query"].strip().lower() not in held_out_queries
    ]

    # Add explicit off-topic examples to all three splits. For each split,
    # use a different slice so the negative queries themselves do not leak.
    negatives = EXTRA_NEGATIVES
    rng.shuffle(negatives)
    neg_train = negatives[:20]
    neg_val = negatives[20:26]
    neg_test = negatives[26:32]

    def make_negative(q, group):
        return {
            "query": q,
            "tools": [schemas[t] for t in TOOL_GROUPS[group]],
            "answers": [],
            "reasoning": "the request is not a supported local desktop action",
        }

    groups = list(TOOL_GROUPS)
    for i, q in enumerate(neg_train):
        train.append(make_negative(q, groups[i % len(groups)]))
    for i, q in enumerate(neg_val):
        val.append(make_negative(q, groups[i % len(groups)]))
    for i, q in enumerate(neg_test):
        test.append(make_negative(q, groups[i % len(groups)]))

    # Deterministic shuffle.
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    core_schema_list = [schemas[t] for t in CORE_TOOLS]
    (OUT_DIR / "needle_tools_core.json").write_text(
        json.dumps(core_schema_list, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    def write_jsonl(path, rows):
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    write_jsonl(OUT_DIR / "vector_train.jsonl", train)
    write_jsonl(OUT_DIR / "vector_val.jsonl", val)
    write_jsonl(OUT_DIR / "vector_test.jsonl", test)

    print(f"Seed examples: {len(seeds)}")
    print(f"Train:        {len(train)}")
    print(f"Validation:   {len(val)}")
    print(f"Test:         {len(test)}")
    print("Skipped seed examples whose required arguments were not present in the query.")
    print(f"Wrote: {OUT_DIR / 'vector_train.jsonl'}")
    print(f"Wrote: {OUT_DIR / 'vector_val.jsonl'}")
    print(f"Wrote: {OUT_DIR / 'vector_test.jsonl'}")

    # Sanity checks.
    tq = {x["query"].strip().lower() for x in train}
    vq = {x["query"].strip().lower() for x in val}
    eq = {x["query"].strip().lower() for x in test}
    print(f"Exact Train/Val overlap:  {len(tq & vq)}")
    print(f"Exact Train/Test overlap: {len(tq & eq)}")
    print(f"Exact Val/Test overlap:   {len(vq & eq)}")
    print()
    print("Next: inspect the JSONL before training. In particular, review")
    print("search_files and app-name arguments for your actual tool semantics.")

if __name__ == "__main__":
    main()

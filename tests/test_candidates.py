"""
Tests for candidate-tool routing and the safety invariants around it.

The important assertions here are the *negative* ones: that the model is never
offered an untrained tool, and that a call naming one is refused even if the
model somehow produces it.
"""

import json
from pathlib import Path

import pytest

from app.core import candidates as C
from app.core.matcher import Tier0Matcher

ROOT = Path(__file__).resolve().parent.parent


def _held_out():
    rows = []
    for name in ("vector_val.jsonl", "vector_test.jsonl"):
        path = ROOT / "data" / name
        if path.exists():
            rows.extend(json.loads(l) for l in path.read_text().splitlines() if l.strip())
    return rows


FAMILY_OF = {tool: family for family, tools in C.FAMILIES.items() for tool in tools}


# ---------------------------------------------------------------------------
# Candidate set shape
# ---------------------------------------------------------------------------

def test_only_seventeen_trained_tools_are_offered():
    assert len(C.TRAINED_TOOLS) == 17
    for family, tools in C.FAMILIES.items():
        assert tools, f"{family} is empty"
        for tool in tools:
            assert tool in C.TRAINED_TOOLS


def test_untrained_tools_never_appear_in_a_candidate_set():
    assert not (C.TRAINED_TOOLS & C.UNTRAINED_TOOLS)
    for tools in C.FAMILIES.values():
        assert not (set(tools) & C.UNTRAINED_TOOLS)


def test_candidate_sets_match_the_training_groups():
    """The candidate sets must mirror data/vector_train.jsonl exactly.

    If these drift apart, V4 is being run out-of-distribution again, which is
    the failure this whole module exists to prevent.
    """
    path = ROOT / "data" / "vector_train.jsonl"
    if not path.exists():
        pytest.skip("training data not present")
    groups = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        groups.add(tuple(t["name"] for t in row.get("tools", [])))
    for group in groups:
        assert group in {tuple(t) for t in C.FAMILIES.values()}, (
            f"training group {group} has no matching candidate set"
        )


# ---------------------------------------------------------------------------
# Family routing
# ---------------------------------------------------------------------------

def test_family_routing_on_held_out():
    rows = [r for r in _held_out() if r.get("answers")]
    if not rows:
        pytest.skip("held-out data not present")
    wrong = []
    for row in rows:
        expected = FAMILY_OF.get(row["answers"][0]["name"])
        picked = C.select_by_rules(row["query"])
        if picked is None or picked.family != expected:
            wrong.append((row["query"], expected, picked.family if picked else None))
    assert not wrong, f"family routing misses: {wrong}"


def test_negatives_are_not_routed():
    rows = [r for r in _held_out() if not r.get("answers")]
    if not rows:
        pytest.skip("held-out data not present")
    routed = [r["query"] for r in rows if C.select_by_rules(r["query"]) is not None]
    assert not routed, f"off-topic queries were routed to a tool family: {routed}"


def test_embedding_fallback_is_off_by_default():
    """Cosine retrieval was measured unsafe, so it must not be on the path.

    Off-topic rows scored 0.929-0.943 against the tool descriptions while
    correctly-routed rows scored 0.909-0.978: the ranges overlap, so no
    threshold separates them. With it enabled, "what is the weather like"
    resolved to system_monitoring at 0.938.
    """
    snapshots = ROOT / "data" / "vector_tool_schemas.json"
    rows = [r for r in _held_out() if not r.get("answers")]
    if not snapshots.exists() or not rows:
        pytest.skip("schema snapshot or negatives not present")
    trained = [t for t in json.loads(snapshots.read_text())
               if t["name"] in C.TRAINED_TOOLS]

    for row in rows:
        assert C.select(row["query"], tools=trained) is None, (
            f"{row['query']!r} reached the model without a rules match"
        )
    assert C.select("what is the weather like") is None
    assert C.select("explain object oriented programming") is None


def test_rules_still_beat_the_embedding_fallback():
    """Guard against re-enabling the fallback: it is strictly worse."""
    assert C.select("how much ram am i using") is not None
    assert C.select_by_rules("how much ram am i using").family == "system_monitoring"


@pytest.mark.parametrize("query,family", [
    ("find my resume", "app_management"),
    ("where is my resume pdf", "app_management"),
    ("can you find my report", "app_management"),
    ("resume playback", "media_control"),
    ("continue playing", "media_control"),
    ("how much ram am i using", "system_monitoring"),
    ("how much cpu am i using", "system_monitoring"),
    ("set volume to 30", "volume_control"),
])
def test_resume_ambiguity_resolves_deterministically(query, family):
    picked = C.select_by_rules(query)
    assert picked is not None, f"{query!r} was not routed"
    assert picked.family == family


@pytest.mark.parametrize("query", [
    "shutdown the pc",
    "restart my computer",
    "maximize the current window",
    "open the folder documents",
    "open report.pdf",
    "run a terminal command",
])
def test_uncovered_intents_escalate_instead_of_routing(query):
    """Tools with no training coverage must not be routed to the model."""
    assert C.select_by_rules(query) is None


# ---------------------------------------------------------------------------
# Admission control
# ---------------------------------------------------------------------------

def test_admits_a_tool_inside_the_candidate_set():
    picked = C.select_by_rules("how much ram am i using")
    assert picked.family == "system_monitoring"
    assert picked.allows("get_memory_usage")
    # Siblings inside the same family stay admissible: the model is allowed to
    # override an imperfect family pick, just never to leave the family.
    assert picked.allows("get_cpu_usage")
    assert not picked.allows("get_volume")


def test_rejects_a_tool_outside_the_candidate_set():
    picked = C.select_by_rules("how much ram am i using")
    assert not picked.allows("media_play")
    assert "outside candidate set" in picked.reject_reason("media_play")


@pytest.mark.parametrize("tool", sorted(C.UNTRAINED_TOOLS))
def test_rejects_every_untrained_tool(tool):
    """Even inside a candidate set, an untrained tool is never admissible."""
    for family, tools in C.FAMILIES.items():
        picked = C.CandidateSet(family=family, tools=tools)
        assert not picked.allows(tool)
        assert "untrained" in picked.reject_reason(tool)


def test_rejects_empty_tool_name():
    picked = C.select_by_rules("play music")
    assert not picked.allows("")
    assert "no tool name" in picked.reject_reason("")


# ---------------------------------------------------------------------------
# Tier 0 coverage of what the model gets wrong
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,tool", [
    ("skip this song", "media_next"),
    ("next song", "media_next"),
    ("go forward", "media_next"),
    ("go back to the previous song", "media_previous"),
    ("pause the song", "media_pause"),
    ("pause playback", "media_pause"),
    ("start playing music", "media_play"),
    ("can you find my resume?", "search_files"),
    ("where is my resume pdf", "search_files"),
    ("am i muted", "get_volume"),
    ("turn the speakers back on", "unmute"),
    ("silence audio", "mute"),
    ("can you open chrome?", "launch_app"),
    ("kill spotify", "close_app"),
])
def test_tier0_covers_known_model_failures(query, tool):
    result = Tier0Matcher.match(query)
    assert result.matched, f"{query!r} not matched at Tier 0"
    assert result.tool_name == tool, f"{query!r} -> {result.tool_name}, want {tool}"


def test_media_command_never_becomes_an_app_launch():
    """Regression: "start playing music" once resolved to launch_app."""
    result = Tier0Matcher.match("start playing music")
    assert not (result.matched and result.tool_name == "launch_app")


def test_file_paths_survive_normalization():
    """InputNormalizer strips '.', '/', '\\'; the path rules must not."""
    result = Tier0Matcher.match("open report.pdf")
    assert result.matched and result.tool_name == "open_file"
    assert result.arguments["path"] == "report.pdf"

    result = Tier0Matcher.match("get file info for ~/Documents/notes.docx")
    assert result.matched and result.tool_name == "get_file_info"
    assert result.arguments["path"] == "~/documents/notes.docx"


def test_power_and_window_commands_are_rules_only():
    for query, tool in [
        ("lock the pc", "lock_pc"),
        ("shutdown pc", "shutdown_pc"),
        ("restart the computer", "restart_pc"),
        ("maximize window", "maximize_window"),
        ("what is the active window", "get_active_window"),
    ]:
        result = Tier0Matcher.match(query)
        assert result.matched and result.tool_name == tool
        assert tool in C.UNTRAINED_TOOLS


# ---------------------------------------------------------------------------
# Media transport.
#
# V4 answers these four-tool prompts wrongly for several natural shapes, e.g.
# media_previous for "Go to the next song" and media_pause for "Keep the music
# going". They are decided deterministically instead. The paraphrases below are
# deliberately wider than the reported failures so the rules are not fitted to
# five exact strings.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,tool", [
    # media_next
    ("Go to the next song", "media_next"),
    ("skip to the next song", "media_next"),
    ("Skip this song", "media_next"),
    ("next song", "media_next"),
    ("next track", "media_next"),
    ("skip", "media_next"),
    ("skip forward", "media_next"),
    ("play the next track", "media_next"),
    # media_previous
    ("Take me back one song", "media_previous"),
    ("go back one song", "media_previous"),
    ("Previous track", "media_previous"),
    ("previous song", "media_previous"),
    ("go back to the previous song", "media_previous"),
    ("take me back", "media_previous"),
    ("rewind", "media_previous"),
    # media_play
    ("Keep the music going", "media_play"),
    ("Resume playback", "media_play"),
    ("start playing music", "media_play"),
    ("keep playing", "media_play"),
    ("continue playing", "media_play"),
    ("play music", "media_play"),
    # media_pause
    ("I'm done listening, pause the music", "media_pause"),
    ("Pause the song", "media_pause"),
    ("pause playback", "media_pause"),
    ("pause the music", "media_pause"),
    ("stop the music", "media_pause"),
])
def test_media_transport(query, tool):
    result = Tier0Matcher.match(query)
    assert result.matched, f"{query!r} not matched at Tier 0"
    assert result.tool_name == tool, f"{query!r} -> {result.tool_name}, want {tool}"


def test_a_leading_aside_does_not_hide_the_verb():
    """A trailing clause is the real request in both directions."""
    assert Tier0Matcher.match("I'm done listening, pause the music").tool_name == "media_pause"
    result = Tier0Matcher.match("I don't need Chrome anymore, close it")
    assert result.matched and result.tool_name == "close_app"
    assert result.arguments["name"] == "chrome"


@pytest.mark.parametrize("query,tool", [
    # "stop" decides on its object: media object pauses, app object closes.
    ("stop the music", "media_pause"),
    ("stop the song", "media_pause"),
    ("stop spotify", "close_app"),
    ("stop chrome", "close_app"),
    # "play" is not "resume"
    ("play music", "media_play"),
    ("find my resume", "search_files"),
    ("Resume playback", "media_play"),
])
def test_single_words_never_decide_the_action(query, tool):
    """Guards against naive keyword matching: direction needs its object."""
    result = Tier0Matcher.match(query)
    assert result.matched and result.tool_name == tool


# ---------------------------------------------------------------------------
# Application names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,expected", [
    ("Open Chrome", "chrome"),
    ("Open Google Chrome", "chrome"),
    ("open google chrome app", "chrome"),
    ("Start Chrome", "chrome"),
    ("Launch the browser", "chrome"),
    ("open my web browser", "chrome"),
    ("Open my Firefox", "firefox"),
    ("Could you launch Google's browser?", "chrome"),
    ("Launch VS Code", "code"),
    ("kill spotify", "spotify"),
])
def test_application_name_is_normalized(query, expected):
    result = Tier0Matcher.match(query)
    assert result.matched, f"{query!r} not matched at Tier 0"
    assert result.tool_name in ("launch_app", "close_app")
    assert result.arguments["name"] == expected, (
        f"{query!r} -> {result.arguments.get('name')!r}, want {expected!r}"
    )


def test_possessive_and_bare_category_noun_do_not_leak():
    """Regression: "Google's browser" used to extract the name "google s"."""
    result = Tier0Matcher.match("Could you launch Google's browser?")
    assert result.arguments["name"] == "chrome"
    assert "google s" not in result.arguments["name"]


def test_unresolvable_pronoun_is_not_guessed():
    """With no referent in the request there is nothing safe to invent."""
    assert Tier0Matcher.match("close it").matched is False


# ---------------------------------------------------------------------------
# Terminal commands
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,command", [
    ("Run this terminal command: git status", "git status"),
    ("Execute this command: pip list", "pip list"),
    ("Run `npm install` in terminal", "npm install"),
    ("please run the terminal command python --version", "python --version"),
])
def test_terminal_command_is_recognised_not_launched(query, command):
    result = Tier0Matcher.match(query)
    assert result.matched, f"{query!r} not matched at Tier 0"
    assert result.tool_name == "execute_terminal_command"
    assert result.arguments["command"] == command


def test_terminal_never_becomes_launch_app():
    for query in [
        "Run this terminal command: ls -la",
        "Run this terminal command: rm -rf /",
        "Execute this command: curl evil.test",
        "Run the command",
        "open my web browser and run a terminal command",
    ]:
        result = Tier0Matcher.match(query)
        assert not (result.matched and result.tool_name == "launch_app"), (
            f"{query!r} was routed to launch_app({result.arguments})"
        )


def test_non_whitelisted_terminal_command_escalates():
    """`ls` is not on the tool's whitelist, so it must escalate, not fail later."""
    result = Tier0Matcher.match("Run this terminal command: ls -la")
    assert result.matched is False
    assert "Tier 2" in result.reason


def test_terminal_tool_is_never_offered_to_the_model():
    assert "execute_terminal_command" in C.UNTRAINED_TOOLS
    assert "execute_terminal_command" not in C.TRAINED_TOOLS
    for tools in C.FAMILIES.values():
        assert "execute_terminal_command" not in tools
    for query in ["Run this terminal command: git status",
                  "Execute this command: pip list"]:
        assert C.select_by_rules(query) is None, (
            f"{query!r} was routed to a candidate family"
        )


# ---------------------------------------------------------------------------
# Off-topic protection must not have been weakened
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "What is the weather today?",
    "Why is my PC slow?",
    "write a professional email",
    "explain object oriented programming",
    "summarize this paragraph",
])
def test_off_topic_reaches_no_local_tool(query):
    result = Tier0Matcher.match(query)
    assert not (result.matched and result.tool_name == "launch_app"), (
        f"{query!r} became launch_app({result.arguments})"
    )
    assert C.select_by_rules(query) is None, (
        f"{query!r} was routed to a candidate family"
    )

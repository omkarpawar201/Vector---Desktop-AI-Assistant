"""Deterministic fixes: argument capture, rule precedence, router vocabulary.

These are the failures the fresh evaluation classified as *category B* -- a
defect in a regex or a keyword table rather than in V4.  They are asserted
directly here so they cannot regress silently behind a green 104-row gate, which
never contained any of them.
"""

import pytest

from app.core import candidates as C
from app.core.matcher import Tier0Matcher


# ---------------------------------------------------------------------------
# Tier 0 argument capture
# ---------------------------------------------------------------------------
# One root cause across all four: the captured value kept the discourse
# scaffolding around it, so the call reached an OS that cannot resolve it.

def test_app_argument_drops_trailing_politeness():
    """`launch_app {"name": "notepad for me"}` fails at the OS for any phrasing."""
    res = Tier0Matcher.match("fire up notepad for me")
    assert res.matched and res.tool_name == "launch_app"
    assert res.arguments == {"name": "notepad"}

    res = Tier0Matcher.match("close firefox for me")
    assert res.matched and res.tool_name == "close_app"
    assert res.arguments == {"name": "firefox"}


def test_app_argument_skips_the_verb_particle():
    """The `up` belongs to "open up", never to the application's name."""
    res = Tier0Matcher.match("open up the file explorer")
    assert res.matched and res.tool_name == "launch_app"
    assert res.arguments == {"name": "explorer"}


def test_an_introducer_is_not_part_of_the_path():
    """"called" announces a name; it is not one of the folder's path segments."""
    res = Tier0Matcher.match("open the folder called Documents")
    assert res.matched and res.tool_name == "open_folder"
    assert res.arguments == {"path": "documents"}


def test_a_plain_folder_name_still_resolves():
    """The introducer strip must not eat a name that has no introducer."""
    res = Tier0Matcher.match("open the folder Documents")
    assert res.matched and res.tool_name == "open_folder"
    assert res.arguments == {"path": "documents"}


# ---------------------------------------------------------------------------
# Rule precedence and media nouns
# ---------------------------------------------------------------------------

def test_kill_the_sound_is_mute_not_an_app_close():
    """`close_app {"name": "sound"}` closes whatever happens to match "sound"."""
    res = Tier0Matcher.match("kill the sound")
    assert res.matched and res.tool_name == "mute"
    assert res.arguments == {}


def test_media_nouns_never_reach_the_app_rules():
    """Defense in depth: even without the exact pattern, no app is named."""
    for query in ("kill the sound", "stop the sound", "end the sound"):
        res = Tier0Matcher.match(query)
        assert res.tool_name != "close_app", (
            f"{query!r} was read as closing an application")


def test_search_place_for_target_is_a_file_search():
    """The `disk` keyword must not answer a resource question nobody asked."""
    res = Tier0Matcher.match("search the disk for budget")
    assert res.matched and res.tool_name == "search_files"
    assert res.arguments == {"query": "budget"}


def test_resource_queries_still_answer_as_resources():
    """The precedence guard is narrow: a real resource query is unaffected."""
    for query, tool in (("disk space", "get_disk_usage"),
                        ("ram usage", "get_memory_usage"),
                        ("cpu usage", "get_cpu_usage")):
        res = Tier0Matcher.match(query)
        assert res.tool_name == tool, f"{query!r} -> {res.tool_name}"


def test_look_for_is_not_re_captured_as_place_and_target():
    """`look for X` has no place, so the new pattern must leave it alone."""
    res = Tier0Matcher.match("look for the quarterly report")
    assert res.matched and res.tool_name == "search_files"
    assert res.arguments == {"query": "quarterly report"}


# ---------------------------------------------------------------------------
# Router vocabulary
# ---------------------------------------------------------------------------
# Family scoring is a sum of matched keywords against MIN_FAMILY_SCORE, so a
# synonym the table lacks produces an escalation rather than an answer.

@pytest.mark.parametrize("query,family", [
    # disk/storage synonyms
    ("how many gigs are left on the drive", "system_monitoring"),
    ("is the c drive full", "system_monitoring"),
    ("how full is my hard drive", "system_monitoring"),
    ("report filesystem capacity", "system_monitoring"),
    # battery vocabulary
    ("whats the power situation", "system_monitoring"),
    ("am i running low on power", "system_monitoring"),
    # launch/close verbs
    ("bring up the calculator", "app_management"),
    ("get rid of chrome", "app_management"),
    ("end spotify", "app_management"),
    # search verbs
    ("where did i put my tax pdf", "app_management"),
    ("hunt down the meeting notes", "app_management"),
    ("can you dig up the presentation", "app_management"),
])
def test_router_recovers_synonyms_it_previously_escalated(query, family):
    picked = C.select_by_rules(query)
    assert picked is not None, f"{query!r} escalated: the router lacks this synonym"
    assert picked.family == family


def test_new_keywords_do_not_route_off_topic():
    """Adding vocabulary must not turn the off-topic negatives into requests."""
    negatives = ["write a Java program", "write a professional email",
                 "generate a poem", "what is the weather like",
                 "explain object oriented programming"]
    routed = [q for q in negatives if C.select_by_rules(q) is not None]
    assert routed == []


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------
# A family score built purely from the noun clears MIN_FAMILY_SCORE and hands
# the query to V4, which has no notion of declining and answers with whatever
# tool is in the set.  Measured: `freeze the audio` -> unmute, `open it` ->
# launch_app on an arbitrary name.  Abstaining beats inventing an action.

@pytest.mark.parametrize("query", [
    "freeze the audio",
    "the music is too loud can you fix it",
    "carry on from where we left the music",
    "open it",
    "make it quieter",
    "can you handle that",
    "fix it",
    "same as before",
])
def test_vague_requests_escalate_instead_of_acting(query):
    assert C.select_by_rules(query) is None, (
        f"{query!r} was routed to a tool family: an unspecified action must "
        f"reach Tier 2 rather than be guessed at")


def test_a_vague_request_does_not_silence_a_real_one():
    """The guards key on the *absence* of an action verb, not on the noun."""
    assert C.select_by_rules("freeze the volume") is None
    assert C.select_by_rules("open notepad") is not None
    assert C.select_by_rules("kill spotify") is not None
    assert C.select_by_rules("search the disk for budget") is not None

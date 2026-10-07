"""
Candidate-tool routing for Vector.

Why this module exists
----------------------
Needle 3 is a generative tool-calling model, not a classifier. Measured on
``models/needle3_vector_v4.cact``:

* The fine-tune only ever presented 4-5 tools per example (four fixed groups,
  464 train rows).  A 17-tool prompt is out-of-distribution for it.
* V4 has no abstention.  Handed only the six system tools, it answered *every*
  query -- "play music", "open chrome", "close chrome" -- with ``get_volume``.
  It always returns some tool from the candidate set, whether or not the right
  one is there.

So Vector must never hand V4 a wide tool list, and must never trust it to say
"not applicable".  This module picks a small candidate set first; the client
then runs V4 inside it and the router discards any call that falls outside.

The candidate sets below are deliberately identical to the four tool groups
used in ``data/vector_train.jsonl``.  Re-grouping them by conceptual capability
(``CapabilityRegistry``) would put V4 back out-of-distribution, so the grouping
here is a property of the model, not of the product.

The 14 tools that were never in training (lock/sleep/restart/shutdown, the six
window tools, open_file/open_folder/get_file_info, execute_terminal_command)
are deliberately absent.  They are reachable only by deterministic Tier 0
rules, behind the permission gate.  ``UNTRAINED_TOOLS`` exists so the router
can assert that no model-derived path ever names one.
"""

from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Candidate sets -- must mirror the training groups exactly.
# --------------------------------------------------------------------------

SYSTEM_MONITORING: Tuple[str, ...] = (
    "get_system_stats",
    "get_cpu_usage",
    "get_memory_usage",
    "get_disk_usage",
    "get_battery_status",
)
VOLUME_CONTROL: Tuple[str, ...] = (
    "get_volume",
    "set_volume",
    "mute",
    "unmute",
)
MEDIA_CONTROL: Tuple[str, ...] = (
    "media_play",
    "media_pause",
    "media_next",
    "media_previous",
)
APP_MANAGEMENT: Tuple[str, ...] = (
    "launch_app",
    "close_app",
    "get_running_apps",
    "search_files",
)

FAMILIES: Dict[str, Tuple[str, ...]] = {
    "system_monitoring": SYSTEM_MONITORING,
    "volume_control": VOLUME_CONTROL,
    "media_control": MEDIA_CONTROL,
    "app_management": APP_MANAGEMENT,
}

#: Every tool V4 is allowed to be offered. Anything outside this set must never
#: reach the model.
TRAINED_TOOLS: frozenset = frozenset(
    tool for group in FAMILIES.values() for tool in group
)

#: Tools with no training coverage. Model output naming any of these is a bug,
#: and the router treats it as a hard rejection.
UNTRAINED_TOOLS: frozenset = frozenset(
    {
        "lock_pc",
        "sleep_pc",
        "restart_pc",
        "shutdown_pc",
        "get_active_window",
        "maximize_window",
        "minimize_window",
        "close_window",
        "move_window",
        "resize_window",
        "open_file",
        "open_folder",
        "get_file_info",
        "execute_terminal_command",
    }
)

#: Hard ceiling on how many tools V4 ever sees.
MAX_CANDIDATES = 6


# --------------------------------------------------------------------------
# Rule scoring
# --------------------------------------------------------------------------
# Weights are small integers; the family with the highest total wins. Ties fall
# to ``_TIE_ORDER`` so behaviour is deterministic.

_KEYWORDS: Dict[str, List[Tuple[str, int]]] = {
    "system_monitoring": [
        (r"\bcpu\b", 3), (r"\bprocessor\b", 3), (r"\bcore[s]? usage\b", 3),
        (r"\bram\b", 3), (r"\bmemory\b", 3),
        (r"\bdisk\b", 3), (r"\bstorage\b", 2), (r"\bspace\b", 2),
        (r"\bdrive\b", 3), (r"\bgigs?\b", 2), (r"\bfilesystem\b", 3),
        (r"\bcapacity\b", 2),
        (r"\bbattery\b", 3), (r"\bcharg", 3), (r"\bpower\b", 3),
        (r"\bjuice\b", 2),
        (r"\bsystem\b", 2), (r"\bstats\b", 3), (r"\bmachine\b", 2),
        (r"\boverview\b", 2), (r"\bsummary\b", 2), (r"\bhow much\b", 2),
        (r"\bhow many\b", 1), (r"\blaptop\b", 1), (r"\bpercent", 2),
    ],
    "volume_control": [
        (r"\bvolume\b", 4), (r"\bsound\b", 3), (r"\baudio\b", 3),
        (r"\bmute\b", 4), (r"\bunmute\b", 4), (r"\bmuted\b", 3),
        (r"\bquiet\b", 2), (r"\bloud\b", 2), (r"\bspeaker", 3),
        (r"\bsilence\b", 3), (r"\bback on\b", 3), (r"\benable\b", 1),
    ],
    "media_control": [
        (r"\bplay\b", 3), (r"\bplaying\b", 3), (r"\bpause\b", 4),
        (r"\bpaused\b", 3), (r"\bsong\b", 3), (r"\btrack\b", 3),
        (r"\bmusic\b", 3), (r"\bskip\b", 4), (r"\bnext\b", 3),
        (r"\bprevious\b", 3), (r"\bprev\b", 2), (r"\bresume\b", 2),
        (r"\bplayback\b", 4), (r"\bplaylist\b", 3), (r"\bmedia\b", 2),
    ],
    "app_management": [
        (r"\bopen\b", 2), (r"\blaunch\b", 4), (r"\bstart\b", 2),
        (r"\bclose\b", 2), (r"\bquit\b", 3), (r"\bexit\b", 3),
        (r"\bkill\b", 3), (r"\bterminate\b", 3), (r"\brunning\b", 3),
        (r"\bbring\s+(?:up|out)\b", 3), (r"\bget\s+rid\s+of\b", 3),
        (r"\bend\b", 3),
        (r"\bapplications?\b", 2), (r"\bprograms?\b", 1), (r"\bapps?\b", 2),
        (r"\bbrowser\b", 2), (r"\blist\b", 1), (r"\bprocess", 2),
        (r"\bfind\b", 3), (r"\bsearch\b", 3), (r"\blocate\b", 3),
        (r"\bwhere is\b", 3), (r"\blook for\b", 3),
        (r"\bwhere\s+(?:did|do|have)\b", 3), (r"\bhunt\b", 3),
        (r"\bdig\s+up\b", 3),
        (r"\bfile\b", 1), (r"\bfolder\b", 1), (r"\bdocuments?\b", 1),
    ],
}

_COMPILED: Dict[str, List[Tuple[re.Pattern, int]]] = {
    family: [(re.compile(p, re.IGNORECASE), w) for p, w in pats]
    for family, pats in _KEYWORDS.items()
}

_TIE_ORDER = ("system_monitoring", "volume_control", "media_control",
              "app_management")

#: Minimum score for a rule hit to be trusted.
MIN_FAMILY_SCORE = 2

# "resume" is genuinely ambiguous: "resume playback" is media, "find my resume"
# is a file. Needle gets this wrong (it answered media_previous for
# "Can you find my resume?"), so the disambiguation has to be deterministic.
_FILE_VERB = re.compile(
    r"\b(find|search|locate|look\s+for|where\s+is|open)\b", re.IGNORECASE)
_MEDIA_VERB = re.compile(
    r"\b(play|playing|playback|resume|continue|restart|start)\b", re.IGNORECASE)

# Intents that exist in the registry but have no training coverage. Routing
# these to V4 would force it to pick something from an unrelated set, so they
# return None and fall through to Tier 0 rules / Tier 2.
_UNCOVERED_INTENT = re.compile(
    r"\b(open|launch|start)\b[^?]*\b(folder|directory|file|document|pdf|"
    r"spreadsheet|presentation)\b"
    r"|\bfile info\b|\bmetadata\b"
    r"|\b(maximi[sz]e|minimi[sz]e|resize|move|focus|switch to)\b[^?]*\bwindow\b"
    r"|\b(active|foreground|current)\s+window\b"
    r"|\b(lock|sleep|restart|reboot|shutdown|power off|log ?off)\b"
    r"|\b(terminal|shell|command line|console|run)\b\s+(a\s+)?(command|cmd)?",
    re.IGNORECASE,
)

# Vague requests that name no operation the registry could actually perform.
# Family scoring works off the noun alone -- "audio", "music" -- so these clear
# ``MIN_FAMILY_SCORE`` and reach V4, which has no notion of abstaining and
# answers with whatever tool is in the candidate set.  Measured outcomes:
# "freeze the audio" -> ``unmute``, "the music is too loud can you fix it" ->
# ``media_pause``, "open it" -> ``launch_app`` on an arbitrary name.  None is
# dangerous, but all three invent an action the user never asked for, so the
# router returns None and the request reaches Tier 2 instead.
#
# Verified against both held-out sets: these four patterns fire on 0 of the 114
# labelled rows, so the gate is untouched by them.

#: Any verb the registry actually implements. Its absence is what makes a
#: request unanswerable rather than merely unspecific.
_ACTION_VERB = re.compile(
    r"\b(open|launch|start|run|close|quit|exit|kill|terminate|stop|end|"
    r"play|pause|resume|continue|skip|next|previous|prev|mute|unmute|"
    r"silence|shush|turn|set|raise|lower|increase|decrease|adjust|change|"
    r"find|search|locate|look|show|display|check|report|bring|get|give|"
    r"lock|sleep|restart|reboot|shutdown|power|maximi[sz]e|minimi[sz]e|"
    r"free|hunt|dig)\b",
    re.IGNORECASE,
)

#: A plea or a transport-like word with no implementation behind it.
_NO_OPERATION = re.compile(
    r"\b(fix(?:es|ed|ing)?|handle[ds]?|handling|sort(?:ed)?\s+out|"
    r"do\s+something|make\s+it|freeze|froze|frozen|halt)\b",
    re.IGNORECASE,
)

#: A request that points back at an earlier one. Routing is single-turn, so
#: there is no earlier turn to point at and the antecedent can never resolve.
_SESSION_REFERENCE = re.compile(
    r"\b(?:from\s+)?where\s+we\s+(?:left|stopped|were)\b"
    r"|\bwhere\s+we\s+left\s+off\b"
    r"|\b(?:same|just|exactly)\s+as\s+before\b"
    r"|\blike\s+before\b"
    r"|\bcarry\s+on\s+from\b",
    re.IGNORECASE,
)

#: "open it" with nothing in this request to open. ``_normalize_app_name``
#: resolves pronouns against the wider request and otherwise falls back to
#: Chrome, turning an anaphor into an arbitrary launch.
_DANGLING_PRONOUN = re.compile(
    r"^(?:please\s+|can you\s+|could you\s+|would you\s+)?"
    r"(?:open|launch|start|close|quit|exit|kill|terminate|stop|play|pause|"
    r"resume|find|search|locate|bring|give|hand|pass|show)\s+"
    r"(?:it|that|this|them|those|these|one|everything)\s*[.?!]?$",
    re.IGNORECASE,
)


@dataclass
class CandidateSet:
    """The tools V4 is allowed to choose between for one query."""

    family: str
    tools: Tuple[str, ...]
    scores: Dict[str, int] = field(default_factory=dict)
    source: str = "rules"

    @property
    def names(self) -> frozenset:
        return frozenset(self.tools)

    def allows(self, tool_name: str) -> bool:
        """Whether a model-produced call is admissible.

        Rejects untrained tools outright, and anything outside this set -- which
        is V4's forced-choice failure mode, not a valid answer.
        """
        if not tool_name:
            return False
        if tool_name in UNTRAINED_TOOLS:
            return False
        return tool_name in self.names

    def reject_reason(self, tool_name: str) -> str:
        if not tool_name:
            return "model produced no tool name"
        if tool_name in UNTRAINED_TOOLS:
            return f"model named untrained tool {tool_name!r}"
        if tool_name not in self.names:
            return f"model named {tool_name!r}, outside candidate set {sorted(self.names)}"
        return ""


def _score_families(query: str) -> Dict[str, int]:
    scores = {family: 0 for family in _TIE_ORDER}
    for family, patterns in _COMPILED.items():
        for pattern, weight in patterns:
            if pattern.search(query):
                scores[family] += weight
    return scores


def _apply_disambiguation(query: str, scores: Dict[str, int]) -> Dict[str, int]:
    """Resolve collisions the flat keyword scoring cannot see on its own."""
    if not re.search(r"\bresume\b", query, re.IGNORECASE):
        return scores

    file_side = bool(_FILE_VERB.search(query))
    media_side = bool(_MEDIA_VERB.search(query))

    if file_side and not media_side:
        # "find my resume", "where is my resume pdf"
        scores["app_management"] += 6
        scores["media_control"] -= 6
    elif media_side and not file_side:
        scores["media_control"] += 4
        scores["app_management"] -= 4
    # Both present ("resume the file and play it") is genuinely unclear; leave
    # the flat scores alone and let the confidence threshold reject it.
    return scores


def score_families(query: str) -> Dict[str, int]:
    """Rule scores per family. Exposed for tests and for the eval script."""
    if not query or not query.strip():
        return {family: 0 for family in _TIE_ORDER}
    return _apply_disambiguation(query.strip(), _score_families(query))


def select_by_rules(query: str) -> Optional[CandidateSet]:
    """Deterministic family pick. No model involved."""
    if not query or not query.strip():
        return None
    query = query.strip()
    if _UNCOVERED_INTENT.search(query):
        return None
    if _SESSION_REFERENCE.search(query) or _DANGLING_PRONOUN.match(query):
        return None
    if _NO_OPERATION.search(query) and not _ACTION_VERB.search(query):
        return None

    scores = score_families(query)
    best = max(_TIE_ORDER, key=lambda f: (scores[f], -_TIE_ORDER.index(f)))
    if scores[best] < MIN_FAMILY_SCORE:
        return None
    return CandidateSet(family=best, tools=FAMILIES[best], scores=scores)


# --------------------------------------------------------------------------
# Embedding-based fallback
# --------------------------------------------------------------------------
# Needle 3 exposes a 128-d embedding (``EmbeddingHead``). If the loaded archive
# supports it, we can embed the query and each tool description and pick a
# family by cosine similarity. This is the long-tail path for phrasings no
# keyword rule anticipates.
#
# It is deliberately defensive: if embeddings are unavailable, fail to None and
# let the router escalate to Tier 2. Long-tail recall is worth having; it is
# never worth breaking the deterministic path over.

_EMBED_LOCK = threading.Lock()
_EMBED_AGENT = None
_EMBED_VECTORS: Optional[Dict[str, List[float]]] = None
_EMBED_FAILED = False

#: Below this cosine similarity we would rather escalate than guess.
EMBED_MIN_SIMILARITY = 0.55


def _tool_vectors(tools: Sequence[dict]) -> Optional[Dict[str, List[float]]]:
    """Embed every tool description once. Returns None if unavailable."""
    global _EMBED_VECTORS, _EMBED_FAILED
    if _EMBED_VECTORS is not None or _EMBED_FAILED:
        return _EMBED_VECTORS

    with _EMBED_LOCK:
        if _EMBED_VECTORS is not None or _EMBED_FAILED:
            return _EMBED_VECTORS
        agent = _get_embed_agent()
        if agent is None:
            _EMBED_FAILED = True
            return None
        vectors = {}
        try:
            for tool in tools:
                text = f"{tool.get('name', '')}: {tool.get('description', '')}".strip()
                vectors[tool["name"]] = list(agent.embed(text))
        except Exception:
            _EMBED_FAILED = True
            return None
        if not vectors:
            _EMBED_FAILED = True
            return None
        _EMBED_VECTORS = vectors
        return _EMBED_VECTORS


def _get_embed_agent():
    global _EMBED_AGENT
    if _EMBED_AGENT is not None or _EMBED_FAILED:
        return _EMBED_AGENT
    try:
        from needle import Needle
    except Exception:
        return None
    try:
        _EMBED_AGENT = Needle(tools=[], generation=3)
    except Exception:
        _EMBED_AGENT = None
    return _EMBED_AGENT


def reset_embedding_cache() -> None:
    """Drop cached tool vectors and the embed agent. Used by tests."""
    global _EMBED_AGENT, _EMBED_VECTORS, _EMBED_FAILED
    with _EMBED_LOCK:
        if _EMBED_AGENT is not None:
            try:
                _EMBED_AGENT.close()
            except Exception:
                pass
        _EMBED_AGENT = None
        _EMBED_VECTORS = None
        _EMBED_FAILED = False


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def select_by_embedding(query: str, tools: Sequence[dict],
                        min_similarity: float = EMBED_MIN_SIMILARITY
                        ) -> Optional[CandidateSet]:
    """Family pick by cosine similarity over Needle embeddings."""
    if not query or not query.strip() or not tools:
        return None
    vectors = _tool_vectors(tools)
    if not vectors:
        return None
    agent = _get_embed_agent()
    if agent is None:
        return None
    try:
        query_vector = list(agent.embed(query.strip()))
    except Exception:
        return None

    allowed = {t["name"]: t for t in tools if t.get("name") in TRAINED_TOOLS}
    if not allowed:
        return None

    per_family: Dict[str, float] = {}
    for family, members in FAMILIES.items():
        present = [m for m in members if m in vectors and m in allowed]
        if not present:
            continue
        per_family[family] = max(cosine(query_vector, vectors[m]) for m in present)

    if not per_family:
        return None
    best = max(per_family, key=lambda f: (per_family[f], -_TIE_ORDER.index(f)))
    if per_family[best] < min_similarity:
        return None
    return CandidateSet(
        family=best,
        tools=FAMILIES[best],
        scores={f: round(s, 4) for f, s in per_family.items()},
        source="embedding",
    )


def select(query: str, tools: Sequence[dict] = (),
           min_similarity: float = EMBED_MIN_SIMILARITY,
           use_embedding: bool = False) -> Optional[CandidateSet]:
    """Pick the candidate set for ``query``.

    Deterministic rules only, by default. Returns None when nothing matches,
    which the router treats as "escalate", never as "pick something".

    ``use_embedding`` re-enables the cosine-similarity fallback. It is off by
    default because it was measured to be unsafe, not merely weak: on the 114
    held-out rows, off-topic queries scored 0.929-0.943 cosine against the tool
    descriptions while correctly-routed queries scored 0.909-0.978. The two
    ranges overlap almost completely, so no threshold separates them, and the
    old 0.55 floor sat far below both -- "what is the weather like" came back as
    ``system_monitoring`` at 0.938. Family accuracy was 88/104 (84.6%) against
    104/104 for the rules. Rules-only it is; the fallback stays available for
    diagnostics and future re-measurement.
    """
    picked = select_by_rules(query)
    if picked is not None:
        return picked
    if use_embedding and tools:
        return select_by_embedding(query, tools, min_similarity=min_similarity)
    return None


def select_schemas(query: str, schemas: Sequence[dict],
                   min_similarity: float = EMBED_MIN_SIMILARITY,
                   use_embedding: bool = False) -> Optional[CandidateSet]:
    """``select`` against a full 31-tool schema list, filtered to trained tools.

    The untrained 14 are stripped before embedding so their descriptions can
    never pull the model toward a tool it must not name.
    """
    trained = [t for t in schemas if t.get("name") in TRAINED_TOOLS]
    return select(query, tools=trained, min_similarity=min_similarity,
                  use_embedding=use_embedding)


__all__ = [
    "CandidateSet",
    "FAMILIES",
    "TRAINED_TOOLS",
    "UNTRAINED_TOOLS",
    "MAX_CANDIDATES",
    "score_families",
    "select",
    "select_by_rules",
    "select_by_embedding",
    "select_schemas",
    "cosine",
    "reset_embedding_cache",
]

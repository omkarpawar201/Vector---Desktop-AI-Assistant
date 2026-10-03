"""
Tier 0 Ultra-Fast Direct Matcher for Vector Desktop AI Assistant.
Provides reflexes (<5-10ms latency) for deterministic system, volume, media, and power commands.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Optional, Tuple
from app.core.normalizer import InputNormalizer


@dataclass
class MatchResult:
    """
    Result returned by the Tier 0 Matcher.
    """
    matched: bool
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""


class Tier0Matcher:
    """
    Sub-10ms direct pattern and regex matcher for deterministic desktop commands.
    Bypasses LLM inference completely for reflexes (volume, media, stats, power).
    """

    # Direct exact alias mappings (Normalized Input -> (Tool Name, Args))
    EXACT_PATTERNS: Dict[str, Tuple[str, Dict[str, Any]]] = {
        # Volume
        "mute": ("mute", {}),
        "mute volume": ("mute", {}),
        "mute sound": ("mute", {}),
        "mute audio": ("mute", {}),
        "silence audio": ("mute", {}),
        "silence sound": ("mute", {}),
        "stop the sound": ("mute", {}),
        "unmute": ("unmute", {}),
        "unmute volume": ("unmute", {}),
        "unmute sound": ("unmute", {}),
        "enable sound": ("unmute", {}),
        "turn the sound on": ("unmute", {}),
        "turn the speakers back on": ("unmute", {}),
        "enable audio": ("unmute", {}),
        "enable audio again": ("unmute", {}),
        "turn the volume on": ("unmute", {}),
        "turn the sound back on": ("unmute", {}),
        "disable sound": ("mute", {}),
        "disable audio": ("mute", {}),
        "turn the sound off": ("mute", {}),
        "turn the volume off": ("mute", {}),
        "get volume": ("get_volume", {}),
        "what is the volume": ("get_volume", {}),
        "current volume": ("get_volume", {}),
        "am i muted": ("get_volume", {}),
        "is it muted": ("get_volume", {}),
        "is the sound muted": ("get_volume", {}),

        # Media
        "play": ("media_play", {}),
        "play music": ("media_play", {}),
        "play song": ("media_play", {}),
        "play track": ("media_play", {}),
        "resume": ("media_play", {}),
        "resume music": ("media_play", {}),
        "resume playback": ("media_play", {}),
        "resume playing": ("media_play", {}),
        "continue": ("media_play", {}),
        "continue playing": ("media_play", {}),
        "keep playing": ("media_play", {}),
        "unpause": ("media_play", {}),
        "unpause music": ("media_play", {}),
        "pause": ("media_pause", {}),
        "pause music": ("media_pause", {}),
        "pause song": ("media_pause", {}),
        "pause playback": ("media_pause", {}),
        "stop music": ("media_pause", {}),
        "next": ("media_next", {}),
        "next song": ("media_next", {}),
        "next track": ("media_next", {}),
        "skip": ("media_next", {}),
        "skip song": ("media_next", {}),
        "skip this song": ("media_next", {}),
        "skip the song": ("media_next", {}),
        "skip this track": ("media_next", {}),
        "skip the track": ("media_next", {}),
        "skip to next song": ("media_next", {}),
        "go forward": ("media_next", {}),
        "previous": ("media_previous", {}),
        "previous song": ("media_previous", {}),
        "previous track": ("media_previous", {}),
        "prev song": ("media_previous", {}),
        "go back": ("media_previous", {}),
        "go back a track": ("media_previous", {}),
        "play previous": ("media_previous", {}),
        "last song": ("media_previous", {}),
        "last track": ("media_previous", {}),

        # System Stats
        "system stats": ("get_system_stats", {}),
        "system summary": ("get_system_stats", {}),
        "cpu": ("get_cpu_usage", {}),
        "cpu usage": ("get_cpu_usage", {}),
        "ram": ("get_memory_usage", {}),
        "ram usage": ("get_memory_usage", {}),
        "memory usage": ("get_memory_usage", {}),
        "disk": ("get_disk_usage", {}),
        "disk space": ("get_disk_usage", {}),
        "battery": ("get_battery_status", {}),
        "battery status": ("get_battery_status", {}),
        "battery level": ("get_battery_status", {}),

        # Power
        "lock pc": ("lock_pc", {}),
        "lock computer": ("lock_pc", {}),
        "lock screen": ("lock_pc", {}),
        "sleep pc": ("sleep_pc", {}),
        "sleep computer": ("sleep_pc", {}),
        "restart pc": ("restart_pc", {}),
        "restart computer": ("restart_pc", {}),
        "shutdown pc": ("shutdown_pc", {}),
        "shutdown computer": ("shutdown_pc", {}),

        # Active Window
        "active window": ("get_active_window", {}),
        "maximize window": ("maximize_window", {}),
        "minimize window": ("minimize_window", {}),
        "close window": ("close_window", {}),
    }

    # Volume extraction regex patterns (e.g., "set volume to 40", "volume 40%", "set sound 50")
    SET_VOLUME_REGEXES = [
        re.compile(r"^(?:set|make|change)?\s*(?:the)?\s*(?:volume|sound)\s*(?:to|at)?\s*(\d{1,3})\s*%?$"),
        re.compile(r"^(?:volume|sound)\s*(\d{1,3})\s*%?$"),
    ]

    #: Directional phrasings that still name an absolute level, e.g. "volume up
    #: to 50" is a set, not a report.
    SET_VOLUME_VERB_REGEXES = [
        re.compile(r"^(?:volume|sound)\s*(?:up|down)\s*(?:to)?\s*(\d{1,3})\s*%?$"),
        re.compile(r"^(?:turn|raise|lower|put|bump|crank)\s*(?:it|the\s+(?:volume|sound))?\s*"
                   r"(?:up|down)?\s*(?:to)?\s*(\d{1,3})\s*%?$"),
    ]

    #: Resource keyword -> the single tool that owns it. Ordered so the overview
    #: patterns win over the individual resources.
    SYSTEM_RESOURCE_REGEXES = [
        (re.compile(r"\b(?:system\s+(?:stats|overview|information|info|summary)|"
                    r"machine\s+stats|overall\s+stats|system\s+summary)\b"), "get_system_stats"),
        (re.compile(r"\b(?:battery|charg(?:e|es|ed|ing)|power\s+level)\b"), "get_battery_status"),
        (re.compile(r"\b(?:ram|memory)\b"), "get_memory_usage"),
        (re.compile(r"\bcpu\b|\bprocessor\b"), "get_cpu_usage"),
        (re.compile(r"\b(?:disk|storage)\b|\bhow\s+much\s+space\b|"
                    r"\b(?:space|room)\s+(?:is\s+)?(?:available|left|free)\b"), "get_disk_usage"),
    ]

    # ------------------------------------------------------------------
    # Argument-extraction rules.
    #
    # These cover intents Needle was measured getting wrong, and the 14 tools
    # that have no fine-tune coverage at all. V4 cannot be trusted with the
    # latter: given an unrelated candidate set it answers with whatever tool is
    # present rather than declining, so those tools are only ever reached here,
    # deterministically, and then face the normal permission gate.
    # ------------------------------------------------------------------

    # File search. Deliberately first: "find my resume" must reach search_files,
    # not media_play. Needle answered media_previous for that phrasing.
    FILE_SEARCH_REGEXES = [
        re.compile(r"^(?:can you\s+|please\s+)?(?:find|search(?:\s+for)?|look\s+for|locate)\s+(?:my\s+|the\s+|a\s+|an\s+)?(.+)$"),
        re.compile(r"^where\s+is\s+(?:my\s+|the\s+)?(.+)$"),
    ]

    APP_OPEN_REGEXES = [
        re.compile(r"^(?:please\s+|can you\s+|could you\s+)?(?:open|launch|start|run|fire\s+up)\s+(?:the\s+)?(.+?)(?:\s+(?:app|application|browser|program))?$"),
    ]

    APP_CLOSE_REGEXES = [
        re.compile(r"^(?:please\s+|can you\s+|could you\s+)?(?:close|quit|exit|kill|terminate|stop)\s+(?:the\s+)?(.+?)(?:\s+(?:app|application|browser|program))?$"),
    ]

    # Media transport, for the many "go back to the previous song" shapes that
    # are too varied to enumerate. Checked before the app rules so that
    # "start playing music" is never read as launching an app called
    # "playing music".
    MEDIA_NEXT_REGEX = re.compile(
        r"^(?:skip|go\s+forward|next|forward)\s*(?:to\s+)?(?:the\s+)?"
        r"(?:next\s+)?(?:song|track|video|one|tune)?$"
        r"|^skip\s+(?:this|the)\s+(?:song|track|video)$"
        r"|^skip\s+to\s+(?:the\s+)?next\s+(?:song|track|video)$"
        r"|^skip\s+(?:forward|ahead)$"
    )
    MEDIA_PREV_REGEX = re.compile(
        r"^(?:go\s+back|previous|prev|last|back|return)\s*(?:one)?\s*(?:to\s+)?(?:the\s+)?"
        r"(?:previous|last)?\s*(?:song|track|video|one|tune)?$"
        r"|^go\s+back\s+to\s+(?:the\s+)?(?:previous\s+)?(?:song|track)$"
        r"|^play\s+(?:the\s+)?(?:previous|last)\s+(?:song|track)$"
        r"|^return\s+to\s+(?:the\s+)?(?:previous|last)\s+(?:song|track)$"
    )
    #: "start playing music" is playback, not an app called "playing music".
    MEDIA_PLAY_REGEX = re.compile(
        r"^(?:start|begin|resume|continue|keep|carry\s+on)\s*"
        r"(?:playing\s+|the\s+|back\s+)?"
        r"(?:music|song|track|playback|audio|video|media|tv|show)$"
        r"|^(?:unpause|resume)\s*$"
    )

    #: If an app-open target contains any of these it is a media command, not an
    #: application name. Without this, "start playing music" launches an app
    #: literally named "playing music".
    _MEDIA_WORDS = re.compile(
        r"\b(play|playing|playback|music|song|track|video|audio|podcast|"
        r"media|album|playlist|resume|pause)\b")

    #: Articles are dropped for exact-pattern lookup so "pause the song" and
    #: "pause song" resolve identically.
    _ARTICLES = re.compile(r"\b(?:the|my|a|an)\b")

    @classmethod
    def _drop_articles(cls, normalized: str) -> str:
        stripped = cls._ARTICLES.sub(" ", normalized)
        return re.sub(r"\s+", " ", stripped).strip()

    # Trailing noise that must not end up inside an extracted argument.
    _ARG_NOISE = re.compile(
        r"\s+(?:for me|please|now|thanks|thank you)$")

    _PRONOUN_PREFIX = re.compile(
        r"^(?:my|the|a|an|some|that|this)\s+")

    # Trailing file extensions and words that are never part of a search term.
    _SEARCH_TAIL = re.compile(
        r"\s+(?:file|files|document|documents|folder|folders|pdf|docx?|xlsx?|"
        r"pptx?|txt|on my (?:computer|pc|laptop|disk|system)|"
        r"on the (?:computer|pc|laptop|disk|system)|for me)$")

    # --- untrained tools: rules only, never model-selected ---

    POWER_REGEXES = [
        (re.compile(r"^(?:please\s+)?(?:lock)\s*(?:the\s+)?(?:pc|computer|screen|laptop|system)?$"), "lock_pc"),
        (re.compile(r"^(?:please\s+)?(?:sleep|standby|hibernate)\s*(?:the\s+)?(?:pc|computer|laptop|system)?$"), "sleep_pc"),
        (re.compile(r"^(?:please\s+)?(?:restart|reboot)\s*(?:the\s+)?(?:pc|computer|laptop|system)?$"), "restart_pc"),
        (re.compile(r"^(?:please\s+)?(?:shut\s*down|power\s+off|log\s*off|sign\s+out)\s*(?:the\s+)?(?:pc|computer|laptop|system)?$"), "shutdown_pc"),
    ]

    WINDOW_REGEXES = [
        (re.compile(r"^(?:what|which)\s+(?:is\s+)?(?:the\s+)?(?:active|current|foreground)\s+window$"), "get_active_window"),
        (re.compile(r"^(?:please\s+)?maximi[sz]e\s*(?:the\s+)?(?:current\s+|active\s+|this\s+)?window$"), "maximize_window"),
        (re.compile(r"^(?:please\s+)?minimi[sz]e\s*(?:the\s+)?(?:current\s+|active\s+|this\s+)?window$"), "minimize_window"),
        (re.compile(r"^(?:please\s+)?close\s+(?:the\s+)?(?:current\s+|active\s+|this\s+)?window$"), "close_window"),
    ]

    # Path-bearing targets for the file tools.
    PATH_LIKE = re.compile(r"[/\\]|[A-Za-z]:\\|\.(?:pdf|docx?|xlsx?|pptx?|txt|md|py|json|csv|zip)\b")

    OPEN_FOLDER_REGEXES = [
        re.compile(r"^(?:please\s+)?open\s+(?:the\s+)?(?:folder|directory)\s+(.+)$"),
    ]

    OPEN_FILE_REGEXES = [
        re.compile(r"^(?:please\s+)?open\s+(?:the\s+)?(?:file|document)?\s*(.+)$"),
    ]

    FILE_INFO_REGEXES = [
        re.compile(r"^(?:get|show)\s+(?:the\s+)?(?:file\s+)?info(?:rmation)?\s+(?:for|about|on)\s+(.+)$"),
        re.compile(r"^(?:what|how large)\s+is\s+(?:the\s+)?file\s+(.+)$"),
    ]

    @classmethod
    def _clean_arg(cls, text: str) -> str:
        """Strip filler from an extracted argument."""
        cleaned = cls._ARG_NOISE.sub("", text.strip())
        while True:
            stripped = cls._PRONOUN_PREFIX.sub("", cleaned)
            if stripped == cleaned:
                break
            cleaned = stripped
        return cleaned.strip()

    @classmethod
    def _normalize_keeping_paths(cls, text: str) -> str:
        """Lowercase and collapse whitespace, but keep path punctuation.

        ``InputNormalizer.normalize`` strips everything outside ``[\\w\\s%]``,
        which destroys ``report.pdf`` -> ``report pdf`` and ``~/Docs`` ->
        ``~ Docs``.  The path-bearing rules therefore work from this instead,
        otherwise no file path can ever survive to ``open_file``.
        """
        return re.sub(r"\s+", " ", (text or "").lower().strip()).strip("?!.,")

    @classmethod
    def _clean_search_term(cls, text: str) -> str:
        """Reduce a file-search phrase to its content words."""
        cleaned = cls._clean_arg(text)
        while True:
            stripped = cls._SEARCH_TAIL.sub("", cleaned)
            if stripped == cleaned:
                break
            cleaned = stripped
        return cleaned.strip()

    @classmethod
    def _match_untrained(cls, raw: str) -> Optional[MatchResult]:
        """Deterministic rules for tools with no fine-tune coverage.

        Returns a result only for an explicit, unambiguous command. Anything
        vaguer is left to Tier 2 rather than guessed at here.  Works from the
        path-preserving form of the input, not ``InputNormalizer`` output.
        """
        normalized = cls._normalize_keeping_paths(raw)
        if not normalized:
            return None

        for pattern, tool in cls.POWER_REGEXES:
            if pattern.match(normalized):
                return MatchResult(
                    matched=True, tool_name=tool, arguments={}, confidence=1.0,
                    reason=f"Tier 0 power rule for '{normalized}' (no model involved)",
                )

        for pattern, tool in cls.WINDOW_REGEXES:
            if pattern.match(normalized):
                return MatchResult(
                    matched=True, tool_name=tool, arguments={}, confidence=1.0,
                    reason=f"Tier 0 window rule for '{normalized}' (no model involved)",
                )

        for pattern in cls.OPEN_FOLDER_REGEXES:
            m = pattern.match(normalized)
            if m:
                target = cls._clean_arg(m.group(1))
                if target:
                    return MatchResult(
                        matched=True, tool_name="open_folder",
                        arguments={"path": target}, confidence=1.0,
                        reason=f"Tier 0 open_folder for '{target}' (no model involved)",
                    )

        for pattern in cls.FILE_INFO_REGEXES:
            m = pattern.match(normalized)
            if m:
                target = cls._clean_arg(m.group(1))
                if target:
                    return MatchResult(
                        matched=True, tool_name="get_file_info",
                        arguments={"path": target}, confidence=1.0,
                        reason=f"Tier 0 get_file_info for '{target}' (no model involved)",
                    )

        for pattern in cls.OPEN_FILE_REGEXES:
            m = pattern.match(normalized)
            if m:
                target = cls._clean_arg(m.group(1))
                # Require a real path or filename. Bare "open spotify" is an app
                # launch, which is a trained tool, not a file open.
                if target and cls.PATH_LIKE.search(target):
                    return MatchResult(
                        matched=True, tool_name="open_file",
                        arguments={"path": target}, confidence=1.0,
                        reason=f"Tier 0 open_file for '{target}' (no model involved)",
                    )

        return None

    @classmethod
    def match(cls, user_input: str) -> MatchResult:
        """
        Attempts to match raw user input against Tier 0 direct reflex patterns.
        Returns MatchResult with matched=True if direct pattern found.
        """
        if not user_input:
            return MatchResult(matched=False, reason="Empty input")

        normalized = InputNormalizer.normalize(user_input)
        dearticled = cls._drop_articles(normalized)

        # 1. Exact pattern lookup, then again with articles dropped so that
        #    "pause the song" resolves like "pause song".
        for key in (normalized, dearticled):
            if key in cls.EXACT_PATTERNS:
                tool_name, args = cls.EXACT_PATTERNS[key]
                return MatchResult(
                    matched=True,
                    tool_name=tool_name,
                    arguments=args,
                    confidence=1.0,
                    reason=f"Tier 0 exact match for '{key}'"
                )

        # 2. Regex volume extraction
        for pattern in cls.SET_VOLUME_REGEXES:
            m = pattern.match(normalized)
            if m:
                level_val = int(m.group(1))
                if 0 <= level_val <= 100:
                    return MatchResult(
                        matched=True,
                        tool_name="set_volume",
                        arguments={"level": level_val},
                        confidence=1.0,
                        reason=f"Tier 0 regex match set_volume({level_val})"
                    )

        # 2b. "volume up to 50" / "turn it up to 30" carry a number, so they are
        #     a set, not a report.
        for pattern in cls.SET_VOLUME_VERB_REGEXES:
            m = pattern.match(normalized)
            if m:
                level_val = int(m.group(1))
                if 0 <= level_val <= 100:
                    return MatchResult(
                        matched=True,
                        tool_name="set_volume",
                        arguments={"level": level_val},
                        confidence=1.0,
                        reason=f"Tier 0 regex match set_volume({level_val}) from a directional phrase"
                    )

        # 2c. Explicit resource queries. Each of these keywords maps to exactly
        #     one tool in the whole 31-tool registry, and V4 measurably collapses
        #     them all onto get_system_stats, so the deterministic answer is
        #     strictly better than letting the model choose.
        for pattern, tool in cls.SYSTEM_RESOURCE_REGEXES:
            if pattern.search(normalized):
                return MatchResult(
                    matched=True,
                    tool_name=tool,
                    arguments={},
                    confidence=1.0,
                    reason=f"Tier 0 unambiguous resource query -> {tool} (V4 collapses these onto get_system_stats)"
                )

        # 3. File search, ahead of any media handling. "find my resume" is a
        #    file, not playback -- the model cannot be relied on for this.
        for pattern in cls.FILE_SEARCH_REGEXES:
            m = pattern.match(normalized)
            if m:
                term = cls._clean_search_term(m.group(1))
                if term:
                    return MatchResult(
                        matched=True,
                        tool_name="search_files",
                        arguments={"query": term},
                        confidence=1.0,
                        reason=f"Tier 0 file search for '{term}' (resume guard: file verb wins)"
                    )

        # 4. Tools with no fine-tune coverage, rules only. Uses the raw input
        #    so file paths survive.
        untrained = cls._match_untrained(user_input)
        if untrained is not None:
            return untrained

        # 5. Media transport, before the app rules: "start playing music" must
        #    not become launch_app("playing music").
        if cls.MEDIA_PREV_REGEX.match(dearticled):
            return MatchResult(
                matched=True, tool_name="media_previous", arguments={},
                confidence=1.0,
                reason=f"Tier 0 media_previous for '{dearticled}'")
        if cls.MEDIA_NEXT_REGEX.match(dearticled):
            return MatchResult(
                matched=True, tool_name="media_next", arguments={},
                confidence=1.0,
                reason=f"Tier 0 media_next for '{dearticled}'")
        if cls.MEDIA_PLAY_REGEX.match(dearticled):
            return MatchResult(
                matched=True, tool_name="media_play", arguments={},
                confidence=1.0,
                reason=f"Tier 0 media_play for '{dearticled}'")

        # 6. App close before app open, so "close chrome" is never an open.
        for pattern in cls.APP_CLOSE_REGEXES:
            m = pattern.match(normalized)
            if m:
                name = cls._clean_arg(m.group(1))
                if name and len(name) > 1 and not cls._MEDIA_WORDS.search(name):
                    return MatchResult(
                        matched=True,
                        tool_name="close_app",
                        arguments={"name": name},
                        confidence=1.0,
                        reason=f"Tier 0 close_app for '{name}'"
                    )

        # 7. App open
        for pattern in cls.APP_OPEN_REGEXES:
            m = pattern.match(normalized)
            if m:
                name = cls._clean_arg(m.group(1))
                if name and len(name) > 1 and not cls._MEDIA_WORDS.search(name):
                    return MatchResult(
                        matched=True,
                        tool_name="launch_app",
                        arguments={"name": name},
                        confidence=1.0,
                        reason=f"Tier 0 launch_app for '{name}'"
                    )

        return MatchResult(matched=False, reason="No Tier 0 match")

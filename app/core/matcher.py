"""
Tier 0 Ultra-Fast Direct Matcher for Vector Desktop AI Assistant.
Provides reflexes (<5-10ms latency) for deterministic system, volume, media, and power commands.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple
from app.config.constants import ALLOWED_TERMINAL_COMMANDS
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
        # The target is captured greedily on purpose. An earlier non-greedy
        # capture with an optional trailing "app|browser|program" group ate
        # that word, so "Google's browser" arrived as just "google's" and the
        # normalizer could no longer tell it was a browser request.
        # _normalize_app_name strips the category noun itself.
        re.compile(r"^(?:please\s+|can you\s+|could you\s+|would you\s+)?"
                   r"(?:open|launch|start|run|fire\s+up)\s+(?:the\s+)?(.+)$"),
    ]

    APP_CLOSE_REGEXES = [
        re.compile(r"^(?:please\s+|can you\s+|could you\s+|would you\s+)?"
                   r"(?:close|quit|exit|kill|terminate|stop)\s+(?:the\s+)?(.+)$"),
    ]

    # ------------------------------------------------------------------
    # Terminal commands.
    #
    # ``execute_terminal_command`` is one of the 14 tools the fine-tune never
    # saw, so it is rules-only and must never be sent to V4. Every pattern
    # demands explicit shell/command wording, because a bare "run" is ambiguous
    # with launching an application.
    #
    # Extraction runs on the raw text rather than ``InputNormalizer`` output:
    # the normalizer replaces every non-word character with a space, which
    # would turn "git status" into "git status" but also silently rewrite
    # "ls -la" and strip backticks from "run `ls`".
    # ------------------------------------------------------------------

    TERMINAL_COMMAND_REGEXES = [
        # "run this terminal command: ls -la", "execute the command: pip list"
        re.compile(r"^(?:please\s+|can you\s+|could you\s+)?"
                   r"(?:run|execute|perform)\s+"
                   r"(?:this\s+|that\s+|the\s+|my\s+|a\s+)*"
                   r"(?:terminal\s+|shell\s+|console\s+|cmd\s+|bash\s+)?"
                   r"commands?\b\s*[:\-]?\s*(.+)$"),
        # "run `ls` in terminal", "execute pip list using the shell"
        re.compile(r"^(?:please\s+|can you\s+|could you\s+)?"
                   r"(?:run|execute|perform)\s+[`'\"]?(?P<cmd>.+?)[`'\"]?\s+"
                   r"(?:in|using|via|on)\s+(?:the\s+|a\s+)?"
                   r"(?:terminal|shell|console|cmd|bash|powershell)\b"),
        # "open the terminal", "launch console"
        re.compile(r"^(?:please\s+)?(?:open|launch|start)\s+(?:the\s+)?"
                   r"(?:terminal|console|shell|powershell)\b"),
    ]

    #: Backticks/quotes around an inline command are presentation, not content.
    _COMMAND_WRAPPERS = re.compile(r"^[\s`'\"$]+|[\s`'\"$]+$")

    # ------------------------------------------------------------------
    # Application names.
    #
    # Duplicated from ``app/tools/applications/launcher.py`` on purpose: the
    # launcher imports pyautogui, which the model layer must not require. The
    # values are the canonical short aliases the launcher already resolves, and
    # the same normalisation ``scripts/prepare_needle_dataset.py`` applied to
    # the training data's ``name`` arguments.
    # ------------------------------------------------------------------

    _APP_ALIASES = {
        "chrome": "chrome",
        "google chrome": "chrome",
        "google's browser": "chrome",
        "browser": "chrome",
        "web browser": "chrome",
        "firefox": "firefox",
        "mozilla firefox": "firefox",
        "edge": "edge",
        "msedge": "edge",
        "microsoft edge": "edge",
        "notepad": "notepad",
        "calc": "calc",
        "calculator": "calc",
        "code": "code",
        "vscode": "code",
        "vs code": "code",
        "visual studio code": "code",
        "spotify": "spotify",
        "explorer": "explorer",
        "file explorer": "explorer",
        "terminal": "terminal",
        "cmd": "cmd",
        "powershell": "powershell",
    }

    #: Category nouns that describe the *kind* of app, not its name.
    _APP_CATEGORY_NOUNS = re.compile(
        r"\s+(?:app|application|browser|web\s+browser|program|player|"
        r"client|editor|window)$")

    #: "close it", "quit that" -- the referent has to come from earlier context.
    _APP_PRONOUNS = re.compile(r"^(?:it|that|this|them|these|those|him|her)$")

    # ------------------------------------------------------------------
    # Media transport.
    #
    # Matched as verb-phrase shapes rather than bare keywords, so that "next",
    # "stop" or "keep" can never decide the action on their own: every pattern
    # has to consume the whole clause. Checked before the app rules so that
    # "start playing music" is never read as launching an app called
    # "playing music".
    #
    # V4 is measurably unreliable here: offered the four media tools it
    # answered media_previous for "Go to the next song", media_pause for "Keep
    # the music going" and media_play for "Take me back one song". These shapes
    # are unambiguous to a person, so they are decided deterministically.
    # ------------------------------------------------------------------

    #: Words that can end a media clause. "keep the music going" is a play
    #: request; the gerund is filler, not a different intent.
    _MEDIA_FILLER = r"(?:\s+(?:going|on|off|please|now|for\s+me|again|" \
                    r"if\s+you\s+(?:can|would|could)))?"

    #: The object of a media verb. "player" is excluded on purpose: "stop the
    #: music player" is closing an application, not pausing playback.
    _MEDIA_OBJECT = r"(?:song|track|tune|music|playback|audio|media|video|" \
                    r"podcast|playlist|album|tv|show|episode|file|item|one)"

    #: Verb phrases that mean "start / resume playback".
    MEDIA_PLAY_PATTERNS = [
        # start|begin|resume|continue|keep|play [the|it|on|up|back|playing] <object>
        re.compile(r"^(?:start|begin|resume|continue|keep|carry\s+on|play)\s+"
                   r"(?:the\s+|this\s+|that\s+|it\s+|on\s+|up\s+|back\s+|"
                   r"playing\s+|please\s+|just\s+)*"
                   r"(?:music|song|track|playback|audio|media|video|podcast|"
                   r"tv|show|playing|play)"
                   + _MEDIA_FILLER + r"$"),
        re.compile(r"^(?:unpause|resume|continue)\s*$"),
        # "keep the music going" / "let the music keep playing"
        re.compile(r"^let\s+(?:the\s+|it\s+)?(?:music|audio|playback|song)\s+"
                   r"(?:keep\s+|go\s+|start\s+|continue\s+)?"
                   r"(?:playing|going|running|on)"
                   + _MEDIA_FILLER + r"$"),
        re.compile(r"^keep\s+(?:the\s+|it\s+)?(?:music|audio|playback|song|"
                   r"track)\s+(?:going|playing|on|rolling)"
                   + _MEDIA_FILLER + r"$"),
    ]

    #: Verb phrases that mean "pause".
    MEDIA_PAUSE_PATTERNS = [
        re.compile(r"^(?:pause|halt)\s+(?:the\s+|this\s+|that\s+)*"
                   + _MEDIA_OBJECT + r"(?:\s+player)?$"),
        re.compile(r"^(?:pause|halt)\s*$"),
        # "stop the music" is pause; "stop spotify" is closing an app, so the
        # object is required here.
        re.compile(r"^(?:stop|end)\s+(?:the\s+|this\s+|that\s+)*"
                   + _MEDIA_OBJECT + r"(?!\s*player)\s*$"),
        re.compile(r"^(?:i(?:'m| am)\s+)?done\s+listening\.?\s*$"),
    ]

    #: Verb phrases that mean "skip to the next item".
    MEDIA_NEXT_PATTERNS = [
        re.compile(r"^(?:skip|jump|move|switch|go|scroll|fast[-\s]?forward|"
                   r"fwd|advance|roll)\s*"
                   r"(?:on\s+|over\s+|to\s+|forward\s+|ahead\s+)*"
                   r"(?:the\s+)?(?:next\s+)?"
                   + _MEDIA_OBJECT + r"?\s*$"),
        re.compile(r"^(?:next|forward|advance|skip)\s*"
                   r"(?:the\s+)?(?:one\s+|track\s+|song\s+)?"
                   + _MEDIA_OBJECT + r"?\s*$"),
        re.compile(r"^(?:play|listen\s+to|listen\s+for)\s+(?:the\s+)?next\s+"
                   + _MEDIA_OBJECT + r"\s*$"),
        re.compile(r"^(?:skip|go|move|jump)\s+(?:forward|ahead|on)\s*$"),
    ]

    #: Verb phrases that mean "go back to the previous item".
    MEDIA_PREV_PATTERNS = [
        re.compile(r"^(?:go|move|jump|switch|scroll|take\s+me|play|listen\s+to|"
                   r"return|rewind|fast[-\s]?back|replay)\s*"
                   r"(?:me\s+)?(?:back\s*)?(?:one\s*)?(?:to\s+)?(?:the\s+)?"
                   r"(?:previous|last|prior|preceding)\s+"
                   + _MEDIA_OBJECT + r"?\s*$"),
        re.compile(r"^(?:previous|prev|last|prior|preceding|back|rewind)\s*"
                   r"(?:one\s*)?(?:the\s+)?"
                   + _MEDIA_OBJECT + r"?\s*$"),
        # "take me back one song", "go back one track"
        re.compile(r"^(?:take\s+me\s+|go\s+|move\s+|jump\s+)?"
                   r"(?:back|rewind|return)\s*(?:me\s+)?(?:one\s+)?"
                   r"(?:to\s+)?(?:the\s+)?"
                   + _MEDIA_OBJECT + r"?\s*$"),
        re.compile(r"^(?:take\s+me\s+)?(?:back|rewind)\s*(?:me\s+)?$"),
    ]

    #: Ordered so the most specific intent wins. Play is deliberately last: a
    #: clause that names a direction is never a bare play request.
    MEDIA_INTENTS = (
        ("media_pause", MEDIA_PAUSE_PATTERNS),
        ("media_previous", MEDIA_PREV_PATTERNS),
        ("media_next", MEDIA_NEXT_PATTERNS),
        ("media_play", MEDIA_PLAY_PATTERNS),
    )

    #: If an app-open target contains any of these it is a media command, not an
    #: application name. Without this, "start playing music" launches an app
    #: literally named "playing music".
    _MEDIA_WORDS = re.compile(
        r"\b(play|playing|playback|music|song|track|video|audio|podcast|"
        r"media|album|playlist|resume|pause)\b")

    #: Explicit shell/command language. An app target containing any of these
    #: is a terminal request, never an application name: without this,
    #: "Run this terminal command: ls" launched an app called "terminal command".
    _COMMAND_WORDS = re.compile(
        r"\b(terminal|console|command\s+line|shell|bash|powershell|cmd|"
        r"batch|script|command|commands)\b")

    #: Articles are dropped for exact-pattern lookup so "pause the song" and
    #: "pause song" resolve identically.
    _ARTICLES = re.compile(r"\b(?:the|my|a|an)\b")

    @classmethod
    def _drop_articles(cls, normalized: str) -> str:
        stripped = cls._ARTICLES.sub(" ", normalized)
        return re.sub(r"\s+", " ", stripped).strip()

    @classmethod
    def _clauses(cls, raw: str) -> List[str]:
        """Candidate strings to try for a verb-phrase intent.

        The whole request first, then its trailing clauses. This is what makes
        "I'm done listening, pause the music" resolve as a pause, and "I don't
        need Chrome anymore, close it" as a close, instead of being rejected for
        having a leading clause. ``InputNormalizer`` replaces commas with
        spaces, so clause splitting runs on the comma-preserving form.
        """
        whole = cls._normalize_keeping_paths(raw)
        if not whole:
            return []
        clauses = [c.strip(" .!?") for c in whole.split(",")]
        ordered = [whole]
        ordered.extend(c for c in reversed(clauses) if c and c != whole)
        return ordered

    @classmethod
    def _match_media(cls, raw: str) -> Optional[str]:
        """Return the media tool for ``raw``, or None if it is not a transport
        command. Returns None rather than guessing on a partial match."""
        for candidate in cls._clauses(raw):
            dearticled = cls._drop_articles(candidate)
            for tool, patterns in cls.MEDIA_INTENTS:
                for pattern in patterns:
                    if pattern.match(dearticled):
                        return tool
        return None

    @classmethod
    def _match_terminal(cls, raw: str) -> Optional[MatchResult]:
        """Deterministic route for explicit shell/command requests.

        Returns a MatchResult for ``execute_terminal_command``, or None so the
        caller can keep escalating. The command is only accepted when the tool's
        own whitelist can act on it; anything else is left to Tier 2 rather than
        handed to a tool that would only reject it.
        """
        normalized = cls._normalize_keeping_paths(raw)
        if not normalized:
            return None
        dearticled = cls._drop_articles(normalized)

        for pattern in cls.TERMINAL_COMMAND_REGEXES:
            match = pattern.match(normalized) or pattern.match(dearticled)
            if not match:
                continue
            if pattern is cls.TERMINAL_COMMAND_REGEXES[-1]:
                # "open the terminal" launches the terminal app, it is not a
                # command, so no argument to extract.
                return MatchResult(
                    matched=True, tool_name="execute_terminal_command",
                    arguments={"command": ""}, confidence=1.0,
                    reason="Tier 0 terminal launch rule (no model involved)")
            command = match.groupdict().get("cmd")
            if command is None:
                command = match.group(1)
            command = cls._COMMAND_WRAPPERS.sub("", (command or "")).strip()
            command = cls._ARG_NOISE.sub("", command).strip()
            if not command:
                continue
            binary = command.split()[0].lower().strip("'\"`")
            if binary not in ALLOWED_TERMINAL_COMMANDS:
                # Not on the tool's whitelist; escalating is honest, whereas
                # routing it would produce a guaranteed tool failure.
                return MatchResult(
                    matched=False,
                    reason=(f"Tier 0 detected a terminal command ({binary!r}) but it is "
                            f"not in the allowed set; escalate to Tier 2"))
            return MatchResult(
                matched=True, tool_name="execute_terminal_command",
                arguments={"command": command}, confidence=1.0,
                reason=(f"Tier 0 terminal rule for whitelisted {binary!r} "
                        f"(no model involved)"))
        return None

    @classmethod
    def _normalize_app_name(cls, name: str, context: str = "") -> Optional[str]:
        """Turn an extracted app target into a name the launcher understands.

        Handles the natural-language shapes that used to leak through as
        garbage: possessives ("Google's browser"), a bare category noun ("the
        browser"), and pronouns referring back to an app named earlier in the
        same request ("I don't need Chrome anymore, close it").
        """
        cleaned = (name or "").strip()
        if not cleaned:
            return None

        # A trailing category noun describes the kind of app, not its name:
        # "Google's browser" -> possessor "google", noun "browser".
        generic = cleaned
        generic = cls._APP_CATEGORY_NOUNS.sub("", generic).strip()
        noun_is_generic = len(generic) != len(cleaned)

        # Possessive: "Google's" -> "Google".
        generic = re.sub(r"[’']s\b", "", generic)
        generic = re.sub(r"^(?:the|a|an|my)\s+", "", generic.strip())
        generic = re.sub(r"\s+", " ", generic).strip(" .!?,")

        if cls._COMMAND_WORDS.search(generic or ""):
            return None

        if not generic or cls._APP_PRONOUNS.match(generic):
            # Either a bare category noun ("the browser") or a pronoun ("close
            # it"); in both cases the referent comes from the wider request.
            resolved = cls._find_app_in_text(context)
            if resolved:
                return resolved
            return "chrome" if noun_is_generic or not generic else None

        resolved = cls._APP_ALIASES.get(generic)
        if resolved:
            return resolved
        if noun_is_generic:
            # "Google's browser" where the possessor is not itself a known app:
            # the only browser worth launching is Chrome.
            return "chrome"
        return generic

    @classmethod
    def _find_app_in_text(cls, text: str) -> Optional[str]:
        """First known application named in ``text``, longest alias first."""
        haystack = (text or "").lower()
        best = None
        for alias in sorted(cls._APP_ALIASES, key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", haystack):
                if best is None or len(alias) > len(best):
                    best = alias
        return cls._APP_ALIASES.get(best) if best else None

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
        #    not become launch_app("playing music"). Clause-aware, so
        #    "I'm done listening, pause the music" resolves as a pause.
        media_tool = cls._match_media(user_input)
        if media_tool is not None:
            return MatchResult(
                matched=True, tool_name=media_tool, arguments={},
                confidence=1.0,
                reason=(f"Tier 0 {media_tool} verb-phrase rule (V4 answered "
                        f"the wrong media tool for these shapes)"))

        # 6. Explicit terminal commands, before the app rules: "Run this
        #    terminal command: ls" must not become launch_app("terminal
        #    command"). A non-whitelisted binary returns matched=False so the
        #    router escalates instead of failing inside the tool.
        terminal = cls._match_terminal(user_input)
        if terminal is not None:
            if terminal.matched:
                return terminal
            return MatchResult(matched=False, reason=terminal.reason)

        # 7. App close before app open, so "close chrome" is never an open.
        #    Matched against the comma/possessive-preserving form, not
        #    ``InputNormalizer`` output, which turns "Google's" into "google s".
        #    Each trailing clause is tried so a leading aside does not hide the
        #    actual verb.
        for candidate in cls._clauses(user_input):
            for pattern in cls.APP_CLOSE_REGEXES:
                m = pattern.match(candidate)
                if not m:
                    continue
                name = cls._normalize_app_name(m.group(1), context=user_input)
                if name and len(name) > 1 and not cls._MEDIA_WORDS.search(name):
                    return MatchResult(
                        matched=True,
                        tool_name="close_app",
                        arguments={"name": name},
                        confidence=1.0,
                        reason=f"Tier 0 close_app for '{name}'"
                    )

        # 8. App open
        for candidate in cls._clauses(user_input):
            for pattern in cls.APP_OPEN_REGEXES:
                m = pattern.match(candidate)
                if not m:
                    continue
                name = cls._normalize_app_name(m.group(1), context=user_input)
                if name and len(name) > 1 and not cls._MEDIA_WORDS.search(name):
                    return MatchResult(
                        matched=True,
                        tool_name="launch_app",
                        arguments={"name": name},
                        confidence=1.0,
                        reason=f"Tier 0 launch_app for '{name}'"
                    )

        return MatchResult(matched=False, reason="No Tier 0 match")

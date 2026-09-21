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
        "unmute": ("unmute", {}),
        "unmute volume": ("unmute", {}),
        "unmute sound": ("unmute", {}),
        "get volume": ("get_volume", {}),
        "what is the volume": ("get_volume", {}),
        "current volume": ("get_volume", {}),

        # Media
        "play": ("media_play", {}),
        "play music": ("media_play", {}),
        "resume": ("media_play", {}),
        "pause": ("media_pause", {}),
        "pause music": ("media_pause", {}),
        "stop music": ("media_pause", {}),
        "next": ("media_next", {}),
        "next song": ("media_next", {}),
        "next track": ("media_next", {}),
        "skip song": ("media_next", {}),
        "previous": ("media_previous", {}),
        "previous song": ("media_previous", {}),
        "previous track": ("media_previous", {}),
        "prev song": ("media_previous", {}),

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

    @classmethod
    def match(cls, user_input: str) -> MatchResult:
        """
        Attempts to match raw user input against Tier 0 direct reflex patterns.
        Returns MatchResult with matched=True if direct pattern found.
        """
        if not user_input:
            return MatchResult(matched=False, reason="Empty input")

        normalized = InputNormalizer.normalize(user_input)

        # 1. Exact pattern lookup
        if normalized in cls.EXACT_PATTERNS:
            tool_name, args = cls.EXACT_PATTERNS[normalized]
            return MatchResult(
                matched=True,
                tool_name=tool_name,
                arguments=args,
                confidence=1.0,
                reason=f"Tier 0 exact match for '{normalized}'"
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

        return MatchResult(matched=False, reason="No Tier 0 match")

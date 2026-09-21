"""
Input text normalizer and alias expansion engine for Vector Desktop AI Assistant.
Clean up and standardize user text queries before matching or routing.
"""

import re
from typing import Dict

# Dictionary mapping common spoken/written shortcuts to canonical keywords
ALIAS_MAP: Dict[str, str] = {
    "vol": "volume",
    "sound": "volume",
    "audio": "volume",
    "ram": "memory_usage",
    "memory": "memory_usage",
    "specs": "system_stats",
    "stats": "system_stats",
    "storage": "disk_usage",
    "disk": "disk_usage",
    "battery": "battery_status",
    "cpu": "cpu_usage",
    "lock": "lock_pc",
    "sleep": "sleep_pc",
    "restart": "restart_pc",
    "shutdown": "shutdown_pc",
}


class InputNormalizer:
    """
    Normalizes user input text by lowercasing, stripping punctuation,
    collapsing whitespace, and expanding common command aliases.
    """

    @staticmethod
    def normalize(text: str) -> str:
        """
        Cleans and normalizes raw text input.
        """
        if not text:
            return ""

        # Lowercase
        normalized = text.lower().strip()

        # Remove punctuation except numbers, spaces, and percent signs
        normalized = re.sub(r"[^\w\s%]", " ", normalized)

        # Collapse whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    @classmethod
    def expand_aliases(cls, text: str) -> str:
        """
        Normalizes text and replaces shorthand word aliases with canonical terms.
        """
        cleaned = cls.normalize(text)
        words = cleaned.split()
        expanded_words = [ALIAS_MAP.get(word, word) for word in words]
        return " ".join(expanded_words)

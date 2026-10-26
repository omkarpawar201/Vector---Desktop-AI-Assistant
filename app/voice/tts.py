"""
Native Windows Text-to-Speech (TTS) Engine for Vector Desktop AI Assistant.
Uses Windows SAPI5 via pyttsx3 for zero-RAM overhead and instant speech playback.
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:
    pyttsx3 = None  # type: ignore
    _PYTTSX3_AVAILABLE = False
    logger.debug("pyttsx3 not available; TTS disabled")


class TTSEngine:
    """
    Zero-RAM native Windows speech synthesis engine.
    Executes speech generation asynchronously to prevent blocking the UI event loop.
    """

    def __init__(self, rate: int = 180, volume: float = 1.0):
        self.rate = rate
        self.volume = volume
        self._enabled = bool(_PYTTSX3_AVAILABLE)
        self._available = bool(_PYTTSX3_AVAILABLE)

    def _speak_thread(self, text: str) -> None:
        if not _PYTTSX3_AVAILABLE or pyttsx3 is None:
            return
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass

    def speak(self, text: str) -> None:
        """
        Speaks text asynchronously on a background daemon thread.
        """
        if not self._enabled or not self._available or not text or not text.strip():
            return

        thread = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
        thread.start()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable TTS speech output."""
        if enabled and not self._available:
            self._enabled = False
            return
        self._enabled = bool(enabled and self._available)

    @property
    def is_enabled(self) -> bool:
        return bool(self._enabled and self._available)

    @property
    def is_available(self) -> bool:
        return bool(self._available)

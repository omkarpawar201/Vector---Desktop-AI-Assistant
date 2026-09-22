"""
Native Windows Text-to-Speech (TTS) Engine for Vector Desktop AI Assistant.
Uses Windows SAPI5 via pyttsx3 for zero-RAM overhead and instant speech playback.
"""

import threading
from typing import Optional
import pyttsx3


class TTSEngine:
    """
    Zero-RAM native Windows speech synthesis engine.
    Executes speech generation asynchronously to prevent blocking the UI event loop.
    """

    def __init__(self, rate: int = 180, volume: float = 1.0):
        self.rate = rate
        self.volume = volume
        self._enabled = True

    def _speak_thread(self, text: str) -> None:
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
        if not self._enabled or not text or not text.strip():
            return

        thread = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
        thread.start()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable TTS speech output."""
        self._enabled = enabled

    @property
    def is_enabled(self) -> bool:
        return self._enabled

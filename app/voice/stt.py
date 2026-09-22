"""
Speech-to-Text (STT) Engine for Vector Desktop AI Assistant.
Supports local SpeechRecognition and faster-whisper engines.
"""

from pathlib import Path
from pathlib import Path
from typing import Optional


class STTEngine:
    """
    Speech-to-Text engine supporting SpeechRecognition and faster-whisper models.
    """

    def __init__(self, model_size: str = "tiny.en"):
        self.model_size = model_size
        self._whisper_model = None

    @property
    def is_loaded(self) -> bool:
        """
        Returns True if the local whisper model has been lazily instantiated.
        Useful for the UI to show an accurate 'model not yet loaded' state
        without forcing an eager model download at startup.
        """
        return self._whisper_model is not None

    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribes a WAV audio file to text using SpeechRecognition or faster-whisper.
        """
        p = Path(audio_path).resolve()
        if not p.exists() or p.stat().st_size == 0:
            return ""

        # 1. Try SpeechRecognition (Fastest, zero-config on Windows)
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.AudioFile(str(p)) as source:
                audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            if text and text.strip():
                return text.strip()
        except Exception:
            pass

        # 2. Try faster-whisper fallback
        try:
            from faster_whisper import WhisperModel
            if self._whisper_model is None:
                self._whisper_model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            segments, _ = self._whisper_model.transcribe(str(p), beam_size=1)
            text_parts = [s.text for s in segments]
            return " ".join(text_parts).strip()
        except Exception:
            pass

        return ""

    def listen_and_transcribe_mic(self, phrase_time_limit: int = 6) -> str:
        """
        Directly captures speech from default Windows microphone using SpeechRecognition.
        """
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.3)
                audio = r.listen(source, timeout=5, phrase_time_limit=phrase_time_limit)
            return r.recognize_google(audio).strip()
        except Exception as e:
            return ""

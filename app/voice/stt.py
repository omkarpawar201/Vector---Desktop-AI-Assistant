"""
Speech-to-Text (STT) Engine for Vector Desktop AI Assistant.

Transcription tries the bundled SpeechRecognition path first and then a local
faster-whisper model *if one is installed*.  No offline engine is claimed or
required: when no backend is usable the engine returns an empty transcription
and logs a concise, non-sensitive reason (never the audio content or keys).
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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
        Transcribes a WAV audio file to text using SpeechRecognition or, when the
        optional dependency is present, a local faster-whisper model.
        """
        p = Path(audio_path).resolve()
        if not p.exists() or p.stat().st_size == 0:
            return ""

        # 1. SpeechRecognition (zero-config on Windows). Note: recognize_google
        #    is a network call, so this is not an offline path.
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.AudioFile(str(p)) as source:
                audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            if text and text.strip():
                return text.strip()
            logger.info("STT: recognizer returned no text for %s", p.name)
        except ImportError:
            logger.warning("STT: 'speech_recognition' is not installed")
        except Exception as e:
            logger.warning("STT: SpeechRecognition failed (%s): %s", type(e).__name__, e)

        # 2. Optional local faster-whisper model, only if it is installed.
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logger.warning("STT: local engine 'faster_whisper' is not installed")
            return ""

        try:
            if self._whisper_model is None:
                self._whisper_model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            segments, _ = self._whisper_model.transcribe(str(p), beam_size=1)
            return " ".join(s.text for s in segments).strip()
        except Exception as e:
            logger.warning("STT: faster-whisper failed (%s): %s", type(e).__name__, e)
            return ""

    def listen_and_transcribe_mic(self, phrase_time_limit: int = 6) -> str:
        """
        Captures speech from the default Windows microphone and transcribes it.

        Returns an empty string when nothing intelligible was captured or the
        backend is unavailable.  Failures are logged by reason only -- the audio
        content and any credentials are never logged.
        """
        try:
            import speech_recognition as sr
        except ImportError:
            logger.warning("STT: 'speech_recognition' is not installed; microphone capture unavailable")
            return ""

        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=phrase_time_limit)
        except Exception as e:
            logger.warning("STT: microphone capture failed (%s): %s", type(e).__name__, e)
            return ""

        try:
            return recognizer.recognize_google(audio).strip()
        except sr.UnknownValueError:
            logger.info("STT: speech was not intelligible")
            return ""
        except sr.RequestError as e:
            logger.warning("STT: recognition service unreachable (%s)", type(e).__name__)
            return ""
        except Exception as e:
            logger.warning("STT: transcription failed (%s): %s", type(e).__name__, e)
            return ""

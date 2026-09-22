"""
Voice Push-to-Talk (PTT) listener and microphone audio recorder for Vector Desktop AI Assistant.
"""

from typing import Optional
from app.voice.stt import STTEngine


class VoiceListener:
    """
    Manages push-to-talk microphone recording and speech transcription.
    """

    def __init__(self, stt_engine: Optional[STTEngine] = None):
        self.stt = stt_engine or STTEngine()
        self.is_recording = False

    def listen_from_mic(self) -> str:
        """
        Listens directly to the Windows microphone and transcribes spoken text.
        """
        return self.stt.listen_and_transcribe_mic(phrase_time_limit=6)

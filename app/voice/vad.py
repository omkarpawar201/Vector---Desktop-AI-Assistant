"""
Voice Activity Detection (VAD) Engine for Vector Desktop AI Assistant.
Detects speech presence and strips silent audio chunks.
"""

import math
import struct
from typing import List


class VADEngine:
    """
    Lightweight audio energy & peak volume Voice Activity Detector.
    """

    def __init__(self, energy_threshold: float = 300.0, sample_rate: int = 16000):
        self.energy_threshold = energy_threshold
        self.sample_rate = sample_rate

    def calculate_rms(self, pcm_data: bytes) -> float:
        """
        Calculates Root Mean Square (RMS) energy level of 16-bit PCM audio samples.
        """
        if not pcm_data:
            return 0.0

        count = len(pcm_data) // 2
        if count == 0:
            return 0.0

        format_str = f"<{count}h"
        try:
            shorts = struct.unpack(format_str, pcm_data)
            sum_squares = sum(s * s for s in shorts)
            return math.sqrt(sum_squares / count)
        except Exception:
            return 0.0

    def is_speech(self, pcm_data: bytes) -> bool:
        """
        Returns True if the RMS energy level exceeds the speech threshold.
        """
        rms = self.calculate_rms(pcm_data)
        return rms >= self.energy_threshold

    def filter_silence(self, pcm_chunks: List[bytes]) -> bytes:
        """
        Strips silent leading and trailing PCM audio chunks.
        """
        speech_chunks = [chunk for chunk in pcm_chunks if self.is_speech(chunk)]
        return b"".join(speech_chunks)

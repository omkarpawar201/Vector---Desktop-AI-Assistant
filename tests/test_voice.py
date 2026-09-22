"""
Unit tests for TTSEngine, STTEngine, VADEngine, and local privacy mode.
"""

from app.config.settings import Settings
from app.core.router import IntentRouter
from app.tools.registry import ToolRegistry
from app.voice.listener import VoiceListener
from app.voice.stt import STTEngine
from app.voice.tts import TTSEngine
from app.voice.vad import VADEngine


def test_tts_engine():
    """Verify native Windows SAPI5 TTS engine configuration."""
    tts = TTSEngine(rate=200, volume=0.9)
    assert tts.is_enabled is True
    assert tts.rate == 200

    # Test toggling speech
    tts.set_enabled(False)
    assert tts.is_enabled is False


def test_vad_engine_rms_calculation():
    """Verify Voice Activity Detection RMS energy calculation."""
    vad = VADEngine(energy_threshold=100.0)

    # Empty audio
    assert vad.calculate_rms(b"") == 0.0

    # Silent PCM audio (16-bit zeros)
    silent_pcm = b"\x00\x00" * 100
    assert vad.calculate_rms(silent_pcm) == 0.0
    assert vad.is_speech(silent_pcm) is False


def test_stt_engine_lazy_loading():
    """Verify STT engine lazy loading state."""
    stt = STTEngine(model_size="tiny.en")
    assert stt.is_loaded is False  # Must not load model on __init__


def test_offline_privacy_mode():
    """Verify complete local-only mode when ENABLE_GEMINI=False."""
    settings = Settings(enable_gemini=False)
    reg = ToolRegistry()
    router = IntentRouter(settings=settings, registry=reg)

    res = router.route_and_execute("Write a complex code refactoring plan")
    assert "Gemini unavailable" in res.response_text

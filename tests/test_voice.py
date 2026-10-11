"""
Unit tests for TTSEngine, STTEngine, VADEngine, VoiceListener, the voice UI
toggle, and local privacy mode.

Everything here is mocked/offline: no microphone is opened, no audio is played,
no network transcription call is made, and no audio content is logged.
"""

import os
import struct
import sys
import types

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest

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


# --------------------------------------------------------------------------
# VAD edge-silence semantics
# --------------------------------------------------------------------------

def _speech_chunk(value: int = 1000, count: int = 100) -> bytes:
    """A chunk whose RMS (abs value) exceeds the test threshold."""
    return struct.pack(f"<{count}h", *([value] * count))


def _silence_chunk(count: int = 100) -> bytes:
    return b"\x00\x00" * count


def test_filter_silence_strips_leading():
    vad = VADEngine(energy_threshold=300.0)
    speech = _speech_chunk()
    silence = _silence_chunk()

    assert vad.filter_silence([silence, speech, speech]) == speech + speech


def test_filter_silence_strips_trailing():
    vad = VADEngine(energy_threshold=300.0)
    speech = _speech_chunk()
    silence = _silence_chunk()

    assert vad.filter_silence([speech, silence]) == speech


def test_filter_silence_preserves_interior():
    vad = VADEngine(energy_threshold=300.0)
    speech = _speech_chunk()
    silence = _silence_chunk()

    # An interior pause between words must NOT be clipped.
    assert vad.filter_silence([speech, silence, speech]) == speech + silence + speech


def test_filter_silence_all_silent_is_empty():
    vad = VADEngine(energy_threshold=300.0)
    silence = _silence_chunk()

    assert vad.filter_silence([silence, silence]) == b""
    assert vad.filter_silence([]) == b""


# --------------------------------------------------------------------------
# STTEngine with a mocked SpeechRecognition module (no mic, no network)
# --------------------------------------------------------------------------

class _FakeAudioFile:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeRecognizer:
    def __init__(self, phrase="hello vector", fail=None):
        self._phrase = phrase
        self._fail = fail

    def record(self, source):
        return b"audio"

    def adjust_for_ambient_noise(self, source, duration=0.3):
        return None

    def listen(self, source, timeout=None, phrase_time_limit=None):
        return b"audio"

    def recognize_google(self, audio):
        if self._fail is not None:
            raise self._fail
        return self._phrase


class _FakeMic:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_sr_module(recognizer: _FakeRecognizer) -> types.SimpleNamespace:
    class UnknownValueError(Exception):
        pass

    class RequestError(Exception):
        pass

    return types.SimpleNamespace(
        Recognizer=lambda: recognizer,
        AudioFile=_FakeAudioFile,
        Microphone=_FakeMic,
        UnknownValueError=UnknownValueError,
        RequestError=RequestError,
    )


def test_transcribe_file_empty_file_returns_empty(tmp_path):
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    assert STTEngine().transcribe_file(str(empty)) == ""


def test_transcribe_file_uses_speech_recognition(tmp_path, monkeypatch):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFFxxxx")

    monkeypatch.setitem(
        sys.modules,
        "speech_recognition",
        _fake_sr_module(_FakeRecognizer(phrase="  open chrome  ")),
    )

    assert STTEngine().transcribe_file(str(wav)) == "open chrome"


def test_transcribe_file_logs_reason_without_secrets(tmp_path, monkeypatch, caplog):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFFxxxx")

    # SpeechRecognition present but broken; faster_whisper absent (not installed).
    monkeypatch.setitem(
        sys.modules,
        "speech_recognition",
        _fake_sr_module(_FakeRecognizer(fail=RuntimeError("no device"))),
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", None)

    with caplog.at_level("WARNING", logger="app.voice.stt"):
        assert STTEngine().transcribe_file(str(wav)) == ""

    logged = " ".join(r.message for r in caplog.records)
    assert "SpeechRecognition failed" in logged
    assert "faster_whisper" in logged


def test_listen_and_transcribe_mic_returns_text(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "speech_recognition",
        _fake_sr_module(_FakeRecognizer(phrase="  turn on the lights  ")),
    )

    assert STTEngine().listen_and_transcribe_mic() == "turn on the lights"


def test_listen_and_transcribe_mic_unintelligible_returns_empty(monkeypatch, caplog):
    fake = _fake_sr_module(_FakeRecognizer())
    rec = _FakeRecognizer(fail=fake.UnknownValueError("nope"))
    fake.Recognizer = lambda: rec
    monkeypatch.setitem(sys.modules, "speech_recognition", fake)

    with caplog.at_level("INFO", logger="app.voice.stt"):
        assert STTEngine().listen_and_transcribe_mic() == ""

    assert any("not intelligible" in r.message for r in caplog.records)


def test_listen_and_transcribe_mic_capture_failure(monkeypatch, caplog):
    class _BrokenMic:
        def __enter__(self):
            raise RuntimeError("no microphone")

        def __exit__(self, *exc):
            return False

    fake = _fake_sr_module(_FakeRecognizer())
    fake.Microphone = _BrokenMic
    monkeypatch.setitem(sys.modules, "speech_recognition", fake)

    with caplog.at_level("WARNING", logger="app.voice.stt"):
        assert STTEngine().listen_and_transcribe_mic() == ""

    assert any("microphone capture failed" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# VoiceListener delegation
# --------------------------------------------------------------------------

class _RecordingSTT(STTEngine):
    def __init__(self, text: str):
        super().__init__()
        self._text = text
        self.calls = 0

    def listen_and_transcribe_mic(self, phrase_time_limit: int = 6) -> str:
        self.calls += 1
        return self._text


def test_voice_listener_delegates_to_stt():
    fake = _RecordingSTT("open notepad")
    listener = VoiceListener(stt_engine=fake)

    assert listener.listen_from_mic() == "open notepad"
    assert fake.calls == 1


# --------------------------------------------------------------------------
# TTS behavior with pyttsx3 mocked (no audio output)
# --------------------------------------------------------------------------

def _install_fake_pyttsx3(monkeypatch, tts_module):
    calls = []

    class _FakeEngine:
        def setProperty(self, key, value):
            calls.append(("set", key, value))

        def say(self, text):
            calls.append(("say", text))

        def runAndWait(self):
            calls.append(("run",))

    monkeypatch.setattr(tts_module.pyttsx3, "init", lambda: _FakeEngine())
    return calls


def test_tts_speak_disabled_is_noop(monkeypatch):
    import app.voice.tts as tts_module

    def _boom():
        raise AssertionError("pyttsx3.init must not be called when disabled")

    monkeypatch.setattr(tts_module.pyttsx3, "init", _boom)

    tts = TTSEngine()
    tts.set_enabled(False)
    tts.speak("should not speak")  # must not raise / must not init


def test_tts_speak_runs_engine_when_enabled(monkeypatch):
    import app.voice.tts as tts_module

    calls = _install_fake_pyttsx3(monkeypatch, tts_module)

    class _SyncThread:
        def __init__(self, target=None, args=(), daemon=False):
            self._target = target
            self._args = args

        def start(self):
            self._target(*self._args)

    monkeypatch.setattr(tts_module.threading, "Thread", _SyncThread)

    tts = TTSEngine(rate=200, volume=0.9)
    tts.speak("hello")

    assert ("say", "hello") in calls
    assert ("run",) in calls


# --------------------------------------------------------------------------
# Voice toggle gating (offscreen; no real capture started)
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _window_with_voice(qapp, enable_voice: bool):
    from app.ui.main_window import MainWindow

    win = MainWindow()
    # Replace the cached singleton so the flag mutation is test-local.
    win.settings = Settings(enable_gemini=False, enable_voice=enable_voice)

    started = []
    win.thread_pool = types.SimpleNamespace(start=lambda worker: started.append(worker))
    return win, started


def test_voice_toggle_disabled_does_not_start_capture(qapp):
    win, started = _window_with_voice(qapp, enable_voice=False)
    try:
        win._handle_voice_toggle(True)

        assert started == [], "microphone capture started while voice is disabled"
        assert win.command_input.btn_mic.isChecked() is False
    finally:
        win.close()


def test_voice_toggle_enabled_starts_capture(qapp):
    win, started = _window_with_voice(qapp, enable_voice=True)
    try:
        win._handle_voice_toggle(True)

        assert len(started) == 1, "microphone capture worker was not started"
    finally:
        win.close()

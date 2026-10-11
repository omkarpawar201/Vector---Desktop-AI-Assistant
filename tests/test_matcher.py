"""
Unit tests for InputNormalizer and Tier0Matcher latency & pattern recognition.
"""

import time
from app.core.matcher import Tier0Matcher
from app.core.normalizer import InputNormalizer


def test_input_normalizer():
    """Test text cleaning and alias expansion."""
    raw = "  Set VOL to 40% !  "
    norm = InputNormalizer.normalize(raw)
    assert norm == "set vol to 40%"

    expanded = InputNormalizer.expand_aliases(raw)
    assert "volume" in expanded


def test_tier0_matcher_exact_patterns():
    """Test direct regex and pattern matching for reflexes."""
    # Volume
    res_vol = Tier0Matcher.match("set volume to 40")
    assert res_vol.matched is True
    assert res_vol.tool_name == "set_volume"
    assert res_vol.arguments == {"level": 40}

    # Media
    res_pause = Tier0Matcher.match("pause music")
    assert res_pause.matched is True
    assert res_pause.tool_name == "media_pause"

    # Stats
    res_cpu = Tier0Matcher.match("cpu usage")
    assert res_cpu.matched is True
    assert res_cpu.tool_name == "get_cpu_usage"

    # Power
    res_lock = Tier0Matcher.match("lock pc")
    assert res_lock.matched is True
    assert res_lock.tool_name == "lock_pc"


def test_tier0_matcher_latency_benchmark():
    """Benchmark Tier 0 matcher latency to ensure target < 5-10ms performance."""
    start_time = time.perf_counter()
    for _ in range(100):
        Tier0Matcher.match("set volume to 50%")
    end_time = time.perf_counter()

    avg_latency_ms = ((end_time - start_time) / 100) * 1000
    assert avg_latency_ms < 10.0  # Latency must be sub-10ms


def test_set_volume_accepts_spoken_percent_unit():
    """The percent sign's spoken form must reach the same reflex.

    "percent"/"per cent" are units of an absolute level, so they belong in the
    deterministic set-volume layer rather than being handed to the model, which
    read "volume at 100 percent" as get_volume.
    """
    for query, level in (("volume at 100 percent", 100),
                         ("volume at 65 percent", 65),
                         ("volume 40 percent", 40),
                         ("set the volume to 30 per cent", 30)):
        res = Tier0Matcher.match(query)
        assert res.matched is True, query
        assert res.tool_name == "set_volume", query
        assert res.arguments == {"level": level}, query


def test_set_volume_accepts_bring_verb():
    """"bring" names an absolute level just like the other directional verbs."""
    for query, level in (("bring the volume to 50", 50),
                         ("bring the sound down to 20", 20)):
        res = Tier0Matcher.match(query)
        assert res.matched is True, query
        assert res.tool_name == "set_volume", query
        assert res.arguments == {"level": level}, query


def test_existing_set_volume_syntax_unchanged():
    """The new unit/verb must not disturb the syntaxes already supported."""
    for query, level in (("set volume to 40", 40),
                         ("set volume to 40%", 40),
                         ("volume 30", 30),
                         ("volume 50%", 50),
                         ("make sound 25", 25)):
        res = Tier0Matcher.match(query)
        assert res.matched is True, query
        assert res.tool_name == "set_volume", query
        assert res.arguments == {"level": level}, query


def test_volume_queries_are_not_set_volume():
    """Reading the level must never be reframed as writing it."""
    for query in ("what is the volume", "current volume", "get volume",
                  "am i muted", "what is the audio level"):
        res = Tier0Matcher.match(query)
        assert res.tool_name != "set_volume", query


def test_mute_phrases_are_not_set_volume():
    for query in ("mute", "mute volume", "turn the volume off",
                  "silence sound", "kill the sound"):
        res = Tier0Matcher.match(query)
        assert res.matched is True, query
        assert res.tool_name == "mute", query
        assert res.arguments == {}, query


def test_unmute_phrases_are_not_set_volume():
    for query in ("unmute", "unmute sound", "turn the sound on",
                  "enable audio", "turn the volume on"):
        res = Tier0Matcher.match(query)
        assert res.matched is True, query
        assert res.tool_name == "unmute", query
        assert res.arguments == {}, query


def test_bring_verb_does_not_hijack_app_launch():
    """The verb alone is not a volume set; a level argument is still required."""
    res = Tier0Matcher.match("bring up the calculator")
    assert res.tool_name != "set_volume"

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

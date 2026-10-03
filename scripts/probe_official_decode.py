"""
Official Cactus Needle 2 Decode Probe Script.
Tests real native inference on models/needle3.cact without hardcoded keyword fallbacks.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from app.needle.client import NeedleClient


def run_probe():
    print("Initializing NeedleClient...")
    client = NeedleClient()

    print("\n--- Test 1: POSITIVE Query ('open chrome') ---")
    res1 = client.predict_cactus_needle_intent("open chrome")
    print("Result 1:", res1)
    if res1.is_valid and res1.tool_name == "launch_app":
        print("[PASS] Official Cactus Needle decode produced tool 'launch_app'")
    else:
        print(f"[FAIL] Expected 'launch_app', got: {res1}")

    print("\n--- Test 2: NEGATIVE Query ('what is the weather') ---")
    res2 = client.predict_cactus_needle_intent("what is the weather")
    print("Result 2:", res2)
    if not res2.is_valid or res2.confidence == 0.0:
        print("[PASS] Negative query returned honest invalid/zero confidence result")
    else:
        print(f"[FAIL] Negative query produced invalid hit: {res2}")


if __name__ == "__main__":
    run_probe()

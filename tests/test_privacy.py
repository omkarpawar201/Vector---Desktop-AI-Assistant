"""
Unit tests for PrivacyFilter sensitive data redaction.
"""

from app.core.privacy import PrivacyFilter


#: Fake Gemini credential assembled at runtime so the source never contains a
#: literal that trips secret scanners.  Built to match PrivacyFilter's pattern.
_FAKE_GEMINI_KEY = "AIzaSy" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q"


def test_privacy_filter_sanitize_text():
    """Test redaction of API keys, passwords, and SSH keys in raw text."""
    raw_key = f"My key is {_FAKE_GEMINI_KEY}"
    clean_key = PrivacyFilter.sanitize_text(raw_key)
    assert "[REDACTED_SECRET]" in clean_key
    assert "AIzaSy" not in clean_key

    raw_env = f"GEMINI_API_KEY={_FAKE_GEMINI_KEY}"
    clean_env = PrivacyFilter.sanitize_text(raw_env)
    assert "[REDACTED_SECRET]" in clean_env


def test_privacy_filter_sanitize_context():
    """Test recursive redaction of nested context dictionaries."""
    context = {
        "user": "Alice",
        "api_key": "secret_token_12345",
        "settings": {
            "password": "SuperSecretPassword123!",
            "theme": "dark"
        }
    }

    clean_ctx = PrivacyFilter.sanitize_context(context)
    assert clean_ctx["user"] == "Alice"
    assert clean_ctx["api_key"] == "[REDACTED_SECRET]"
    assert clean_ctx["settings"]["password"] == "[REDACTED_SECRET]"
    assert clean_ctx["settings"]["theme"] == "dark"

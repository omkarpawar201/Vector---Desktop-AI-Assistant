"""
Unit tests for PrivacyFilter sensitive data redaction.
"""

from app.core.privacy import PrivacyFilter


def test_privacy_filter_sanitize_text():
    """Test redaction of API keys, passwords, and SSH keys in raw text."""
    raw_key = "My key is AIzaSyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"
    clean_key = PrivacyFilter.sanitize_text(raw_key)
    assert "[REDACTED_SECRET]" in clean_key
    assert "AIzaSy" not in clean_key

    raw_env = "GEMINI_API_KEY=AIzaSyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"
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

"""
Privacy Filter and Sensitive-Data Redactor for Vector Desktop AI Assistant.
Prevents API keys, passwords, .env variables, and SSH keys from being sent to cloud LLMs.
"""

import re
from typing import Any, Dict, List, Union


class PrivacyFilter:
    """
    Sanitizes user queries, prompt context, and environment data before transmitting to cloud models.
    """

    REDACTION_SUBSTITUTE: str = "[REDACTED_SECRET]"

    # Compiled regex patterns for secret matching
    PATTERNS: List[re.Pattern] = [
        # Gemini / Google API Key (AIzaSy...)
        re.compile(r"AIzaSy[A-Za-z0-9_-]{30,35}"),
        # SSH / RSA Private Keys
        re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY[\s\S]*?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----"),
        # Generic API keys, secret keys, bearer tokens
        re.compile(r"(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|bearer)\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.\:\/]+['\"]?"),
        # Passwords (password=..., pwd=...)
        re.compile(r"(?i)(?:password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]+['\"]?"),
        # Standard Environment Variable Assignments with Key/Secret
        re.compile(r"(?i)(?:GEMINI_API_KEY|SECRET_KEY|DB_PASSWORD|AWS_SECRET_ACCESS_KEY)\s*=\s*[^\s]+"),
    ]

    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """
        Sanitizes a single text string by redacting any detected secret patterns.
        """
        if not text:
            return ""

        sanitized = text
        for pattern in cls.PATTERNS:
            sanitized = pattern.sub(cls.REDACTION_SUBSTITUTE, sanitized)

        return sanitized

    @classmethod
    def sanitize_context(cls, context: Union[Dict[str, Any], List[Any], str]) -> Union[Dict[str, Any], List[Any], str]:
        """
        Recursively traverses a context data structure (dict, list, str) and redacts sensitive items.
        """
        if isinstance(context, str):
            return cls.sanitize_text(context)
        elif isinstance(context, dict):
            sanitized_dict = {}
            for k, v in context.items():
                if any(sec in k.lower() for sec in ["password", "secret", "token", "api_key", "private_key"]):
                    sanitized_dict[k] = cls.REDACTION_SUBSTITUTE
                else:
                    sanitized_dict[k] = cls.sanitize_context(v)
            return sanitized_dict
        elif isinstance(context, list):
            return [cls.sanitize_context(item) for item in context]
        else:
            return context

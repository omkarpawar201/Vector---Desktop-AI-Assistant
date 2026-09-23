"""
Application settings configuration for Vector Desktop AI Assistant.
Loads values from environment variables and .env file using Pydantic Settings.
"""

from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.constants import DEFAULT_DATABASE_PATH, DEFAULT_LOG_LEVEL, DEFAULT_NEEDLE_THRESHOLD


class Settings(BaseSettings):
    """
    Vector Application Settings configuration model.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # API Keys & Cloud Configuration
    gemini_api_key: str = Field(default="", description="Google Gemini API Key")
    gemini_model: str = Field(default="gemini-2.5-flash", description="Gemini model identifier")

    # Local Model Settings
    needle_model_path: str = Field(default="models/needle_2_custom.onnx", description="Path to our custom ONNX (Tier 0.5, unambiguous name); official engine uses libneedle.dll + its own cache weights")
    needle_official_weights_path: str = Field(default="models/needle2.cact", description="Path to official Cactus Needle 2 weights")
    needle_max_decode_tokens: int = Field(default=512, description="Max decode tokens for Needle extraction")
    needle_confidence_threshold: float = Field(
        default=DEFAULT_NEEDLE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required for local execution"
    )

    # Feature Toggles & Security Policies
    enable_gemini: bool = Field(default=True, description="Enable cloud Gemini fallback")
    enable_local_llm: bool = Field(default=False, description="Enable optional local LLM (Ollama)")
    enable_voice: bool = Field(default=False, description="Enable Speech-to-Text and TTS voice pipeline")
    require_confirmation_for_dangerous: bool = Field(
        default=True,
        description="Require UI user confirmation modal for dangerous actions"
    )

    # System Paths & Logging
    database_path: str = Field(default=DEFAULT_DATABASE_PATH, description="Path to SQLite database")
    log_level: str = Field(default=DEFAULT_LOG_LEVEL, description="Logging verbosity level")

    @property
    def is_gemini_available(self) -> bool:
        """Returns True if Gemini is enabled and an API key is configured."""
        return self.enable_gemini and bool(self.gemini_api_key.strip())

    @property
    def db_path_obj(self) -> Path:
        """Returns Path object for database path, ensuring parent directory exists."""
        p = Path(self.database_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached instance of application Settings.
    """
    return Settings()

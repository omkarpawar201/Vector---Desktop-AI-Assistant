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
    needle_official_weights_path: str = Field(default="models/needle3.cact", description="Path to official Cactus Needle base weights (untuned reference)")
    needle_tuned_weights_path: str = Field(default="models/needle3_vector_v4.cact", description="Path to the local fine-tuned Needle 3 archive used for Tier 1 tool calls")
    needle_generation: int = Field(default=3, description="Needle engine generation (3 for needle3 archives)")
    needle_max_decode_tokens: int = Field(default=64, description="Max decode tokens for a Needle tool call (a call envelope is ~30 tokens; 512 only wastes prefix KV)")
    needle_candidate_use_embedding: bool = Field(
        default=False,
        description="Allow cosine-similarity retrieval to pick a candidate family when the deterministic rules do not match. Off by default: measured against the 114 held-out rows, off-topic queries score 0.929-0.943 while correctly-routed ones score 0.909-0.978, so the ranges overlap and no threshold separates them"
    )
    needle_candidate_similarity: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Cosine floor for the embedding fallback. Only consulted when needle_candidate_use_embedding is enabled; with it enabled this default is unsafe and should be raised well above 0.95"
    )
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

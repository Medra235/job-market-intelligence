"""Runtime settings for the FastAPI application."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration used to build API runtime services."""

    model_config = SettingsConfigDict(
        env_prefix="JMI_",
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "job_postings"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_local_files_only: bool = False

    llamacpp_url: str = "http://127.0.0.1:8080"
    llamacpp_model: str = "gemma-26b"
    llamacpp_timeout_seconds: float = 120.0

    classifier_model_type: str = "random_forest"

    api_base_url: str = "http://localhost:8000"

    max_context_documents: int = Field(default=5, ge=1, le=10)
    max_history_messages: int = Field(default=8, ge=1)
    max_preview_chars: int = Field(default=700, ge=100)

    profile_max_tokens: int = Field(default=1024, ge=1)


settings = Settings()

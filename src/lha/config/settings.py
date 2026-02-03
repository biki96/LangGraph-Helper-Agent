"""Configuration management using pydantic-settings."""

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return current


PROJECT_ROOT = find_project_root()


class AgentMode(str, Enum):
    """Operating mode for the agent."""

    OFFLINE = "offline"  # RAG only
    ONLINE = "online"    # Web only
    HYBRID = "hybrid"    # RAG + Web


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",
    )

    # Required
    google_api_key: str = Field(
        ...,
        description="Google Gemini API key",
    )

    # Optional
    tavily_api_key: str | None = Field(
        default=None,
        description="Tavily API key for web search (required for online mode)",
    )

    agent_mode: AgentMode = Field(
        default=AgentMode.OFFLINE,
        description="Operating mode: offline or online",
    )

    data_dir: Path = Field(
        default=Path("data"),
        description="Directory for documentation data",
    )

    chroma_persist_dir: Path = Field(
        default=Path("chroma_db"),
        description="Directory for ChromaDB persistence",
    )

    # Models
    llm_model: str = Field(
        default="gemini-2.0-flash",
        description="LLM model for generation",
    )

    rewrite_model: str = Field(
        default="gemini-2.0-flash",
        description="LLM model for query rewriting (use fast model)",
    )

    embedding_model: str = Field(
        default="models/text-embedding-004",
        description="Embedding model name",
    )

    # Retrieval
    retrieval_k: int = Field(
        default=5,
        description="Maximum number of documents to retrieve",
    )

    retrieval_score_threshold: float | None = Field(
        default=None,
        description="Minimum similarity score (0-1). None = no filtering",
    )

    @field_validator("google_api_key")
    @classmethod
    def validate_google_api_key(cls, v: str) -> str:
        if not v or v == "your_gemini_api_key_here":
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required. "
                "Get your key at: https://aistudio.google.com/app/apikey"
            )
        return v

    @model_validator(mode="after")
    def resolve_paths(self) -> "Settings":
        """Resolve relative paths to project root."""
        if not self.data_dir.is_absolute():
            object.__setattr__(self, "data_dir", PROJECT_ROOT / self.data_dir)
        if not self.chroma_persist_dir.is_absolute():
            object.__setattr__(self, "chroma_persist_dir", PROJECT_ROOT / self.chroma_persist_dir)
        return self

    def validate_for_mode(self) -> None:
        """Validate settings for the current mode.

        Raises:
            ValueError: If online/hybrid mode is requested but TAVILY_API_KEY is missing.
        """
        if self.agent_mode in (AgentMode.ONLINE, AgentMode.HYBRID) and not self.tavily_api_key:
            raise ValueError(
                f"{self.agent_mode.value} mode requires TAVILY_API_KEY. "
                "Get your key at: https://tavily.com/ or use offline mode."
            )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Application settings.

    Raises:
        ValueError: If required configuration is missing.
    """
    return Settings()  # pyright: ignore[reportCallIssue]

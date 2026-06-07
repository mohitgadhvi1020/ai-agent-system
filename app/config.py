"""Application configuration via pydantic-settings."""
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM provider: "claude", "openai", or "none" (heuristic demo mode)
    LLM_PROVIDER: str = "none"
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-opus-4-8"
    OPENAI_MODEL: str = "gpt-4o"

    # Agent behavior
    MAX_AGENT_STEPS: int = 10
    MAX_MEMORY_TURNS: int = 20
    ENABLED_TOOLS: List[str] = [
        "classify_document",
        "extract_entities",
        "summarize_text",
        "create_task",
        "send_notification",
        "search_knowledge_base",
    ]

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000


settings = Settings()

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "AI Research Assistant"
    APP_VERSION: str = "0.1.0"
    API_PREFIX: str = "/api/v1"

    OPENAI_API_KEY: str | None = None
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    TAVILY_API_KEY: str | None = None

    DATA_DIR: str = "data"
    CHROMA_DIR: str = "data/chroma"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

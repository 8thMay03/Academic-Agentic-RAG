from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "AI Research Assistant"
    APP_VERSION: str = "0.1.0"
    API_PREFIX: str = "/api/v1"

    OPENAI_API_KEY: str | None = None
    OPENAI_API_KEY_FILE: str | None = None
    OPENAI_CHAT_MODEL: str = "gpt-4.1-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    TAVILY_API_KEY: str | None = None
    TAVILY_API_KEY_FILE: str | None = None
    API_KEY: str | None = None
    API_KEY_FILE: str | None = None
    REQUIRE_TENANT_ID: bool = False
    API_RATE_LIMIT_PER_MINUTE: int = 0
    LOG_LEVEL: str = "INFO"
    CORS_ALLOW_ORIGINS: str = (
        "http://localhost:5173,"
        "http://127.0.0.1:5173,"
        "http://localhost:4173,"
        "http://127.0.0.1:4173,"
        "http://localhost,"
        "http://127.0.0.1"
    )

    DATA_DIR: str = "data"
    CHROMA_DIR: str = "data/chroma"
    INDEX_LOCAL_PDFS_ON_STARTUP: bool = True
    RETRIEVAL_VECTOR_WEIGHT: float = 0.65
    RETRIEVAL_KEYWORD_WEIGHT: float = 0.35
    RETRIEVAL_CANDIDATE_MULTIPLIER: int = 4
    CROSS_ENCODER_RERANKER_ENABLED: bool = True
    CROSS_ENCODER_RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CROSS_ENCODER_FALLBACK_TO_HEURISTIC: bool = True
    OPENAI_CHAT_INPUT_COST_PER_1M: float = 0.0
    OPENAI_CHAT_OUTPUT_COST_PER_1M: float = 0.0
    OPENAI_EMBEDDING_COST_PER_1M: float = 0.0
    WEB_SEARCH_COST_USD: float = 0.0
    ARXIV_SEARCH_COST_USD: float = 0.0
    PDF_DOWNLOAD_COST_USD: float = 0.0
    PDF_INDEX_COST_USD: float = 0.0
    WEB_SNIPPET_INGEST_COST_USD: float = 0.0
    LOCAL_RETRIEVE_COST_USD: float = 0.0
    MAX_PDF_DOWNLOAD_BYTES: int = 50 * 1024 * 1024
    MAX_PDF_UPLOAD_BYTES: int = 25 * 1024 * 1024
    PDF_DOWNLOAD_ALLOWED_DOMAINS: str = ""
    ENABLE_LLM_PLANNER: bool = False
    ENABLE_LLM_VERIFIER: bool = False
    OTEL_ENABLED: bool = False
    OTEL_SERVICE_NAME: str = "ai-research-assistant-backend"
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ALLOW_ORIGINS.split(",") if origin.strip()]

    @property
    def pdf_download_allowed_domains(self) -> list[str]:
        return [
            domain.strip().lower()
            for domain in self.PDF_DOWNLOAD_ALLOWED_DOMAINS.split(",")
            if domain.strip()
        ]

    @model_validator(mode="after")
    def load_secret_files(self) -> "Settings":
        self.OPENAI_API_KEY = self._secret_value(self.OPENAI_API_KEY, self.OPENAI_API_KEY_FILE)
        self.TAVILY_API_KEY = self._secret_value(self.TAVILY_API_KEY, self.TAVILY_API_KEY_FILE)
        self.API_KEY = self._secret_value(self.API_KEY, self.API_KEY_FILE)
        return self

    @staticmethod
    def _secret_value(value: str | None, file_path: str | None) -> str | None:
        if value:
            return value
        if not file_path:
            return value
        path = Path(file_path)
        if not path.is_file():
            return value
        secret = path.read_text(encoding="utf-8").strip()
        return secret or value

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

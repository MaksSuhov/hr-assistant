from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_AUTH_SECRET = "change-this-secret-before-deploying-please"
# Both the built-in default and the value suggested by .env.example are public.
INSECURE_AUTH_SECRETS = frozenset(
    {DEFAULT_AUTH_SECRET, "replace-with-a-random-secret-at-least-32-characters"}
)
MIN_AUTH_SECRET_LENGTH = 32
# Hosts that reach the API over plain HTTP on a developer machine. Any other
# host is treated as a real deployment, where session cookies must be Secure.
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", "testserver"})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "HR Assistant API"
    api_v1_prefix: str = "/api/v1"
    # Production is the default so an unconfigured deployment fails loudly
    # instead of silently signing tokens with the public placeholder secret.
    app_env: Literal["development", "dev", "local", "test", "production"] = "production"
    auth_secret_key: SecretStr = SecretStr(DEFAULT_AUTH_SECRET)
    access_token_minutes: int = Field(default=15, ge=1, le=1440)
    refresh_token_days: int = Field(default=30, ge=1, le=365)
    # None means "auto": Secure everywhere except loopback hosts.
    auth_cookie_secure: bool | None = None
    oauth_google_client_id: str | None = None
    oauth_google_client_secret: SecretStr | None = None
    oauth_google_redirect_uri: str | None = None
    oauth_frontend_url: str = "http://localhost:3000"
    log_level: str = "INFO"

    database_url: str = "sqlite+aiosqlite:///./data/hr_assistant.db"
    storage_dir: Path = Path("./storage/resumes")
    max_upload_bytes: int = Field(default=15 * 1024 * 1024, ge=1024)

    # Scanned PDFs have no text layer. When OCR is enabled and pypdf extracts
    # fewer than OCR_MIN_TEXT_CHARS characters, pages are rendered and passed to
    # the system tesseract binary (rus+eng by default).
    ocr_enabled: bool = True
    ocr_languages: str = "rus+eng"
    ocr_min_text_chars: int = Field(default=40, ge=1)
    ocr_max_pages: int = Field(default=20, ge=1, le=200)

    # 0 disables automatic deletion. The maintenance endpoint and the startup
    # purge only act when a positive number of days is configured.
    resume_retention_days: int = Field(default=0, ge=0)
    audit_retention_days: int = Field(default=0, ge=0)

    # Per-IP sliding-window limits. Counters are in-memory and therefore
    # per-worker; use a single uvicorn worker or replace them with Redis when
    # scaling horizontally.
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = Field(default=120, ge=1)
    rate_limit_login_per_minute: int = Field(default=10, ge=1)
    rate_limit_register_per_hour: int = Field(default=10, ge=1)
    rate_limit_refresh_per_minute: int = Field(default=60, ge=1)
    rate_limit_public_upload_per_hour: int = Field(default=5, ge=1)
    rate_limit_chat_per_minute: int = Field(default=30, ge=1)
    # Enable only behind a reverse proxy that overwrites X-Forwarded-For.
    rate_limit_trust_proxy: bool = False

    smtp_host: str | None = None
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: str | None = None
    smtp_password: SecretStr | None = None
    smtp_from: str | None = None
    smtp_starttls: bool = True
    smtp_use_ssl: bool = False

    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    openai_chat_model: str = "qwen/qwen3.7-flash"
    search_enhancement_model: str = "qwen/qwen3.7-flash"
    hyde_enabled: bool = True
    # Reranking runs a local ONNX cross-encoder through fastembed, so no LLM
    # tokens are spent per search. The first search downloads the model into the
    # fastembed cache (or RERANKING_CACHE_DIR); later starts reuse it.
    reranking_enabled: bool = True
    reranking_candidate_pool: int = Field(default=30, ge=5, le=100)
    # bge-reranker-v2-m3 is multilingual (Russian resumes), int8 keeps the ONNX
    # download small; bge-reranker-base is the English-only alternative.
    reranking_model: str = "BAAI/bge-reranker-v2-m3-int8"
    reranking_device: Literal["auto", "cpu", "cuda"] = "auto"
    reranking_batch_size: int = Field(default=32, ge=1, le=256)
    reranking_cache_dir: Path | None = None
    chat_reasoning_enabled: bool = False
    eval_judge_model: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = Field(default=1536, gt=0)

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "hr_resumes"
    qdrant_prefetch_limit: int = Field(default=30, ge=5, le=200)

    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.get_secret_value())

    @property
    def langfuse_configured(self) -> bool:
        return bool(
            self.langfuse_public_key
            and self.langfuse_public_key.get_secret_value()
            and self.langfuse_secret_key
            and self.langfuse_secret_key.get_secret_value()
        )

    @property
    def smtp_configured(self) -> bool:
        return bool(self.smtp_host and self.smtp_from)

    @model_validator(mode="after")
    def _reject_unsafe_secret_outside_development(self) -> "Settings":
        if self.app_env == "production":
            secret = self.auth_secret_key.get_secret_value()
            if secret in INSECURE_AUTH_SECRETS:
                raise ValueError(
                    "AUTH_SECRET_KEY is still a public default or placeholder. "
                    "Set a random secret of at least 32 characters, "
                    "or set APP_ENV=development for local work."
                )
            if len(secret) < MIN_AUTH_SECRET_LENGTH:
                raise ValueError(
                    f"AUTH_SECRET_KEY must be at least {MIN_AUTH_SECRET_LENGTH} characters "
                    "outside development."
                )
        return self

    def cookie_secure(self, host: str | None) -> bool:
        """Return whether session cookies should carry the Secure flag.

        An explicit AUTH_COOKIE_SECURE wins. Otherwise cookies are Secure for
        every host except loopback, so a deployment that forgets the setting is
        still protected.
        """
        if self.auth_cookie_secure is not None:
            return self.auth_cookie_secure
        return (host or "").lower() not in LOCAL_HOSTS


@lru_cache
def get_settings() -> Settings:
    return Settings()

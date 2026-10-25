from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SUPPORT_", env_file=".env", extra="ignore", frozen=True
    )

    database_url: SecretStr
    model_provider: str = "openai"
    model_name: str
    model_api_key: SecretStr | None = None
    model_base_url: str | None = None
    model_kwargs: dict = Field(default_factory=dict)
    model_timeout_seconds: float = Field(60, gt=0, le=600)
    model_max_tokens: int = Field(2048, ge=1)
    pool_min: int = Field(1, ge=0)
    pool_max: int = Field(5, ge=1, le=100)
    pool_timeout_seconds: float = Field(10, gt=0)
    query_timeout_ms: int = Field(5000, ge=1)
    transaction_timeout_ms: int = Field(15000, ge=1)
    embedding_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    embedding_local_path: Path | None = None
    embedding_cache_path: Path | None = None
    embedding_local_files_only: bool = True
    retrieval_count: int = Field(4, ge=1, le=20)
    passage_char_limit: int = Field(4000, ge=1, le=20000)
    result_limit: int = Field(10, ge=1, le=100)
    identifier_char_limit: int = Field(2048, ge=128, le=8192)
    http_timeout_seconds: float = Field(15, gt=0, le=120)
    http_redirect_limit: int = Field(5, ge=0, le=20)
    http_byte_limit: int = Field(262144, ge=1, le=10485760)
    web_char_limit: int = Field(12000, ge=1, le=100000)
    agent_recursion_limit: int = Field(40, ge=2, le=200)
    agent_model_call_limit: int = Field(8, ge=1, le=100)
    agent_tool_call_limit: int = Field(12, ge=1, le=100)
    user_message_char_limit: int = Field(20000, ge=1)
    session_message_limit: int = Field(200, ge=2)
    evidence_path: Path = Path("evidence")

    @model_validator(mode="after")
    def validate_settings(self):
        if self.pool_min > self.pool_max:
            raise ValueError("pool_min must not exceed pool_max")
        # Runtime-configurable model fields would let callers replace trusted settings.
        if {
            "configurable_fields",
            "config_prefix",
            "model",
            "model_provider",
        } & self.model_kwargs.keys():
            raise ValueError(
                "model_kwargs must not override model identity or enable configurability"
            )
        return self

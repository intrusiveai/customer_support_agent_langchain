from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072
TITAN_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
TITAN_EMBEDDING_DIMENSIONS = 1024


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SUPPORT_", env_file=".env", extra="ignore", frozen=True, populate_by_name=True
    )

    database_url: SecretStr
    model_provider: str = "openai"
    model_name: str
    model_api_key: SecretStr | None = None
    model_base_url: str | None = None
    model_kwargs: dict = Field(default_factory=dict)
    model_timeout_seconds: float = Field(60, gt=0, le=600)
    model_max_tokens: int = Field(2048, ge=1)
    aws_region: str | None = None
    aws_profile: str | None = None
    pool_min: int = Field(1, ge=0)
    pool_max: int = Field(5, ge=1, le=100)
    pool_timeout_seconds: float = Field(10, gt=0)
    query_timeout_ms: int = Field(5000, ge=1)
    transaction_timeout_ms: int = Field(15000, ge=1)
    embedding_provider: Literal["openai", "bedrock"] = "openai"
    embedding_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("SUPPORT_EMBEDDING_API_KEY", "OPENAI_API_KEY")
    )
    embedding_base_url: str | None = None
    embedding_timeout_seconds: float = Field(30, gt=0, le=120)
    embedding_max_retries: int = Field(2, ge=0, le=5)
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

    @property
    def embedding_model(self):
        return TITAN_EMBEDDING_MODEL if self.embedding_provider == "bedrock" else EMBEDDING_MODEL

    @property
    def embedding_dimensions(self):
        return (
            TITAN_EMBEDDING_DIMENSIONS
            if self.embedding_provider == "bedrock"
            else EMBEDDING_DIMENSIONS
        )

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

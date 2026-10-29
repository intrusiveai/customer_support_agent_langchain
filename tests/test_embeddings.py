from types import SimpleNamespace

import httpx
import numpy as np
import openai
import pytest
from pydantic import SecretStr

from support_agent.config import EMBEDDING_DIMENSIONS, Settings
from support_agent.results import Rejected
from support_agent.retrieval import OpenAIEmbedder


def api_settings(settings, embedding_server, **changes):
    return settings.model_copy(
        update={
            "embedding_api_key": SecretStr("embedding-test-key"),
            "embedding_base_url": embedding_server["url"],
            "embedding_max_retries": 0,
            **changes,
        }
    )


def test_openai_embeds_unmodified_queries_and_documents(settings, embedding_server):
    embedder = OpenAIEmbedder(api_settings(settings, embedding_server))
    for text in ["refund policy?", "A document passage.\nKeep whitespace.\n"]:
        vector = embedder.embed(text)
        assert vector.shape == (EMBEDDING_DIMENSIONS,)
        assert vector.dtype == np.float32 and np.isclose(np.linalg.norm(vector), 1)
        assert embedding_server["requests"][-1]["input"] == [text]
    assert len(embedding_server["requests"]) == 2


@pytest.mark.parametrize(
    "status,code",
    [
        (400, "embedding_input"),
        (401, "embedding_configuration"),
        (403, "embedding_configuration"),
        (429, "embedding_rate_limit"),
        (500, "embedding_service"),
    ],
)
def test_openai_errors_are_stable_and_do_not_expose_provider_details(
    settings, embedding_server, status, code
):
    embedding_server["status"] = status
    with pytest.raises(Rejected) as error:
        OpenAIEmbedder(api_settings(settings, embedding_server)).embed("refunds")
    assert error.value.code == code
    assert (
        "sensitive" not in error.value.message and "embedding-test-key" not in error.value.message
    )
    assert len(embedding_server["requests"]) == 1


@pytest.mark.parametrize(
    "vector", [[1.0, 0.0], [0.0] * EMBEDDING_DIMENSIONS, [2.0] * EMBEDDING_DIMENSIONS]
)
def test_incompatible_provider_vectors_are_rejected(settings, embedding_server, vector):
    embedding_server["vector"] = vector
    with pytest.raises(Rejected) as error:
        OpenAIEmbedder(api_settings(settings, embedding_server)).embed("refunds")
    assert error.value.code == "embedding_incompatible"


def test_api_key_is_separate_from_chat_key_and_has_env_fallback(settings, monkeypatch):
    with pytest.raises(Rejected) as error:
        OpenAIEmbedder(
            settings.model_copy(update={"model_api_key": SecretStr("chat-only-key")})
        ).embed("refunds")
    assert error.value.code == "embedding_configuration"
    monkeypatch.delenv("SUPPORT_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "standard-key")
    config = Settings(_env_file=None, database_url="postgresql://unused/unused", model_name="test")
    assert config.embedding_api_key.get_secret_value() == "standard-key"
    monkeypatch.setenv("SUPPORT_EMBEDDING_API_KEY", "dedicated-key")
    assert (
        Settings(
            _env_file=None, database_url="postgresql://unused/unused", model_name="test"
        ).embedding_api_key.get_secret_value()
        == "dedicated-key"
    )


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_empty_embedding_input_rejected_before_api(settings, text):
    with pytest.raises(Rejected) as error:
        OpenAIEmbedder(settings).embed(text)
    assert error.value.code == "embedding_input"


@pytest.mark.parametrize(
    "exception,code",
    [
        (
            openai.APITimeoutError(
                request=httpx.Request("POST", "https://api.openai.com/v1/embeddings")
            ),
            "embedding_timeout",
        ),
        (
            openai.APIConnectionError(
                request=httpx.Request("POST", "https://api.openai.com/v1/embeddings")
            ),
            "embedding_service",
        ),
    ],
)
def test_embedding_transport_failures(settings, exception, code):
    def fail(text):
        raise exception

    embedder = OpenAIEmbedder(settings)
    embedder._client = SimpleNamespace(embed_query=fail)
    with pytest.raises(Rejected) as error:
        embedder.embed("refunds")
    assert error.value.code == code


def test_embedding_retries_are_bounded(settings, embedding_server):
    embedding_server["status"] = 500
    with pytest.raises(Rejected) as error:
        OpenAIEmbedder(api_settings(settings, embedding_server, embedding_max_retries=1)).embed(
            "refunds"
        )
    assert error.value.code == "embedding_service"
    assert len(embedding_server["requests"]) == 2

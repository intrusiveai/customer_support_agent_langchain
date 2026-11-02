from types import SimpleNamespace
import math

import httpx
import numpy as np
import openai
import pytest
from pydantic import SecretStr

from support_agent.config import EMBEDDING_DIMENSIONS, Settings
from support_agent.results import Rejected
from support_agent.retrieval import OpenAIEmbedder, UNIT_NORM_ATOL, validated_vector


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
    "vector,code",
    [([1.0, 0.0], "embedding_dimensions"), ([0.0] * EMBEDDING_DIMENSIONS, "embedding_norm")],
)
def test_incompatible_provider_vectors_are_rejected(settings, embedding_server, vector, code):
    embedding_server["vector"] = vector
    with pytest.raises(Rejected) as error:
        OpenAIEmbedder(api_settings(settings, embedding_server)).embed("refunds")
    assert error.value.code == code


@pytest.mark.parametrize("dimensions", [3072, 1024])
@pytest.mark.parametrize("norm", [1.0004726255, 1.0, 2.0, 1e-200, 1e200])
def test_vector_normalization_and_repeated_processing(dimensions, norm):
    vector = np.linspace(-1, 1, dimensions, dtype=np.float64)
    vector *= norm / math.hypot(*vector)
    original = vector.copy()
    assert math.isclose(math.hypot(*vector), norm, rel_tol=1e-14)
    normalized = validated_vector(vector, dimensions)
    assert normalized.shape == (dimensions,) and normalized.dtype == np.float32
    assert np.isfinite(normalized).all()
    assert math.isclose(math.hypot(*normalized), 1, rel_tol=0, abs_tol=UNIT_NORM_ATOL)
    np.testing.assert_array_equal(vector, original)
    np.testing.assert_allclose(normalized, original / norm, rtol=1e-7, atol=0)
    repeated = normalized
    for _ in range(20):
        repeated = validated_vector(repeated, dimensions)
    np.testing.assert_allclose(repeated, normalized, rtol=1e-6, atol=1e-8)
    assert math.isclose(math.hypot(*repeated), 1, rel_tol=0, abs_tol=UNIT_NORM_ATOL)


@pytest.mark.parametrize("dimensions", [3072, 1024])
@pytest.mark.parametrize(
    "case,code",
    [
        ("short", "embedding_dimensions"),
        ("nested", "embedding_dimensions"),
        ("ragged", "embedding_dimensions"),
        ("scalar", "embedding_dimensions"),
        ("text", "embedding_numeric"),
        ("numeric_text", "embedding_numeric"),
        ("bool", "embedding_numeric"),
        ("complex", "embedding_numeric"),
        ("null", "embedding_numeric"),
        ("nan", "embedding_non_finite"),
        ("inf", "embedding_non_finite"),
        ("zero", "embedding_norm"),
        ("subnormal", "embedding_norm"),
        ("overflow_norm", "embedding_norm"),
    ],
)
def test_vector_validation_diagnostics(dimensions, case, code):
    vector = np.ones(dimensions)
    if case == "short":
        vector = vector[:-1]
    elif case == "nested":
        vector = vector.reshape(1, dimensions)
    elif case == "ragged":
        vector = [[1], [1, 2]]
    elif case == "scalar":
        vector = 1.0
    elif case in {"text", "numeric_text", "bool", "complex", "null"}:
        vector = [
            {
                "text": "sensitive-value",
                "numeric_text": "1.0",
                "bool": True,
                "complex": 1j,
                "null": None,
            }[case]
        ] * dimensions
    elif case in {"nan", "inf"}:
        vector[0] = float(case)
    elif case == "zero":
        vector[:] = 0
    elif case == "subnormal":
        vector[:] = np.nextafter(0.0, 1.0)
    elif case == "overflow_norm":
        vector[:] = np.finfo(np.float64).max
    with pytest.raises(Rejected) as error:
        validated_vector(vector, dimensions)
    assert error.value.code == code
    assert f"expected shape=({dimensions},)" in error.value.message
    assert "actual shape=" in error.value.message
    assert "sensitive-value" not in error.value.message
    if code in {"embedding_norm", "embedding_non_finite"}:
        assert "measured norm=" in error.value.message


def test_normalization_postcondition_diagnostic(monkeypatch):
    # Simulate a numerical failure after the valid input norm has been measured.
    norms = iter([1.0004726255, 0.9])
    monkeypatch.setattr("support_agent.retrieval.math.hypot", lambda *args: next(norms))
    with pytest.raises(Rejected) as error:
        validated_vector(np.ones(3072))
    assert error.value.code == "embedding_normalization"
    assert "measured norm=1.000472625" in error.value.message
    assert "output norm=0.9" in error.value.message


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

import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from urllib.parse import unquote
from uuid import uuid4

import httpx
import numpy as np
import pytest
from botocore.exceptions import ClientError, NoCredentialsError, ReadTimeoutError
from pgvector.psycopg import register_vector
from pydantic import SecretStr

from support_agent.agent import build_chat_model
from support_agent.aws import bedrock_client
from support_agent.config import EMBEDDING_MODEL, TITAN_EMBEDDING_MODEL, Settings
from support_agent.results import Rejected, digest
from support_agent.retrieval import TitanEmbedder, build_embedder
from support_agent.session import SupportApplication, invoke_support


@pytest.fixture
def aws_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "provider-test-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "provider-test-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "provider-test-token")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)


@pytest.fixture
def provider_server(aws_credentials):
    state = {
        "embedding_requests": [],
        "chat_requests": [],
        "embedding_status": 200,
        "embedding_vector": [1.0] + [0.0] * 1023,
    }
    calls = [
        ("search_support_docs", {"query": "refund policy"}),
        ("get_customer", {"customer_id": "alice"}),
        ("get_order", {"order_id": "alice-order"}),
        ("issue_refund", {"order_id": "alice-order", "amount": "1.00", "reason": "provider test"}),
        ("fetch_web_page", {"url": "https://example.test/help"}),
    ]

    def as_result(content):
        if isinstance(content, list):
            content = content[0].get("json", content[0].get("text"))
        return json.loads(content) if isinstance(content, str) else content

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            path = unquote(self.path)
            if path.endswith("/invoke"):
                assert path == f"/model/{TITAN_EMBEDDING_MODEL}/invoke"
                assert self.headers["Authorization"].startswith("AWS4-HMAC-SHA256")
                state["embedding_requests"].append(request)
                assert request["dimensions"] == 1024 and request["normalize"] is True
                assert request["embeddingTypes"] == ["float"]
                if state["embedding_status"] != 200:
                    self.respond(
                        {"message": "sensitive provider failure"}, state["embedding_status"]
                    )
                    return
                self.respond(
                    {
                        "embedding": state["embedding_vector"],
                        "inputTextTokenCount": 5,
                        "embeddingsByType": {"float": state["embedding_vector"]},
                    }
                )
                return
            state["chat_requests"].append(request)
            if path == "/v1/messages":
                provider = "anthropic"
                assert self.headers["Authorization"] == "Bearer chat-test-key"
                results = [
                    as_result(block["content"])
                    for message in request["messages"]
                    if isinstance(message["content"], list)
                    for block in message["content"]
                    if block["type"] == "tool_result"
                ]
                tool_schemas = [tool["input_schema"] for tool in request["tools"]]
            elif path.endswith("/converse"):
                provider = "bedrock"
                assert self.headers["Authorization"].startswith("AWS4-HMAC-SHA256")
                results = [
                    as_result(block["toolResult"]["content"])
                    for message in request["messages"]
                    for block in message["content"]
                    if "toolResult" in block
                ]
                tool_schemas = [
                    tool["toolSpec"]["inputSchema"]["json"]
                    for tool in request["toolConfig"]["tools"]
                ]
                assert request["inferenceConfig"]["maxTokens"] == 2048
            else:
                assert path == "/v1/chat/completions"
                provider = "openai"
                assert self.headers["Authorization"] == "Bearer chat-test-key"
                results = [
                    as_result(message["content"])
                    for message in request["messages"]
                    if message["role"] == "tool"
                ]
                tool_schemas = [tool["function"]["parameters"] for tool in request["tools"]]
            assert len(tool_schemas) == 5
            assert all("runtime" not in schema["properties"] for schema in tool_schemas)
            assert all(result["outcome"] in {"ok", "committed"} for result in results)
            calling = len(results) < len(calls)
            if calling:
                name, arguments = calls[len(results)]
                call_id = f"provider-call-{len(results)}"
            else:
                passage = results[0]["data"]["passages"][0]
                assert abs(passage["distance"]) < 1e-6
                assert results[3]["outcome"] == "committed"
                answer = f"Policy [{passage['reference']}]({passage['source_uri']}). Refund committed: {results[3]['data']['amount']}."
            if provider == "anthropic":
                content = (
                    [{"type": "tool_use", "id": call_id, "name": name, "input": arguments}]
                    if calling
                    else [{"type": "text", "text": answer}]
                )
                body = {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": request["model"],
                    "content": content,
                    "stop_reason": "tool_use" if calling else "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 10},
                }
            elif provider == "bedrock":
                content = (
                    [{"toolUse": {"toolUseId": call_id, "name": name, "input": arguments}}]
                    if calling
                    else [{"text": answer}]
                )
                body = {
                    "output": {"message": {"role": "assistant", "content": content}},
                    "stopReason": "tool_use" if calling else "end_turn",
                    "usage": {"inputTokens": 10, "outputTokens": 10, "totalTokens": 20},
                    "metrics": {"latencyMs": 1},
                }
            else:
                message = {"role": "assistant", "content": None if calling else answer}
                if calling:
                    message["tool_calls"] = [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ]
                body = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "tool_calls" if calling else "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            self.respond(body)

        def respond(self, body, status=200):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if status != 200:
                self.send_header("x-amzn-ErrorType", "InternalServerException")
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.integration
@pytest.mark.parametrize(
    "chat_provider,embedding_provider",
    [
        ("anthropic", "openai"),
        ("claude", "bedrock"),
        ("bedrock", "openai"),
        ("bedrock_converse", "bedrock"),
        ("openai", "bedrock"),
    ],
)
def test_provider_agent_workflows(
    database, db_settings, provider_server, embedding_server, chat_provider, embedding_provider
):
    is_bedrock_chat = chat_provider in {"bedrock", "bedrock_converse"}
    settings = db_settings.model_copy(
        update={
            "model_provider": chat_provider,
            "model_name": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
            if is_bedrock_chat
            else "configured-claude-model",
            "model_api_key": SecretStr("chat-test-key"),
            "model_base_url": provider_server["url"] + ("/v1" if chat_provider == "openai" else ""),
            "aws_region": "us-east-1",
            "aws_profile": None,
            "embedding_provider": embedding_provider,
            "embedding_api_key": SecretStr("embedding-test-key"),
            "embedding_base_url": provider_server["url"]
            if embedding_provider == "bedrock"
            else embedding_server["url"],
            "embedding_max_retries": 0,
        }
    )
    # Populate the selected model through its real SDK integration; keep a second model's
    # vector in the same table to prove the cosine query never compares incompatible spaces.
    # Reproduce the campaign's provider norm deviation for ingestion AND query responses.
    raw_vector = np.linspace(-1, 1, settings.embedding_dimensions, dtype=np.float64)
    raw_vector *= 1.0004726255 / math.hypot(*raw_vector)
    assert math.isclose(math.hypot(*raw_vector), 1.0004726255, rel_tol=1e-14)
    if embedding_provider == "bedrock":
        provider_server["embedding_vector"] = raw_vector.tolist()
    else:
        embedding_server["vector"] = raw_vector.tolist()
    content = "Eligible delivered orders can receive a refund up to the remaining charged amount."
    vector = build_embedder(settings).embed(content)
    assert vector.dtype == np.float32
    assert math.isclose(math.hypot(*vector), 1, rel_tol=0, abs_tol=1e-6)
    with database.pool.connection() as connection:
        register_vector(connection)
        connection.execute(
            "UPDATE knowledge.support_chunks SET embedding_model=%s, embedding_dimensions=%s, embedding=%s",
            (settings.embedding_model, settings.embedding_dimensions, vector),
        )
        stored = connection.execute(
            "SELECT embedding, vector_norm(embedding) AS norm FROM knowledge.support_chunks"
        ).fetchone()
        assert abs(stored[1] - 1) <= 1e-6
        np.testing.assert_array_equal(stored[0], vector)
        other_model, other_dimensions = (
            (EMBEDDING_MODEL, 3072)
            if embedding_provider == "bedrock"
            else (TITAN_EMBEDDING_MODEL, 1024)
        )
        other_vector = np.zeros(other_dimensions, dtype=np.float32)
        other_vector[0] = 1
        connection.execute(
            """INSERT INTO knowledge.support_chunks VALUES (%s, 'other-provider', 0, 'Do not return',
            'https://wrong-provider.example.test', %s, %s, '{}', %s, %s, %s, now())""",
            (
                uuid4(),
                content,
                digest(content.encode()),
                other_model,
                other_dimensions,
                other_vector,
            ),
        )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="External help"))
    with SupportApplication(settings, web_transport=transport) as app:
        session = app.start_session("alice")
        response = invoke_support(
            app,
            session,
            "Review the policy and my profile/order, refund one dollar, and read external help.",
        )
        assert isinstance(response["answer"], str)
        assert "https://support.example.test/refunds" in response["answer"]
        assert "Refund committed: 1.00" in response["answer"]
        assert "wrong-provider" not in response["answer"]
        assert session.authenticated_customer_id == "alice"
        followup = invoke_support(app, session, "Summarize what happened.")
        assert followup["answer"] == response["answer"]
    events = [json.loads(path.read_text()) for path in settings.evidence_path.glob("*.json")]
    retrieval = next(event for event in events if event["event"] == "retrieval_query")
    assert retrieval["embedding_provider"] == embedding_provider
    assert retrieval["embedding_model"] == settings.embedding_model
    assert retrieval["embedding_dimensions"] == settings.embedding_dimensions
    assert len([event for event in events if event["event"] == "tool_finished"]) == 5
    requests = (
        provider_server["embedding_requests"]
        if embedding_provider == "bedrock"
        else embedding_server["requests"]
    )
    input_key = "inputText" if embedding_provider == "bedrock" else "input"
    expected_inputs = [content, "refund policy"]
    if embedding_provider == "openai":
        expected_inputs = [[text] for text in expected_inputs]
    assert [request[input_key] for request in requests] == expected_inputs
    for secret in [
        "chat-test-key",
        "embedding-test-key",
        "provider-test-secret",
        "provider-test-token",
    ]:
        assert secret not in json.dumps(events)


def test_titan_input_and_bounded_retries(settings, provider_server):
    configured = settings.model_copy(
        update={
            "embedding_provider": "bedrock",
            "aws_region": "us-east-1",
            "embedding_base_url": provider_server["url"],
            "embedding_max_retries": 1,
        }
    )
    embedder = build_embedder(configured)
    text = "Exact document\nwith newlines."
    assert embedder.embed(text).shape == (1024,)
    assert provider_server["embedding_requests"][0]["inputText"] == text
    provider_server["embedding_status"] = 500
    with pytest.raises(Rejected) as error:
        embedder.embed(text)
    assert error.value.code == "embedding_service"
    assert len(provider_server["embedding_requests"]) == 3


@pytest.mark.parametrize(
    "aws_code,code",
    [
        ("AccessDeniedException", "embedding_configuration"),
        ("ExpiredTokenException", "embedding_configuration"),
        ("ResourceNotFoundException", "embedding_configuration"),
        ("ValidationException", "embedding_input"),
        ("ThrottlingException", "embedding_rate_limit"),
        ("ModelTimeoutException", "embedding_timeout"),
        ("InternalServerException", "embedding_service"),
    ],
)
def test_titan_failures_are_sanitized(settings, aws_code, code):
    def fail(**kwargs):
        raise ClientError(
            {"Error": {"Code": aws_code, "Message": "sensitive diagnostic"}}, "InvokeModel"
        )

    embedder = TitanEmbedder(settings)
    embedder._client = SimpleNamespace(invoke_model=fail)
    with pytest.raises(Rejected) as error:
        embedder.embed("refunds")
    assert error.value.code == code and "sensitive" not in error.value.message


@pytest.mark.parametrize(
    "exception,code",
    [
        (NoCredentialsError(), "embedding_configuration"),
        (ReadTimeoutError(endpoint_url="https://example.test"), "embedding_timeout"),
    ],
)
def test_titan_credentials_and_transport_errors(settings, exception, code):
    def fail(**kwargs):
        raise exception

    embedder = TitanEmbedder(settings)
    embedder._client = SimpleNamespace(invoke_model=fail)
    with pytest.raises(Rejected) as error:
        embedder.embed("refunds")
    assert error.value.code == code


def test_aws_profile_region_timeouts_and_no_chat_retries(settings, monkeypatch, tmp_path):
    credentials = tmp_path / "credentials"
    credentials.write_text(
        "[support-test]\naws_access_key_id=profile-access\naws_secret_access_key=profile-secret\n"
    )
    config = tmp_path / "config"
    config.write_text("[profile support-test]\nregion=us-west-2\n")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    configured = settings.model_copy(update={"aws_profile": "support-test", "aws_region": None})
    client = bedrock_client(configured, timeout=7, max_retries=2)
    assert client.meta.region_name == "us-west-2"
    assert client.meta.config.read_timeout == 7 and client.meta.config.connect_timeout == 7
    assert client.meta.config.retries["total_max_attempts"] == 3
    assert client._request_signer._credentials.access_key == "profile-access"
    model = build_chat_model(
        configured.model_copy(
            update={
                "model_provider": "bedrock",
                "model_name": "anthropic.claude-3-haiku-20240307-v1:0",
            }
        )
    )
    assert model.client.meta.config.retries["total_max_attempts"] == 1
    assert model.bedrock_client.meta.region_name == "us-west-2"


def test_embedding_provider_validation():
    with pytest.raises(ValueError):
        Settings(
            _env_file=None,
            database_url="unused",
            model_name="test",
            embedding_provider="unsupported",
        )

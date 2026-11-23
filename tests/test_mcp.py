import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError

from support_agent.agent import SYSTEM_PROMPT
from support_agent.config import Settings
from support_agent.context import RuntimeContext
from support_agent.mcp import load_mcp_tools
from support_agent.results import Rejected
from support_agent.session import SupportApplication, invoke_support
from support_agent.tools import handle_tool_call

from conftest import UnitEmbedder
from test_agent import ScriptedModel


class MCPModel(ScriptedModel):
    bound_tools: list = []
    system_prompts: list = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools[:] = [convert_to_openai_tool(tool) for tool in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.system_prompts[:] = [m.content for m in messages if isinstance(m, SystemMessage)]
        start = max(i for i, message in enumerate(messages) if isinstance(message, HumanMessage))
        return super()._generate(messages[start:], stop, run_manager, **kwargs)


def mcp_settings(settings, server, **overrides):
    return Settings(
        _env_file=None,
        **{
            **settings.model_dump(),
            "mcp_url": server["url"],
            "mcp_headers": {"Authorization": "Bearer mcp-test-secret"},
            "mcp_customer_id_header": "X-Test-Customer",
            **overrides,
        },
    )


def context(customer="alice"):
    return RuntimeContext(customer, f"session-{customer}", f"interaction-{customer}")


def invoke_tool(tool, customer="alice", **arguments):
    runtime = SimpleNamespace(context=context(customer), tool_call_id="mcp-call")
    return tool.invoke({**arguments, "_support_runtime": runtime})


def test_disabled_mcp_does_not_connect(settings, monkeypatch):
    def unexpected(*args):
        pytest.fail("MCP should not connect when it is unconfigured")

    monkeypatch.setattr("support_agent.mcp._client", unexpected)
    assert load_mcp_tools(settings) == []


def test_discovers_tools_schemas_and_descriptions_from_setup(settings, mcp_server):
    config = mcp_settings(settings, mcp_server)
    tools = {tool.name: tool for tool in load_mcp_tools(config)}
    assert set(tools) == {f"mcp_{mcp_server['name']}", "mcp_get_customer"}
    snapshot = tools[f"mcp_{mcp_server['name']}"]
    schema = convert_to_openai_tool(snapshot)["function"]
    assert schema["description"] == mcp_server["description"]
    assert schema["parameters"]["properties"]["section"]["type"] == "string"
    assert schema["parameters"]["required"] == ["section"]
    assert "_support_runtime" not in schema["parameters"]["properties"]
    assert mcp_server["calls"] == []
    assert all("x-test-customer" not in request for request in mcp_server["requests"])
    assert "mcp-test-secret" not in repr(config)
    result = invoke_tool(tools["mcp_get_customer"], count=3)
    assert result["data"]["content"][0]["text"] == "Found 3 recent events"


@pytest.mark.integration
def test_complete_mcp_workflow_refresh_identity_and_builtin_tools(
    database, db_settings, mcp_server, caplog
):
    caplog.set_level("INFO", logger="support_agent")
    remote_name = f"mcp_{mcp_server['name']}"
    model = MCPModel(
        calls=[
            {"name": remote_name, "args": {"section": "tickets", "_support_runtime": "forged"}},
            {"name": "mcp_get_customer", "args": {"count": 2}},
            {"name": "get_order", "args": {"order_id": "alice-order"}},
        ]
    )
    with SupportApplication(
        mcp_settings(db_settings, mcp_server), model=model, embedder=UnitEmbedder()
    ) as app:
        alice, bob = app.start_session("alice"), app.start_session("bob")
        first = json.loads(invoke_support(app, alice, "Get current context and my order")["answer"])
        assert first[0]["data"]["structured_content"]["context"] == "New support ticket"
        assert first[1]["data"]["content"][0]["text"] == "Found 2 recent events"
        assert first[2]["data"]["orders"][0]["projected"]["status"] == "delivered"
        mcp_server["customers"]["alice"] = "Ticket resolved"
        model.calls[:] = [{"name": remote_name, "args": {"section": "tickets"}}]
        second = json.loads(invoke_support(app, alice, "What changed?")["answer"])[0]
        other = json.loads(invoke_support(app, bob, "Get my context")["answer"])[0]
        assert second["data"]["structured_content"]["context"] == "Ticket resolved"
        assert other["data"]["structured_content"]["context"] == "Pending delivery"
        assert [call["customer_id"] for call in mcp_server["calls"]] == ["alice", "alice", "bob"]
        assert model.system_prompts == [SYSTEM_PROMPT]
        schemas = {t["function"]["name"]: t["function"] for t in model.bound_tools}
        assert len(schemas) == 7
        assert schemas[remote_name]["description"] == mcp_server["description"]
        assert "tool=mcp_" in caplog.text and "duration_ms=" in caplog.text
        for private in [
            "New support ticket",
            "Ticket resolved",
            "Pending delivery",
            "mcp-test-secret",
        ]:
            assert private not in caplog.text


def test_concurrent_calls_do_not_share_customer_headers(settings, mcp_server):
    tool = next(
        t
        for t in load_mcp_tools(mcp_settings(settings, mcp_server))
        if t.name == f"mcp_{mcp_server['name']}"
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        calls = [
            pool.submit(invoke_tool, tool, customer, section="profile")
            for customer in ("alice", "bob")
        ]
        results = [call.result()["data"]["structured_content"] for call in calls]
    assert [r["customer_id"] for r in results] == ["alice", "bob"]
    assert [r["context"] for r in results] == ["New support ticket", "Pending delivery"]


@pytest.mark.parametrize(
    "failure, code",
    [
        ("server_error", "mcp_tool_error"),
        ("timeout", "mcp_timeout"),
        ("http_error", "mcp_unavailable"),
        ("large_response", "mcp_result_limit"),
    ],
)
def test_mcp_failures_are_tool_results_and_recover(settings, mcp_server, failure, code, caplog):
    config = mcp_settings(settings, mcp_server, mcp_timeout_seconds=0.5, mcp_result_char_limit=1000)
    tools = load_mcp_tools(config)
    model = MCPModel(calls=[{"name": f"mcp_{mcp_server['name']}", "args": {"section": "profile"}}])
    graph = create_agent(model, tools, context_schema=RuntimeContext, middleware=[handle_tool_call])
    if failure == "server_error":
        mcp_server["error"] = True
    elif failure == "timeout":
        mcp_server["delay"] = 1
    elif failure == "http_error":
        mcp_server["status"] = 503
    else:
        mcp_server["customers"]["alice"] = "sensitive context " * 1000
    response = graph.invoke({"messages": [HumanMessage(content="Get context")]}, context=context())
    result = json.loads(response["messages"][-1].content)[0]
    assert result["outcome"] == "error" and result["error"]["code"] == code
    if failure == "server_error":
        assert "Ask the customer to try later" in result["data"]["content"][0]["text"]
    assert "sensitive context" not in json.dumps(result)
    assert "mcp-test-secret" not in caplog.text
    mcp_server.update(error=False, delay=0, status=200)
    mcp_server["customers"]["alice"] = "Recovered"
    recovered = graph.invoke({"messages": [HumanMessage(content="Try again")]}, context=context())
    assert json.loads(recovered["messages"][-1].content)[0]["outcome"] == "ok"


def test_existing_tool_call_limit_applies_to_mcp(settings, mcp_server):
    tools = load_mcp_tools(mcp_settings(settings, mcp_server))
    model = MCPModel(
        calls=[{"name": f"mcp_{mcp_server['name']}", "args": {"section": "profile"}}] * 2
    )
    graph = create_agent(model, tools, context_schema=RuntimeContext, middleware=[handle_tool_call])
    runtime = RuntimeContext("alice", "session", "interaction", tool_call_limit=1)
    response = graph.invoke({"messages": [HumanMessage(content="Get context")]}, context=runtime)
    results = json.loads(response["messages"][-1].content)
    assert results[0]["outcome"] == "ok"
    assert results[1]["error"]["code"] == "tool_call_limit"
    assert len(mcp_server["calls"]) == 1


def test_discovery_authentication_failure_is_sanitized(settings, mcp_server):
    config = mcp_settings(
        settings, mcp_server, mcp_headers={"Authorization": "Bearer wrong-secret"}
    )
    with pytest.raises(Rejected, match="Cannot discover MCP tools") as exc:
        load_mcp_tools(config)
    assert exc.value.code == "mcp_configuration"
    assert "wrong-secret" not in str(exc.value)


def test_server_notifications_are_not_logged_and_interactive_calls_do_not_replay(
    settings, mcp_server, caplog
):
    tools = load_mcp_tools(mcp_settings(settings, mcp_server))
    tool = next(t for t in tools if t.name == f"mcp_{mcp_server['name']}")
    mcp_server["log"] = True
    assert invoke_tool(tool, section="profile")["outcome"] == "ok"
    assert "private MCP log notification" not in caplog.text
    mcp_server["elicit"] = True
    assert invoke_tool(tool, section="profile")["outcome"] == "error"
    assert len(mcp_server["calls"]) == 2  # One execution for each explicitly invoked call.


@pytest.mark.parametrize(
    "values",
    [
        {"mcp_url": "file:///tmp/server.py", "mcp_customer_id_header": "X-Customer"},
        {"mcp_url": "https://example.test/mcp"},
        {"mcp_url": "https://user:secret@example.test/mcp", "mcp_customer_id_header": "X-Customer"},
        {"mcp_customer_id_header": "Bad\nHeader"},
        {"mcp_customer_id_header": "Authorization"},
        {"mcp_customer_id_header": "Mcp-Method"},
        {"mcp_customer_id_header": "X-Customer", "mcp_headers": {"x-customer": "spoofed"}},
    ],
)
def test_invalid_mcp_configuration_is_rejected(settings, values):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{**settings.model_dump(), **values})

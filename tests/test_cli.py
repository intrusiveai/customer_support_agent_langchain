import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest


def run_cli(tmp_path, extra_environment, *, script=None):
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("SUPPORT_")
    }
    environment.update(
        SUPPORT_DATABASE_URL="postgresql://unused/unused",
        SUPPORT_MODEL_NAME="test",
    )
    environment.update(extra_environment)
    return subprocess.run(
        [
            sys.executable,
            *(["-c", script] if script else ["-m", "support_agent.cli"]),
            "--customer-id", "alice", "--message", "Test logging",
        ],
        env=environment,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
@pytest.mark.parametrize("to_file", [False, True])
def test_cli_log_destination_and_severity(tmp_path, level, to_file):
    log_file = tmp_path / "agent.log"
    environment = {"SUPPORT_LOG_LEVEL": level}
    if to_file:
        environment["SUPPORT_LOG_FILE"] = "agent.log"
        log_file.write_text("Existing log entry\n", encoding="utf-8")
    completed = run_cli(tmp_path, environment, script="""
import logging
from support_agent import cli
from support_agent.results import Rejected

def startup(settings):
    for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        logging.getLogger("support_agent.session").log(
            getattr(logging, level), "Agent probe %s café", level
        )
        logging.getLogger("third_party").log(
            getattr(logging, level), "Dependency probe %s", level
        )
    raise Rejected("probe_complete", "Logging probe complete")

cli.SupportApplication = startup
cli.main()
""")
    assert completed.returncode == 1
    assert "Logging probe complete" in completed.stderr
    assert completed.stdout == ""
    logs = log_file.read_text(encoding="utf-8") if to_file else completed.stderr
    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    for index, severity in enumerate(levels):
        assert (f"Agent probe {severity} café" in logs) == (index >= levels.index(level))
        assert (f"Dependency probe {severity}" in logs) == (index >= max(2, levels.index(level)))
    if to_file:
        assert logs.startswith("Existing log entry\n")
        assert "Agent probe" not in completed.stderr
        assert "Dependency probe" not in completed.stderr
    else:
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "environment, error",
    [
        ({"SUPPORT_LOG_LEVEL": "VERBOSE"}, "log_level"),
        ({"SUPPORT_LOG_FILE": ""}, "log_file"),
        ({"SUPPORT_LOG_FILE": "missing/agent.log"}, "cannot open SUPPORT_LOG_FILE"),
        ({"SUPPORT_LOG_FILE": "."}, "cannot open SUPPORT_LOG_FILE"),
    ],
)
def test_cli_rejects_invalid_logging_configuration(tmp_path, environment, error):
    completed = run_cli(tmp_path, environment)
    assert completed.returncode == 1
    assert "Configuration error:" in completed.stderr
    assert error in completed.stderr
    assert completed.stdout == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.integration
@pytest.mark.parametrize("log_level", [None, "INFO", "WARNING"])
def test_cli_uses_configured_chat_integration_and_real_account_tool(
    database, db_settings, tmp_path, log_level
):
    """Actual CLI -> configured ChatOpenAI -> local protocol server -> real DB tool."""
    requests = []

    class ModelServer(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            assert self.path == "/v1/chat/completions"
            assert self.headers["Authorization"] == "Bearer local-test-only"
            assert request["model"] == "configured-local-model"
            tools = {tool["function"]["name"]: tool["function"] for tool in request["tools"]}
            assert len(tools) == 5
            assert "runtime" not in tools["get_order"]["parameters"]["properties"]
            if any(message["role"] == "tool" for message in request["messages"]):
                tool_result = json.loads(
                    next(
                        message["content"]
                        for message in request["messages"]
                        if message["role"] == "tool"
                    )
                )
                assert tool_result["data"]["orders"][0]["projected"]["status"] == "delivered"
                message = {"role": "assistant", "content": "Order alice-order is delivered."}
                finish = "stop"
            else:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "provider-call-1",
                            "type": "function",
                            "function": {
                                "name": "get_order",
                                "arguments": json.dumps({"order_id": "alice-order"}),
                            },
                        }
                    ],
                }
                finish = "tool_calls"
            body = json.dumps(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": request["model"],
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelServer)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        environment = {
            **{key: value for key, value in os.environ.items() if not key.startswith("SUPPORT_")},
            "SUPPORT_DATABASE_URL": db_settings.database_url.get_secret_value(),
            "SUPPORT_MODEL_NAME": "configured-local-model",
            "SUPPORT_MODEL_PROVIDER": "openai",
            "SUPPORT_MODEL_API_KEY": "local-test-only",
            "SUPPORT_MODEL_KWARGS": "{}",
            "SUPPORT_MODEL_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
        }
        log_file = tmp_path / "agent.log"
        if log_level is not None:
            environment.update(SUPPORT_LOG_FILE=str(log_file), SUPPORT_LOG_LEVEL=log_level)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "support_agent.cli",
                "--customer-id",
                "alice",
                "--message",
                "Check alice-order",
            ],
            env=environment,
            cwd=tmp_path,
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )
        assert json.loads(completed.stdout)["answer"] == "Order alice-order is delivered."
        assert len(requests) == 2
        logs = log_file.read_text(encoding="utf-8") if log_level else completed.stderr
        assert ("Interaction completed" in logs) == (log_level != "WARNING")
        assert ("tool=get_order outcome=ok" in logs) == (log_level != "WARNING")
        assert "local-test-only" not in logs and "support-test" not in logs
        if log_level:
            assert "Interaction completed" not in completed.stderr
            assert "tool=get_order" not in completed.stderr
            assert list(tmp_path.iterdir()) == [log_file]
        else:
            assert list(tmp_path.iterdir()) == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

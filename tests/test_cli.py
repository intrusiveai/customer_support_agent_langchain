import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest


@pytest.mark.integration
def test_cli_uses_configured_chat_integration_and_real_account_tool(
    database, db_settings, tmp_path
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
            **os.environ,
            "SUPPORT_DATABASE_URL": db_settings.database_url.get_secret_value(),
            "SUPPORT_MODEL_NAME": "configured-local-model",
            "SUPPORT_MODEL_PROVIDER": "openai",
            "SUPPORT_MODEL_API_KEY": "local-test-only",
            "SUPPORT_MODEL_KWARGS": "{}",
            "SUPPORT_MODEL_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
            "SUPPORT_EMBEDDING_REVISION": db_settings.embedding_revision,
            "SUPPORT_EVIDENCE_PATH": str(tmp_path / "cli-evidence"),
        }
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
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )
        assert json.loads(completed.stdout)["answer"] == "Order alice-order is delivered."
        assert len(requests) == 2
        evidence = "".join(p.read_text() for p in (tmp_path / "cli-evidence").glob("*.json"))
        assert "local-test-only" not in evidence and "support-test" not in evidence
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from support_agent.results import Rejected
from support_agent.session import SupportApplication, invoke_support
from test_agent import ScriptedModel

pytestmark = pytest.mark.integration


class ResponseModel(ScriptedModel):
    calls: list = []
    response: AIMessage
    invocations: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.invocations += 1
        return ChatResult(generations=[ChatGeneration(message=self.response.model_copy(deep=True))])


def events(settings):
    return [json.loads(path.read_text()) for path in settings.evidence_path.glob("*.json")]


@pytest.mark.parametrize(
    "usage", [None, {"input_tokens": 17, "output_tokens": 2048, "total_tokens": 2065}]
)
def test_empty_without_termination_metadata_is_not_exhaustion(database, db_settings, usage):
    model = ResponseModel(response=AIMessage(content=" \n", usage_metadata=usage))
    with SupportApplication(db_settings, model=model) as app:
        with pytest.raises(Rejected) as error:
            invoke_support(app, app.start_session("alice"), "Help.")
    assert error.value.code == "model_empty_response"
    failed = next(e for e in events(db_settings) if e["event"] == "interaction_failed")
    metadata = failed["messages"][-1]["model_metadata"]
    for field in [
        "finish_reason",
        "stop_reason",
        "stopReason",
        "status",
        "incomplete_reason",
        "reported_model",
        "reasoning_tokens",
    ]:
        assert metadata[field] is None
    assert metadata["input_tokens"] == (17 if usage else None)
    assert metadata["output_tokens"] == (2048 if usage else None)
    assert model.invocations == 1


@pytest.mark.parametrize(
    "field,reason",
    [("finish_reason", "length"), ("stop_reason", "max_tokens"), ("stopReason", "max_tokens")],
)
def test_exhausted_response_cannot_execute_tool_calls(database, db_settings, field, reason):
    model = ResponseModel(
        response=AIMessage(
            content="",
            response_metadata={field: reason},
            tool_calls=[
                {
                    "id": "refund-call",
                    "name": "issue_refund",
                    "args": {"order_id": "alice-order", "amount": "1.00", "reason": "test"},
                }
            ],
        )
    )
    with SupportApplication(db_settings, model=model) as app:
        with pytest.raises(Rejected) as error:
            invoke_support(app, app.start_session("alice"), "Refund one dollar.")
    assert error.value.code == "model_output_limit"
    assert model.invocations == 1
    assert not any(e["event"] == "tool_started" for e in events(db_settings))
    with database.transaction(read_only=True) as tx:
        assert tx.execute("SELECT count(*) AS n FROM support.refunds").fetchone()["n"] == 0


@pytest.mark.parametrize("text", ["", "Partial answer"])
@pytest.mark.parametrize(
    "reason,code",
    [("max_output_tokens", "model_output_limit"), ("content_filter", "model_incomplete_response")],
)
def test_openai_responses_api_incomplete_metadata(database, db_settings, text, reason, code):
    model = ResponseModel(
        response=AIMessage(
            content=text,
            response_metadata={
                "status": "incomplete",
                "incomplete_details": {"reason": reason},
            },
        )
    )
    with SupportApplication(db_settings, model=model) as app:
        with pytest.raises(Rejected) as error:
            invoke_support(app, app.start_session("alice"), "Help.")
    assert error.value.code == code
    failed = next(e for e in events(db_settings) if e["event"] == "interaction_failed")
    assert failed["messages"][-1]["content"] == text
    assert failed["messages"][-1]["model_metadata"]["incomplete_reason"] == reason


def test_metadata_is_allowlisted_and_bounded(database, db_settings):
    model = ResponseModel(
        response=AIMessage(
            content="Complete answer.",
            response_metadata={
                "model_name": "m" * 1000,
                "finish_reason": "stop",
                "headers": {"Authorization": "sensitive-test-credential"},
                "sdk_object": object(),
            },
        )
    )
    with SupportApplication(db_settings, model=model) as app:
        output = invoke_support(app, app.start_session("alice"), "Help.")
    assert output["answer"] == "Complete answer."
    records = events(db_settings)
    finished = next(e for e in records if e["event"] == "interaction_finished")
    metadata = finished["messages"][-1]["model_metadata"]
    assert metadata["reported_model"] == "m" * 256
    assert metadata["reasoning_tokens"] is None
    assert "sdk_object" not in json.dumps(records)
    assert "sensitive-test-credential" not in json.dumps(records)

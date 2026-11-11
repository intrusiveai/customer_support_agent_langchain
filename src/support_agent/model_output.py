"""Reject incomplete generation without replay and log failure diagnostics."""

import logging

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from .results import Rejected

logger = logging.getLogger(__name__)


def model_metadata(message, settings):
    """Allowlisted scalar metadata only; absent/invalid values remain explicitly unknown."""
    metadata = message.response_metadata or {}
    usage = message.usage_metadata or {}
    details = usage.get("output_token_details") or {}

    def text(value):
        return value[:256] if isinstance(value, str) else None

    def count(value):
        return value if type(value) is int and 0 <= value <= 2**63 - 1 else None

    incomplete = metadata.get("incomplete_details")
    reasoning = next(
        (
            details[key]
            for key in ("reasoning", "priority_reasoning", "flex_reasoning")
            if key in details
        ),
        None,
    )
    return {
        "provider": text(settings.model_provider),
        "model": text(settings.model_name),
        "completion_allowance": settings.model_max_tokens,
        "reported_model": text(metadata.get("model_name") or metadata.get("model")),
        "finish_reason": text(metadata.get("finish_reason")),
        "stop_reason": text(metadata.get("stop_reason")),
        "stopReason": text(metadata.get("stopReason")),
        "status": text(metadata.get("status")),
        "incomplete_reason": text(incomplete.get("reason"))
        if isinstance(incomplete, dict)
        else None,
        "input_tokens": count(usage.get("input_tokens")),
        "output_tokens": count(usage.get("output_tokens")),
        "reasoning_tokens": count(reasoning),
    }


def validate_model_output(message, settings, *, final=False):
    if isinstance(message, AIMessage):
        metadata = model_metadata(message, settings)
        if (
            metadata["finish_reason"] == "length"
            or metadata["stop_reason"] == "max_tokens"
            or metadata["stopReason"] == "max_tokens"
            or (
                metadata["status"] == "incomplete"
                and metadata["incomplete_reason"] == "max_output_tokens"
            )
        ):
            raise Rejected(
                "model_output_limit",
                "The model exhausted its completion-token allowance before producing a complete answer. "
                "Check application logs and the refund ledger before retrying.",
            )
        if metadata["status"] == "incomplete":
            raise Rejected(
                "model_incomplete_response",
                "The model reported incomplete generation. Check application logs and the refund ledger before retrying.",
            )
        if not final and message.tool_calls:
            return
        if message.text.strip():
            return
    raise Rejected(
        "model_empty_response",
        "The model returned an empty final answer without a token-exhaustion indication. "
        "Check application logs and the refund ledger before retrying.",
    )


class ModelOutputMiddleware(AgentMiddleware):
    def __init__(self, settings):
        self.settings = settings

    def after_model(self, state, runtime):
        message = state["messages"][-1]
        try:
            validate_model_output(message, self.settings)
        except Rejected as exc:
            logger.warning(
                "Model response rejected session=%s interaction=%s code=%s metadata=%s",
                runtime.context.session_id,
                runtime.context.interaction_id,
                exc.code,
                model_metadata(message, self.settings) if isinstance(message, AIMessage) else None,
            )
            raise

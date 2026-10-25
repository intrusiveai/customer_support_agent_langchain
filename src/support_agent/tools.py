import json
from typing import Annotated, Any

import psycopg
from langchain.agents.middleware import wrap_tool_call
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolArg
from psycopg_pool import PoolTimeout
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from .context import RuntimeContext
from .database import diagnostics
from .results import Rejected, json_text, result


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Required by validation, hidden from model tool schemas, replaced by ToolNode.
    runtime: Annotated[Any, InjectedToolArg()] = Field(exclude=True)


def build_tools(accounts, retrieval, web, settings):
    identifier = Field(min_length=1, max_length=settings.identifier_char_limit, strict=True)
    order_schema = create_model("OrderArgs", __base__=ToolArgs, order_id=(str, identifier))
    customer_schema = create_model("CustomerArgs", __base__=ToolArgs, customer_id=(str, identifier))
    refund_schema = create_model(
        "RefundArgs",
        __base__=ToolArgs,
        order_id=(str, identifier),
        amount=(str, Field(min_length=1, max_length=32, strict=True)),
        reason=(str, Field(min_length=1, max_length=2000, strict=True)),
    )
    search_schema = create_model(
        "SearchArgs",
        __base__=ToolArgs,
        query=(str, Field(min_length=1, max_length=4000, strict=True)),
    )
    web_schema = create_model(
        "WebArgs", __base__=ToolArgs, url=(str, Field(min_length=1, max_length=8192, strict=True))
    )

    @tool(args_schema=search_schema)
    def search_support_docs(query: str, runtime: ToolRuntime[RuntimeContext]) -> dict:
        """Search shared support policies and troubleshooting passages; cite returned references."""
        return json.loads(
            json_text(retrieval.search(runtime.context.call(runtime.tool_call_id), query))
        )

    @tool(args_schema=order_schema)
    def get_order(order_id: str, runtime: ToolRuntime[RuntimeContext]) -> dict:
        """Look up an order, its purchased items and its refund history."""
        return json.loads(
            json_text(accounts.get_order(runtime.context.call(runtime.tool_call_id), order_id))
        )

    @tool(args_schema=customer_schema)
    def get_customer(customer_id: str, runtime: ToolRuntime[RuntimeContext]) -> dict:
        """Retrieve a customer profile and contact preferences."""
        return json.loads(
            json_text(
                accounts.get_customer(runtime.context.call(runtime.tool_call_id), customer_id)
            )
        )

    @tool(args_schema=refund_schema)
    def issue_refund(
        order_id: str, amount: str, reason: str, runtime: ToolRuntime[RuntimeContext]
    ) -> dict:
        """Record a simulated USD refund. Amount is a decimal string, for example '12.50'."""
        if runtime.context.uncertain_refunds:
            return result(
                "rejected",
                code="refund_resolution_required",
                message="A previous refund has an uncertain outcome. Inspect the ledger before any further refund.",
            )
        ctx = runtime.context.call(runtime.tool_call_id)
        response = accounts.issue_refund(ctx, order_id, amount, reason)
        if response["outcome"] == "uncertain":
            runtime.context.uncertain_refunds.add(ctx.invocation_id)
        return json.loads(json_text(response))

    @tool(args_schema=web_schema)
    def fetch_web_page(url: str, runtime: ToolRuntime[RuntimeContext]) -> dict:
        """Retrieve readable content from an absolute HTTP or HTTPS URL."""
        return web.fetch(runtime.context.call(runtime.tool_call_id), url)

    return [search_support_docs, get_order, get_customer, issue_refund, fetch_web_page]


@wrap_tool_call
def retain_tool_evidence(request, handler):
    context = request.runtime.context
    ctx = context.call(request.tool_call["id"])
    ctx.record(
        "tool_started",
        tool=request.tool_call["name"],
        arguments=request.tool_call["args"],
        model_tool_call_id=request.tool_call["id"],
    )
    try:
        # Enforce here so denied calls receive the same trusted IDs and evidence as executed calls.
        context.consume_tool_call()
        message = handler(request)
    except ValidationError:
        response = result(
            "rejected",
            code="invalid_arguments",
            message="Tool arguments do not match the bounded input schema.",
        )
    except Rejected as exc:
        response = result("rejected", code=exc.code, message=exc.message)
    except (psycopg.Error, PoolTimeout) as exc:
        ctx.record("database_error", diagnostics=diagnostics(exc))
        response = result(
            "error", code="database_error", message="Database operation failed; see local evidence."
        )
    else:
        try:
            response = json.loads(message.content)
        except (json.JSONDecodeError, TypeError):
            response = result(
                "rejected",
                code="invalid_arguments" if message.status == "error" else "execution_limit",
                message="Tool invocation rejected by argument validation or execution limits.",
            )
    # Normalize UUIDs, timestamps and integer money without altering projected strings.
    encoded = json_text(response)
    try:
        ctx.record("tool_finished", tool=request.tool_call["name"], result=json.loads(encoded))
    except OSError:
        # A committed refund must remain committed in the caller's view even if disk fails.
        if response["outcome"] != "committed":
            raise
        response["evidence_error"] = (
            "Could not persist final event; resolve using the committed ledger result."
        )
        encoded = json_text(response)
    return ToolMessage(
        content=encoded, tool_call_id=request.tool_call["id"], name=request.tool_call["name"]
    )

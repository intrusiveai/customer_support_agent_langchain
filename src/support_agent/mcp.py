"""Discover remote tools and bridge their async MCP calls into the synchronous agent."""

import asyncio
import logging
from time import monotonic
from urllib.parse import quote

from langchain.tools import ToolRuntime
from langchain_core.tools import StructuredTool

from .context import RuntimeContext
from .results import Rejected, json_text, result

logger = logging.getLogger(__name__)


def _client(settings, ctx=None):
    # Optional integration: leave MCP dependencies unloaded when no server is configured.
    import httpx2
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    headers = {name: value.get_secret_value() for name, value in settings.mcp_headers.items()}
    if ctx is not None:
        headers[settings.mcp_customer_id_header] = quote(ctx.authenticated_customer_id, safe="")

    def http_client(**kwargs):
        # Custom credentials and customer identity must stay at the configured endpoint.
        return httpx2.AsyncClient(**{**kwargs, "follow_redirects": False, "trust_env": False})

    async def ignore_server_log(message):
        # Server notifications may contain customer data; log local outcomes instead.
        pass

    return Client(
        StreamableHttpTransport(
            str(settings.mcp_url), headers=headers, httpx_client_factory=http_client
        ),
        timeout=settings.mcp_timeout_seconds,
        init_timeout=settings.mcp_timeout_seconds,
        log_handler=ignore_server_log,
        input_required_max_rounds=0,
    )


def load_mcp_tools(settings):
    if settings.mcp_url is None:
        return []
    try:
        return asyncio.run(_discover(settings))
    except Exception:
        # Connection exceptions may contain configured URLs, headers or server content.
        raise Rejected(
            "mcp_configuration",
            "Cannot discover MCP tools; check the server URL, credentials and tool schemas.",
        ) from None


async def _discover(settings):
    from langchain.mcp import as_langchain_tool

    async with asyncio.timeout(settings.mcp_timeout_seconds):
        async with _client(settings) as client:
            definitions = await client.list_tools()
            tools = []
            names = set()
            for definition in definitions:
                adapted = await as_langchain_tool(definition, client)
                if definition.name in names or "_support_runtime" in adapted.args_schema.get(
                    "properties", {}
                ):
                    raise ValueError("Duplicate tool name or reserved runtime argument")
                names.add(definition.name)
                tools.append(_sync_tool(settings, adapted))
            return tools


def _sync_tool(settings, adapted):
    remote_name = adapted.name

    def invoke(_support_runtime: ToolRuntime[RuntimeContext], **arguments):
        ctx = _support_runtime.context.call(_support_runtime.tool_call_id)
        started = monotonic()
        response = asyncio.run(_call(settings, remote_name, arguments, ctx))
        logger.info(
            "MCP call completed session=%s interaction=%s invocation=%s duration_ms=%d outcome=%s",
            ctx.session_id,
            ctx.interaction_id,
            ctx.invocation_id,
            (monotonic() - started) * 1000,
            response["outcome"],
        )
        return response

    return StructuredTool(
        name=f"mcp_{remote_name}",
        description=adapted.description,
        args_schema=adapted.args_schema,
        func=invoke,
    )


async def _call(settings, name, arguments, ctx):
    try:
        # Each call owns its client and event loop; no shared customer headers or sessions.
        async with asyncio.timeout(settings.mcp_timeout_seconds):
            async with _client(settings, ctx) as client:
                remote = await client.call_tool_mcp(name, arguments)
        data = {
            "content": [
                block.model_dump(mode="json", by_alias=True, exclude_none=True)
                for block in remote.content
            ],
            "structured_content": remote.structured_content,
        }
        response = (
            result("error", data=data, code="mcp_tool_error", message="MCP tool reported an error.")
            if remote.is_error
            else result(data=data)
        )
        if len(json_text(response)) > settings.mcp_result_char_limit:
            return result(
                "error",
                code="mcp_result_limit",
                message="MCP tool response exceeds the configured character limit.",
            )
        return response
    except TimeoutError:
        return result("error", code="mcp_timeout", message="MCP request timed out.")
    except Exception:
        return result(
            "error",
            code="mcp_unavailable",
            message="MCP request failed; check server availability and configuration.",
        )

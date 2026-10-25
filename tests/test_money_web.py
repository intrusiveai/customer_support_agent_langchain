import asyncio

import httpx
import pytest

from support_agent.results import Rejected, amount_minor, usd
from support_agent.web import Web, validate_url


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "-1",
        "+1",
        "1e2",
        "NaN",
        "Infinity",
        " 1",
        ".50",
        "1.",
        "0.001",
        "1.2301",
        "92233720368547758.08",
    ],
)
def test_invalid_amount(value):
    with pytest.raises(Rejected):
        amount_minor(value)


def test_exact_money():
    assert amount_minor("00012.5000") == 1250
    assert usd(amount_minor("92233720368547758.07")) == "92233720368547758.07"


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "/relative", "https://a:b@example.test", "https://user@example.test"],
)
def test_invalid_url(url):
    with pytest.raises(Rejected):
        validate_url(url)


def test_web_redirect_content_limits_and_no_cookies(settings, ctx):
    settings = settings.model_copy(update={"web_char_limit": 20})

    def handler(request):
        assert "cookie" not in request.headers and "authorization" not in request.headers
        if request.url.path == "/start":
            return httpx.Response(
                302, headers={"location": "http://127.0.0.1/final", "set-cookie": "session=secret"}
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<script>bad()</script><p>Readable page with a lot of content.</p>",
        )

    response = Web(settings, httpx.MockTransport(handler)).fetch(ctx, "https://example.test/start")
    assert response["outcome"] == "ok"
    data = response["data"]
    assert data["final_url"] == "http://127.0.0.1/final" and len(data["redirects"]) == 1
    assert data["content"] == "Readable page with a" and data["truncated"]


@pytest.mark.parametrize(
    "status,headers,body,code",
    [
        (404, {"content-type": "text/plain"}, b"missing", "http_error"),
        (200, {"content-type": "application/pdf"}, b"pdf", "unsupported_content"),
        (200, {"content-type": "text/plain"}, b"", "empty_response"),
        (204, {}, b"", "empty_response"),
        (404, {"content-type": "image/png"}, b"missing", "http_error"),
        (302, {"location": "https://u:p@example.test"}, b"", "url_credentials"),
        (
            200,
            {"content-type": "text/plain", "content-encoding": "br"},
            b"",
            "unsupported_encoding",
        ),
    ],
)
def test_web_failures(settings, ctx, status, headers, body, code):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(status, headers=headers, content=body)
    )
    assert Web(settings, transport).fetch(ctx, "http://example.test")["error"]["code"] == code


def test_web_total_deadline(settings, ctx):
    async def slow(request):
        await asyncio.sleep(0.1)
        return httpx.Response(200, text="too late")

    limited = settings.model_copy(update={"http_timeout_seconds": 0.01})
    assert (
        Web(limited, httpx.MockTransport(slow)).fetch(ctx, "https://example.test")["error"]["code"]
        == "http_timeout"
    )


def test_web_byte_limit_and_json(settings, ctx):
    limited = settings.model_copy(update={"http_byte_limit": 5})
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"large": "a" * 100}))
    response = Web(limited, transport).fetch(ctx, "https://example.test")
    assert response["data"]["retained_bytes"] == 5
    assert response["data"]["truncated"] and response["data"]["content"] == '{"lar'
    full = Web(settings, transport).fetch(ctx, "https://example.test")
    assert not full["data"]["truncated"] and full["data"]["content_type"] == "application/json"


def test_web_redirect_limit_and_transport_error(settings, ctx):
    limited = settings.model_copy(update={"http_redirect_limit": 1})
    loop = httpx.MockTransport(lambda request: httpx.Response(302, headers={"location": "/loop"}))
    response = Web(limited, loop).fetch(ctx, "https://example.test")
    assert response["error"]["code"] == "redirect_limit"
    assert len(response["data"]["redirects"]) == 2

    def fail(request):
        raise httpx.ConnectError("test transport failure", request=request)

    assert (
        Web(settings, httpx.MockTransport(fail)).fetch(ctx, "https://example.test")["error"]["code"]
        == "transport_error"
    )

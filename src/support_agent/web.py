import asyncio

import httpx
from bs4 import BeautifulSoup

from .results import Rejected, digest, result


def validate_url(value):
    try:
        url = httpx.URL(value)
    except (httpx.InvalidURL, ValueError) as exc:
        raise Rejected("invalid_url", "An absolute HTTP or HTTPS URL is required.") from exc
    if url.scheme not in {"http", "https"} or not url.host or not url.is_absolute_url:
        raise Rejected("invalid_url", "An absolute HTTP or HTTPS URL is required.")
    if url.userinfo:
        raise Rejected("url_credentials", "Embedded URL credentials are not allowed.")
    return url


class Web:
    def __init__(self, settings, transport=None):
        self.settings, self.transport = settings, transport

    def fetch(self, ctx, url):
        return asyncio.run(self._fetch(ctx, url))

    async def _fetch(self, ctx, requested):
        redirects = []
        details = {
            "requested_url": requested,
            "final_url": None,
            "redirects": redirects,
            "status": None,
            "content_type": None,
            "content": "",
            "truncated": False,
            "content_digest": digest(b""),
        }
        try:
            current = validate_url(requested)
            details["final_url"] = str(current)
            # Covers redirects, decompression and streaming, including slow-drip servers.
            async with asyncio.timeout(self.settings.http_timeout_seconds):
                async with httpx.AsyncClient(
                    timeout=self.settings.http_timeout_seconds,
                    verify=True,
                    follow_redirects=False,
                    trust_env=False,
                    transport=self.transport,
                    headers={
                        "User-Agent": "PortableSupportAgent/0.1",
                        "Accept-Encoding": "identity",
                    },
                ) as client:
                    for hop in range(self.settings.http_redirect_limit + 1):
                        # Do not carry even server-issued cookies between redirect destinations.
                        client.cookies.clear()
                        details["final_url"] = str(current)
                        async with client.stream("GET", current) as response:
                            details.update(
                                status=response.status_code,
                                content_type=response.headers.get("content-type", ""),
                            )
                            if response.status_code in {301, 302, 303, 307, 308}:
                                location = response.headers.get("location")
                                if not location:
                                    return result(
                                        "error",
                                        details,
                                        "invalid_redirect",
                                        "Redirect is missing Location.",
                                    )
                                try:
                                    target_text = str(current.join(location))
                                except (httpx.InvalidURL, ValueError):
                                    redirects.append(
                                        {
                                            "url": str(current),
                                            "status": response.status_code,
                                            "location": location,
                                        }
                                    )
                                    return result(
                                        "rejected",
                                        details,
                                        "invalid_redirect",
                                        "Redirect target is not a valid URL.",
                                    )
                                redirects.append(
                                    {
                                        "url": str(current),
                                        "status": response.status_code,
                                        "location": target_text,
                                    }
                                )
                                target = validate_url(target_text)
                                if hop == self.settings.http_redirect_limit:
                                    return result(
                                        "error",
                                        details,
                                        "redirect_limit",
                                        "Redirect limit exceeded.",
                                    )
                                current = target
                                continue
                            if response.status_code in {204, 205}:
                                return result(
                                    "empty",
                                    details,
                                    "empty_response",
                                    "No readable content returned.",
                                )
                            mime = details["content_type"].split(";", 1)[0].strip().lower()
                            if mime not in {
                                "text/html",
                                "application/xhtml+xml",
                                "text/plain",
                                "application/json",
                            } and not mime.endswith("+json"):
                                return result(
                                    "error",
                                    details,
                                    "http_error" if response.is_error else "unsupported_content",
                                    f"HTTP status {response.status_code}."
                                    if response.is_error
                                    else "Only HTML, plain text and JSON are supported.",
                                )
                            # Reject unexpected compression instead of allocating an unbounded decompression buffer.
                            if response.headers.get("content-encoding", "identity").lower() not in {
                                "",
                                "identity",
                            }:
                                return result(
                                    "error",
                                    details,
                                    "unsupported_encoding",
                                    "Server ignored identity content encoding.",
                                )
                            raw = bytearray()
                            async for chunk in response.aiter_bytes(
                                chunk_size=min(8192, self.settings.http_byte_limit + 1)
                            ):
                                remaining = self.settings.http_byte_limit + 1 - len(raw)
                                raw.extend(chunk[:remaining])
                                if len(raw) > self.settings.http_byte_limit:
                                    break
                            limited = len(raw) > self.settings.http_byte_limit
                            retained = bytes(raw[: self.settings.http_byte_limit])
                            try:
                                text = retained.decode(
                                    response.encoding or "utf-8", errors="replace"
                                )
                            except LookupError:
                                text = retained.decode("utf-8", errors="replace")
                            if mime in {"text/html", "application/xhtml+xml"}:
                                soup = BeautifulSoup(text, "html.parser")
                                for node in soup(["script", "style", "noscript", "template"]):
                                    node.decompose()
                                text = soup.get_text(" ", strip=True)
                            content = text[: self.settings.web_char_limit]
                            details.update(
                                content=content,
                                content_digest=digest(content.encode("utf-8")),
                                retained_body_digest=digest(retained),
                                retained_bytes=len(retained),
                                truncated=limited or len(text) > len(content),
                            )
                            if response.is_error:
                                return result(
                                    "error",
                                    details,
                                    "http_error",
                                    f"HTTP status {response.status_code}.",
                                )
                            if not content.strip():
                                return result(
                                    "empty",
                                    details,
                                    "empty_response",
                                    "No readable content returned.",
                                )
                            return result(data=details)
        except (TimeoutError, httpx.TimeoutException):
            return result("error", details, "http_timeout", "Web request deadline exceeded.")
        except httpx.RequestError:
            return result("error", details, "transport_error", "Web transport failed.")
        except Rejected as exc:
            return result("rejected", details, exc.code, exc.message)

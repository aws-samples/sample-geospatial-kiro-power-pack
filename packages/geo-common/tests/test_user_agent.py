"""The shared HttpClient sends a descriptive, non-default User-Agent.

Open geospatial APIs (OpenStreetMap Overpass/Nominatim) reject the stock
``python-httpx/<ver>`` User-Agent with HTTP 406, so the shared client must send
a descriptive one by default — while still letting a caller override it
per-request. These tests capture the outgoing ``User-Agent`` via a mock
transport (no network).
"""

from __future__ import annotations

import httpx
import pytest

from geo_common.http import DEFAULT_USER_AGENT, HttpClient


async def _noop_sleep(_seconds: float) -> None:
    return None


def _client_recording(seen: dict) -> HttpClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen["user_agent"] = request.headers.get("User-Agent")
        return httpx.Response(200, json={})

    return HttpClient(transport=httpx.MockTransport(handler), sleep=_noop_sleep)


async def test_default_user_agent_is_descriptive_not_httpx() -> None:
    seen: dict = {}
    client = _client_recording(seen)
    try:
        await client.get("https://example.org/x")
    finally:
        await client.aclose()
    assert seen["user_agent"] == DEFAULT_USER_AGENT
    assert "geospatial-kiro-power-pack" in seen["user_agent"]
    assert "python-httpx" not in seen["user_agent"]


async def test_constructor_user_agent_override() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["user_agent"] = request.headers.get("User-Agent")
        return httpx.Response(200, json={})

    client = HttpClient(
        transport=httpx.MockTransport(handler),
        sleep=_noop_sleep,
        user_agent="geo-stac/0.1 (test)",
    )
    try:
        await client.get("https://example.org/x")
    finally:
        await client.aclose()
    assert seen["user_agent"] == "geo-stac/0.1 (test)"


async def test_per_request_user_agent_takes_precedence() -> None:
    seen: dict = {}
    client = _client_recording(seen)
    try:
        await client.get("https://example.org/x", headers={"User-Agent": "custom/9"})
    finally:
        await client.aclose()
    assert seen["user_agent"] == "custom/9"

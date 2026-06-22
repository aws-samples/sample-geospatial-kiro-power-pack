"""Unit tests for the IUCN Red List source (geo-biodiversity).

Drives ``IUCNSource`` through an ``httpx.MockTransport`` returning canned IUCN
Red List API v4 payloads. No real network I/O happens. Covers the credential
guard (no token -> AuthenticationError naming the key), the name-based query
shape and ``Authorization`` header, conservation-status parsing, and the
no-taxon short-circuit.
"""

from __future__ import annotations

import httpx
import pytest

from geo_common.errors import AuthenticationError, ErrorCategory
from geo_common.http import HttpClient
from geo_common.models import CredentialClassification
from geo_common.retry import RetryPolicy

from geo_biodiversity.occurrences import IUCNSource
from geo_biodiversity.server import GeoBiodiversityServer

SMALL_BBOX = (13.40, 52.50, 13.41, 52.51)

_IUCN_PAYLOAD = {
    "taxon": {"sis_id": 15951, "scientific_name": "Panthera leo"},
    "assessments": [
        {
            "assessment_id": 174811,
            "year_published": "2023",
            "latest": True,
            "red_list_category_code": "VU",
        }
    ],
}


def _client(handler, **kwargs):
    return HttpClient(
        RetryPolicy(max_attempts=1), transport=httpx.MockTransport(handler), **kwargs
    )


async def test_iucn_without_token_is_authentication_error():
    """No token -> AuthenticationError naming IUCN_TOKEN, before any request."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        calls["n"] += 1
        return httpx.Response(200, json=_IUCN_PAYLOAD)

    source = IUCNSource(token=None)
    client = _client(handler)
    try:
        with pytest.raises(AuthenticationError) as exc:
            await source.fetch(client, SMALL_BBOX, "Panthera leo", 100)
    finally:
        await client.aclose()

    err = exc.value
    assert err.category is ErrorCategory.AUTHENTICATION
    assert err.detail.get("mcp_json_key") == "IUCN_TOKEN"
    assert "IUCN_TOKEN" in str(err)
    assert calls["n"] == 0


async def test_iucn_queries_by_name_with_auth_header():
    """With a token, IUCN is queried by genus/species and authenticates via the
    ``Authorization`` header; the assessment is parsed into a record."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["genus"] = request.url.params.get("genus_name")
        seen["species"] = request.url.params.get("species_name")
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=_IUCN_PAYLOAD)

    source = IUCNSource(token="secret-token")
    client = _client(handler)
    try:
        records = await source.fetch(client, SMALL_BBOX, "Panthera leo", 100)
    finally:
        await client.aclose()

    assert seen["path"].endswith("/taxa/scientific_name")
    assert seen["genus"] == "Panthera" and seen["species"] == "leo"
    assert seen["auth"] == "secret-token"  # token in the Authorization header

    assert len(records) == 1
    rec = records[0]
    assert rec.source == "iucn"
    assert rec.scientific_name == "Panthera leo"
    assert rec.properties["category"] == "VU"
    # An assessment carries no point geometry.
    assert rec.longitude is None and rec.latitude is None


async def test_iucn_without_taxon_returns_no_records():
    """IUCN is name-based: with no taxon there is nothing to look up."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        calls["n"] += 1
        return httpx.Response(200, json=_IUCN_PAYLOAD)

    source = IUCNSource(token="secret-token")
    client = _client(handler)
    try:
        records = await source.fetch(client, SMALL_BBOX, None, 100)
    finally:
        await client.aclose()
    assert records == []
    assert calls["n"] == 0


def test_iucn_declared_optional_credential():
    """The server declares IUCN_TOKEN as an Optional credential."""
    specs = GeoBiodiversityServer().required_credentials()
    iucn = [s for s in specs if s.mcp_json_key == "IUCN_TOKEN"]
    assert len(iucn) == 1
    assert iucn[0].classification is CredentialClassification.OPTIONAL


def test_iucn_source_always_selectable_and_opt_in(monkeypatch):
    """IUCN is always in the source set (so ``source="iucn"`` resolves), but is
    excluded from the default merge via ``default_merge=False``."""
    monkeypatch.delenv("IUCN_TOKEN", raising=False)
    without = GeoBiodiversityServer()
    iucn = [s for s in without.sources if s.name == "iucn"]
    assert len(iucn) == 1  # present even with no token
    assert iucn[0].default_merge is False  # but opt-in, not in the default merge
    assert iucn[0].token is None

    monkeypatch.setenv("IUCN_TOKEN", "secret-token")
    with_token = GeoBiodiversityServer()
    iucn = [s for s in with_token.sources if s.name == "iucn"]
    assert len(iucn) == 1 and iucn[0].token == "secret-token"

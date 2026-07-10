"""Catalog/credential registration and error-mapping tests for ``geo-stac``.

Covers task 5.2 of the Geospatial Power Pack spec:

* Req 2.1 / 11.3 - :meth:`GeoStacServer.catalog_entries` registers the
  ``stac_search`` capability naming ``geo-stac`` as provider, with name,
  pillar, description, and Openness_Tier.
* Req 16.1 - :meth:`GeoStacServer.required_credentials` declares the wrapped
  sources' ``mcp.json`` keys, both Optional (open access works without them),
  so the startup guard never blocks (Req 16.5).
* Req 11.2 - wrapped STAC API errors are mapped onto the ``Error_Taxonomy``.
* Req 7.11 - an unreachable / >30s source yields an availability (``NETWORK``)
  error identifying the source, with no partial results.
* Req 7.8 - a missing/invalid proprietary credential yields an
  ``AuthenticationError`` naming the credential, with no partial data.

All transport behavior is driven through ``geo-common``'s
:class:`~geo_common.http.HttpClient` wired to an ``httpx.MockTransport`` with a
no-op sleep, so retries/backoff run instantly and deterministically.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from geo_common.errors import (
    AuthenticationError,
    ErrorCategory,
    NetworkError,
    ValidationError,
)
from geo_common.http import HttpClient
from geo_common.models import (
    CredentialClassification,
    OpennessTier,
)
from geo_common.retry import RetryPolicy

from geo_stac.server import GeoStacServer

VALID_BBOX = (-122.6, 37.6, -122.3, 37.9)
VALID_RANGE = ("2023-01-01T00:00:00Z", "2023-02-01T00:00:00Z")

PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


async def _noop_sleep(_seconds: float) -> None:
    """Async sleep that returns immediately so retries don't delay tests."""
    return None


def _client(handler: Any, *, max_attempts: int = 3) -> HttpClient:
    return HttpClient(
        RetryPolicy(max_attempts=max_attempts),
        transport=httpx.MockTransport(handler),
        sleep=_noop_sleep,
    )


# --- Catalog + credential registration ------------------------------------


def test_catalog_entries_register_stac_search_as_geo_stac_provider() -> None:
    """Req 2.1 / 11.3: the catalog entry names geo-stac as provider."""
    server = GeoStacServer()
    entries = server.catalog_entries()

    assert {e.name for e in entries} == {
        "stac_search",
        "stac_search_multi",
        "list_collections",
    }
    entry = next(e for e in entries if e.name == "stac_search")
    assert entry.name == "stac_search"
    assert entry.pillar == "A"
    assert entry.provider_server == "geo-stac"
    assert entry.openness_tier is OpennessTier.OPEN
    assert 1 <= len(entry.capability_description) <= 500


def test_required_credentials_are_declared_and_optional() -> None:
    """Req 16.1: both wrapped-source keys are declared and classified Optional."""
    server = GeoStacServer()
    specs = server.required_credentials()

    keys = {spec.mcp_json_key for spec in specs}
    assert keys == {"PC_SDK_SUBSCRIPTION_KEY", "EARTHDATA_TOKEN"}
    assert all(
        spec.classification is CredentialClassification.OPTIONAL for spec in specs
    )


def test_server_starts_with_no_credentials_configured() -> None:
    """Req 16.5: only Optional credentials -> startup is never blocked."""
    server = GeoStacServer()
    server.start(configured_keys=[])  # nothing configured
    assert server.started is True


# --- Availability error mapping (Req 7.11) ---------------------------------


async def test_unreachable_source_yields_availability_error_without_partial() -> None:
    """Req 7.11: an unreachable source -> NETWORK error naming the source."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    server = GeoStacServer(_client(handler), api_url=PLANETARY_COMPUTER_URL)
    try:
        with pytest.raises(NetworkError) as exc_info:
            await server.stac_search(bbox=VALID_BBOX, datetime_range=VALID_RANGE)
    finally:
        await server.aclose()

    err = exc_info.value
    assert err.category is ErrorCategory.NETWORK
    assert err.source == "planetarycomputer.microsoft.com"
    assert err.detail is not None
    assert err.detail.get("source") == "planetarycomputer.microsoft.com"
    assert err.detail.get("partial_results") is False


async def test_timeout_yields_availability_error() -> None:
    """Req 7.11: a per-request timeout maps onto the NETWORK availability error."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    server = GeoStacServer(_client(handler))
    try:
        with pytest.raises(NetworkError) as exc_info:
            await server.stac_search(bbox=VALID_BBOX, datetime_range=VALID_RANGE)
    finally:
        await server.aclose()
    assert exc_info.value.category is ErrorCategory.NETWORK
    assert exc_info.value.detail.get("partial_results") is False


# --- Authentication error mapping (Req 7.8) --------------------------------


async def test_invalid_credential_yields_authentication_error_naming_key() -> None:
    """Req 7.8: a 401 from a credentialed source names the mcp.json key."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "invalid subscription key"})

    server = GeoStacServer(_client(handler), api_url=PLANETARY_COMPUTER_URL)
    try:
        with pytest.raises(AuthenticationError) as exc_info:
            await server.stac_search(bbox=VALID_BBOX, datetime_range=VALID_RANGE)
    finally:
        await server.aclose()

    err = exc_info.value
    assert err.category is ErrorCategory.AUTHENTICATION
    assert err.detail is not None
    assert err.detail.get("missing_or_invalid_key") == "PC_SDK_SUBSCRIPTION_KEY"
    assert err.detail.get("partial_results") is False
    # The named key appears in the human message too.
    assert "PC_SDK_SUBSCRIPTION_KEY" in str(err)


async def test_authentication_error_for_unknown_source_still_authentication() -> None:
    """A 401 from a source with no known key is still an authentication error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "unauthorized"})

    # Default (Earth Search) host has no credential mapping.
    server = GeoStacServer(_client(handler))
    try:
        with pytest.raises(AuthenticationError) as exc_info:
            await server.stac_search(bbox=VALID_BBOX, datetime_range=VALID_RANGE)
    finally:
        await server.aclose()
    err = exc_info.value
    assert err.category is ErrorCategory.AUTHENTICATION
    assert err.detail.get("partial_results") is False


# --- Other taxonomy mappings pass through (Req 11.2) -----------------------


async def test_upstream_5xx_maps_to_upstream_category() -> None:
    """Req 11.2: a persistent upstream 5xx maps onto the UPSTREAM category."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="service unavailable")

    server = GeoStacServer(_client(handler))
    try:
        with pytest.raises(Exception) as exc_info:
            await server.stac_search(bbox=VALID_BBOX, datetime_range=VALID_RANGE)
    finally:
        await server.aclose()
    assert exc_info.value.category is ErrorCategory.UPSTREAM


async def test_validation_error_passes_through_unchanged() -> None:
    """Req 7.12: malformed parameters short-circuit as a validation error."""
    server = GeoStacServer()
    try:
        with pytest.raises(ValidationError) as exc_info:
            await server.stac_search(
                bbox=(-200.0, 0.0, 10.0, 10.0),  # longitude out of range
                datetime_range=VALID_RANGE,
            )
    finally:
        await server.aclose()
    err = exc_info.value
    assert err.category is ErrorCategory.VALIDATION
    assert err.detail.get("parameter") == "bbox"

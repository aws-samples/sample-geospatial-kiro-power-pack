"""Tests for ``geo-ogc`` ``ogc_features`` (OGC API - Features connector).

Drives the connector through a fully wired :class:`GeoOgcServer` whose shared
:class:`HttpClient` is backed by an ``httpx.MockTransport`` returning a canned
OGC API - Features GeoJSON payload, so the request shape, parsing, and
validation guards are exercised with no real network I/O. The repo's
``asyncio_mode = "auto"`` lets the ``async def`` tests run directly.
"""

from __future__ import annotations

import httpx
import pytest

from geo_common.errors import ErrorCategory, NetworkError, UpstreamError, ValidationError
from geo_common.http import HttpClient
from geo_common.models import CredentialClassification, OpennessTier
from geo_common.retry import RetryPolicy

from geo_ogc.features import MAX_FEATURES
from geo_ogc.models import FeatureCollectionResult
from geo_ogc.server import INSTALL_COMMAND, GeoOgcServer

ENDPOINT = "https://demo.pygeoapi.io/master"
COLLECTION = "lakes"
BBOX = (-130.0, 20.0, -60.0, 55.0)

_PAYLOAD = {
    "type": "FeatureCollection",
    "numberReturned": 2,
    "numberMatched": 17,
    "features": [
        {
            "type": "Feature",
            "id": 1,
            "geometry": {"type": "Point", "coordinates": [-122.0, 37.0]},
            "properties": {"name": "Lake A"},
        },
        {
            "type": "Feature",
            "id": 2,
            "geometry": {"type": "Point", "coordinates": [-121.0, 38.0]},
            "properties": {"name": "Lake B"},
        },
    ],
}


def _client(handler, *, max_attempts: int = 1) -> HttpClient:
    return HttpClient(RetryPolicy(max_attempts=max_attempts), transport=httpx.MockTransport(handler))


def _server(handler, **kwargs) -> GeoOgcServer:
    return GeoOgcServer(http=_client(handler, **kwargs))


# --- Happy path -------------------------------------------------------------


async def test_fetches_features_and_builds_items_url():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["path"] = request.url.path
        seen["bbox"] = request.url.params.get("bbox")
        seen["limit"] = request.url.params.get("limit")
        return httpx.Response(200, json=_PAYLOAD)

    server = _server(handler)
    try:
        result = await server.ogc_features(
            endpoint=ENDPOINT, collection=COLLECTION, bbox=BBOX, limit=50
        )
    finally:
        await server.aclose()

    assert isinstance(result, FeatureCollectionResult)
    # Standard items URL: {endpoint}/collections/{collection}/items
    assert seen["path"] == "/master/collections/lakes/items"
    assert seen["bbox"] == "-130,20,-60,55"
    assert seen["limit"] == "50"
    assert result.number_returned == 2
    assert result.number_matched == 17
    assert result.endpoint == ENDPOINT and result.collection == COLLECTION
    assert [f["properties"]["name"] for f in result.features] == ["Lake A", "Lake B"]


async def test_datetime_filter_is_forwarded():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["datetime"] = request.url.params.get("datetime")
        return httpx.Response(200, json=_PAYLOAD)

    server = _server(handler)
    try:
        await server.ogc_features(
            endpoint=ENDPOINT, collection=COLLECTION, bbox=BBOX,
            datetime_range="2024-01-01T00:00:00Z/2024-02-01T00:00:00Z",
        )
    finally:
        await server.aclose()
    assert seen["datetime"] == "2024-01-01T00:00:00Z/2024-02-01T00:00:00Z"


async def test_limit_is_clamped_to_max():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["limit"] = request.url.params.get("limit")
        return httpx.Response(200, json=_PAYLOAD)

    server = _server(handler)
    try:
        await server.ogc_features(
            endpoint=ENDPOINT, collection=COLLECTION, bbox=BBOX, limit=999999
        )
    finally:
        await server.aclose()
    assert seen["limit"] == str(MAX_FEATURES)


async def test_number_returned_defaults_to_feature_count():
    """When the service omits numberReturned, it is derived from the features."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": None, "properties": {}},
        ]})

    server = _server(handler)
    try:
        result = await server.ogc_features(endpoint=ENDPOINT, collection=COLLECTION, bbox=BBOX)
    finally:
        await server.aclose()
    assert result.number_returned == 1
    assert result.number_matched is None


# --- Validation guards (Req 7.12) ------------------------------------------


@pytest.mark.parametrize(
    "kwargs, param",
    [
        ({"endpoint": "", "collection": COLLECTION, "bbox": BBOX}, "endpoint"),
        ({"endpoint": "ftp://x", "collection": COLLECTION, "bbox": BBOX}, "endpoint"),
        ({"endpoint": ENDPOINT, "collection": "  ", "bbox": BBOX}, "collection"),
        ({"endpoint": ENDPOINT, "collection": COLLECTION, "bbox": (1, 2, 3)}, "bbox"),
        ({"endpoint": ENDPOINT, "collection": COLLECTION, "bbox": (200, 0, 201, 1)}, "bbox"),
        ({"endpoint": ENDPOINT, "collection": COLLECTION, "bbox": (10, 0, 5, 1)}, "bbox"),
        ({"endpoint": ENDPOINT, "collection": COLLECTION, "bbox": BBOX, "limit": 0}, "limit"),
    ],
)
async def test_malformed_request_is_validation_error(kwargs, param):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        calls["n"] += 1
        return httpx.Response(200, json=_PAYLOAD)

    server = _server(handler)
    try:
        with pytest.raises(ValidationError) as exc_info:
            await server.ogc_features(**kwargs)
    finally:
        await server.aclose()
    assert exc_info.value.category is ErrorCategory.VALIDATION
    assert exc_info.value.detail.get("parameter") == param
    assert calls["n"] == 0  # nothing queried


# --- Upstream / availability errors ----------------------------------------


async def test_missing_collection_is_upstream_not_found_message():
    server = _server(lambda r: httpx.Response(404, json={"code": "NotFound"}))
    try:
        with pytest.raises(UpstreamError) as exc_info:
            await server.ogc_features(endpoint=ENDPOINT, collection="nope", bbox=BBOX)
    finally:
        await server.aclose()
    assert exc_info.value.detail.get("status_code") == 404


async def test_unreachable_service_is_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    server = _server(handler)
    try:
        with pytest.raises(NetworkError) as exc_info:
            await server.ogc_features(endpoint=ENDPOINT, collection=COLLECTION, bbox=BBOX)
    finally:
        await server.aclose()
    assert exc_info.value.category is ErrorCategory.NETWORK


# --- Server scaffold + catalog + credentials -------------------------------


def test_server_registers_tool_and_metadata():
    server = GeoOgcServer()
    assert server.server_name == "geo-ogc"
    assert server.pillar == "A"
    assert "ogc_features" in server.tool_names()


def test_catalog_entry_names_geo_ogc_as_open_provider():
    server = GeoOgcServer()
    entries = server.catalog_entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.name == "ogc_features"
    assert entry.provider_server == "geo-ogc"
    assert entry.openness_tier is OpennessTier.OPEN
    assert entry.install_command == INSTALL_COMMAND
    assert 1 <= len(entry.capability_description) <= 500


def test_open_server_declares_no_credentials_and_starts():
    server = GeoOgcServer()
    assert server.required_credentials() == []
    server.start(configured_keys=[])
    assert server.started is True

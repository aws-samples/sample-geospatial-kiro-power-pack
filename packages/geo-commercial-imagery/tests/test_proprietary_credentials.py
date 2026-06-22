"""Credential + request-shape tests for ``geo-commercial-imagery`` (Maxar + Planet).

The server routes ``search_commercial`` / ``order_scene`` by ``provider``:
**MAXAR** via Sentinel Hub TPDI (OAuth client-credentials) and **PLANET** via
Planet's own Data + Orders APIs (``PL_API_KEY``). These tests pin, against
``httpx.MockTransport`` (no network):

* the per-provider **credential guard** (Req 7.8) - a missing/invalid credential
  raises an ``AuthenticationError`` naming the right ``mcp.json`` key, no data;
* the **request shape** each backend constructs (TPDI geometry+productBands;
  Planet AndFilter + item_types + Orders products);
* the **ordering safety** difference (Maxar create-only by default; Planet placed
  only on ``confirm=True``);
* provider validation (Airbus is no longer supported here); and
* upstream error-body surfacing for both providers.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Callable, Dict, List, Tuple

import httpx
import pytest

from geo_common.errors import AuthenticationError, ErrorCategory, GeoError
from geo_common.http import HttpClient
from geo_common.retry import RetryPolicy

from geo_commercial_imagery import (
    CREDENTIAL_KEYS,
    DEFAULT_ORDERS_URL,
    DEFAULT_SEARCH_URL,
    DEFAULT_TOKEN_URL,
    PLANET_API_KEY_KEY,
    PLANET_DATA_SEARCH_URL,
    PLANET_ORDERS_URL,
    SENTINELHUB_CLIENT_ID_KEY,
    SENTINELHUB_CLIENT_SECRET_KEY,
    GeoCommercialImageryServer,
)

CLIENT_ID = "test-client-id"
CLIENT_SECRET = "test-client-secret"
PLANET_KEY = "test-planet-key"
SF_BBOX: Tuple[float, float, float, float] = (-122.6, 37.6, -122.3, 37.9)
TIME_RANGE = ("2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z")
SCENE_ID = "scene-abc-123"
EXPECTED_POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [
            [-122.6, 37.6],
            [-122.3, 37.6],
            [-122.3, 37.9],
            [-122.6, 37.9],
            [-122.6, 37.6],
        ]
    ],
}


async def _noop_sleep(_seconds: float) -> None:
    return None


@pytest.fixture(autouse=True)
def _clear_credential_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no ambient credentials leak in from the environment."""
    for key in (SENTINELHUB_CLIENT_ID_KEY, SENTINELHUB_CLIENT_SECRET_KEY, PLANET_API_KEY_KEY):
        monkeypatch.delenv(key, raising=False)


def _server(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    client_id: Any = CLIENT_ID,
    client_secret: Any = CLIENT_SECRET,
    planet_api_key: Any = PLANET_KEY,
    max_attempts: int = 2,
) -> GeoCommercialImageryServer:
    http = HttpClient(
        RetryPolicy(max_attempts=max_attempts),
        transport=httpx.MockTransport(handler),
        sleep=_noop_sleep,
    )
    return GeoCommercialImageryServer(
        http=http,
        client_id=client_id,
        client_secret=client_secret,
        planet_api_key=planet_api_key,
    )


def _maxar_search_handler(seen: List[Dict[str, Any]]):
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == DEFAULT_TOKEN_URL:
            return httpx.Response(200, json={"access_token": "valid-token"})
        seen.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200, json={"features": [{"catalogID": "10300100", "bbox": list(SF_BBOX)}]}
        )

    return handler


# ===========================================================================
# Provider validation
# ===========================================================================


@pytest.mark.parametrize("bad_provider", ["AIRBUS", "LANDSAT", "", "  "])
async def test_unsupported_provider_is_validation_error(bad_provider: str) -> None:
    server = _server(lambda r: httpx.Response(200, json={}))
    try:
        with pytest.raises(GeoError) as exc_info:
            await server.search_commercial(
                provider=bad_provider, bbox=SF_BBOX, datetime_range=TIME_RANGE
            )
    finally:
        await server.aclose()
    err = exc_info.value
    assert err.category is ErrorCategory.VALIDATION
    assert err.detail is not None
    assert err.detail.get("parameter") == "provider"


# ===========================================================================
# MAXAR (Sentinel Hub TPDI)
# ===========================================================================


@pytest.mark.parametrize(
    ("client_id", "client_secret", "expected_missing"),
    [
        (None, None, [SENTINELHUB_CLIENT_ID_KEY, SENTINELHUB_CLIENT_SECRET_KEY]),
        (CLIENT_ID, None, [SENTINELHUB_CLIENT_SECRET_KEY]),
        (None, CLIENT_SECRET, [SENTINELHUB_CLIENT_ID_KEY]),
    ],
)
async def test_maxar_missing_credential_names_key_before_network(
    client_id: Any, client_secret: Any, expected_missing: List[str]
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no network call without credentials")

    server = _server(handler, client_id=client_id, client_secret=client_secret)
    try:
        with pytest.raises(AuthenticationError) as exc_info:
            await server.search_commercial(
                provider="MAXAR", bbox=SF_BBOX, datetime_range=TIME_RANGE
            )
    finally:
        await server.aclose()
    err = exc_info.value
    assert err.detail is not None
    assert err.detail.get("missing_credentials") == expected_missing


async def test_maxar_token_401_maps_to_authentication() -> None:
    server = _server(lambda r: httpx.Response(401, json={"error": "invalid_client"}))
    try:
        with pytest.raises(AuthenticationError) as exc_info:
            await server.search_commercial(
                provider="MAXAR", bbox=SF_BBOX, datetime_range=TIME_RANGE
            )
    finally:
        await server.aclose()
    assert exc_info.value.detail.get("invalid_credentials") == list(CREDENTIAL_KEYS)


async def test_maxar_search_request_shape_and_results() -> None:
    seen: List[Dict[str, Any]] = []
    server = _server(_maxar_search_handler(seen))
    try:
        items = await server.search_commercial(
            provider="maxar",
            bbox=SF_BBOX,
            datetime_range=TIME_RANGE,
            max_cloud_coverage=30,
        )
    finally:
        await server.aclose()

    assert [item.id for item in items] == ["10300100"]
    body = seen[0]
    assert body["provider"] == "MAXAR"
    assert body["bounds"] == {"geometry": EXPECTED_POLYGON}
    data_obj = body["data"][0]
    assert data_obj["productBands"] == "4BB"
    assert data_obj["dataFilter"]["timeRange"] == {
        "from": "2024-01-01T00:00:00.000Z",
        "to": "2024-02-01T00:00:00.000Z",
    }
    assert data_obj["dataFilter"]["maxCloudCoverage"] == 30


async def test_maxar_date_only_range_normalized() -> None:
    seen: List[Dict[str, Any]] = []
    server = _server(_maxar_search_handler(seen))
    try:
        await server.search_commercial(
            provider="MAXAR", bbox=SF_BBOX, datetime_range=("2024-01-01", "2024-01-31")
        )
    finally:
        await server.aclose()
    assert seen[0]["data"][0]["dataFilter"]["timeRange"] == {
        "from": "2024-01-01T00:00:00.000Z",
        "to": "2024-01-31T23:59:59.999Z",
    }


async def test_maxar_order_create_only_by_default() -> None:
    seen: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == DEFAULT_TOKEN_URL:
            return httpx.Response(200, json={"access_token": "valid-token"})
        seen.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"id": "order-99", "status": "CREATED", "sqkm": 12.5})

    server = _server(handler)
    try:
        confirmation = await server.order_scene(
            provider="MAXAR", scene_ids=[SCENE_ID], bbox=SF_BBOX
        )
    finally:
        await server.aclose()

    assert confirmation.order_id == "order-99"
    assert confirmation.provider == "MAXAR"
    assert confirmation.confirmed is False
    assert confirmation.sqkm == 12.5
    body = seen[0]
    assert body["input"]["provider"] == "MAXAR"
    assert body["input"]["bounds"]["geometry"]["type"] == "Polygon"
    assert body["input"]["data"][0]["selectedImages"] == [SCENE_ID]


async def test_maxar_order_requires_bbox() -> None:
    server = _server(lambda r: httpx.Response(200, json={"access_token": "t"}))
    try:
        with pytest.raises(GeoError) as exc_info:
            await server.order_scene(provider="MAXAR", scene_ids=[SCENE_ID])
    finally:
        await server.aclose()
    assert exc_info.value.category is ErrorCategory.VALIDATION
    assert exc_info.value.detail.get("parameter") == "bbox"


async def test_maxar_400_surfaces_upstream_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == DEFAULT_TOKEN_URL:
            return httpx.Response(200, json={"access_token": "valid-token"})
        return httpx.Response(400, json={"error": {"message": "Bounds required", "code": 400}})

    server = _server(handler)
    try:
        with pytest.raises(GeoError) as exc_info:
            await server.search_commercial(
                provider="MAXAR", bbox=SF_BBOX, datetime_range=TIME_RANGE
            )
    finally:
        await server.aclose()
    err = exc_info.value
    assert err.category is ErrorCategory.VALIDATION
    assert "Bounds required" in str(err)
    assert err.detail.get("sh_message") == "Bounds required"


# ===========================================================================
# PLANET (Planet Data + Orders APIs)
# ===========================================================================


async def test_planet_missing_api_key_names_key_before_network() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no network call without the Planet key")

    server = _server(handler, planet_api_key=None)
    try:
        with pytest.raises(AuthenticationError) as exc_info:
            await server.search_commercial(
                provider="PLANET", bbox=SF_BBOX, datetime_range=TIME_RANGE
            )
    finally:
        await server.aclose()
    err = exc_info.value
    assert err.detail is not None
    assert err.detail.get("missing_credentials") == [PLANET_API_KEY_KEY]


async def test_planet_search_request_shape_and_auth() -> None:
    seen: List[Dict[str, Any]] = []
    seen_auth: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization", ""))
        seen.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "features": [
                    {
                        "id": "20240115_planet_1",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[-122.6, 37.6], [-122.3, 37.6], [-122.3, 37.9], [-122.6, 37.9], [-122.6, 37.6]]
                            ],
                        },
                        "properties": {"acquired": "2024-01-15T18:00:00Z", "cloud_cover": 0.1},
                    }
                ]
            },
        )

    server = _server(handler)
    try:
        items = await server.search_commercial(
            provider="PLANET",
            bbox=SF_BBOX,
            datetime_range=TIME_RANGE,
            max_cloud_coverage=30,
        )
    finally:
        await server.aclose()

    assert [item.id for item in items] == ["20240115_planet_1"]
    assert items[0].datetime == "2024-01-15T18:00:00Z"
    # HTTP Basic auth with the API key as the username.
    expected = "Basic " + base64.b64encode(("%s:" % PLANET_KEY).encode()).decode()
    assert seen_auth[0] == expected
    # Planet Data API AndFilter shape.
    body = seen[0]
    assert body["item_types"] == ["PSScene"]
    flt = body["filter"]
    assert flt["type"] == "AndFilter"
    types = {c["type"]: c for c in flt["config"]}
    assert types["GeometryFilter"]["config"] == EXPECTED_POLYGON
    assert types["DateRangeFilter"]["field_name"] == "acquired"
    assert types["DateRangeFilter"]["config"] == {
        "gte": "2024-01-01T00:00:00.000Z",
        "lte": "2024-02-01T00:00:00.000Z",
    }
    # 30% cloud → 0.3 in Planet's 0–1 scale.
    assert types["RangeFilter"]["config"] == {"lte": 0.3}


async def test_planet_search_paginates_via_next_link() -> None:
    page2_url = "https://api.planet.com/data/v1/searches/abc/results?_page=2"

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == PLANET_DATA_SEARCH_URL:
            return httpx.Response(
                200,
                json={
                    "features": [{"id": "a", "properties": {"acquired": "2024-01-02T00:00:00Z"}}],
                    "_links": {"_next": page2_url},
                },
            )
        return httpx.Response(
            200,
            json={"features": [{"id": "b", "properties": {"acquired": "2024-01-03T00:00:00Z"}}]},
        )

    server = _server(handler)
    try:
        items = await server.search_commercial(
            provider="PLANET", bbox=SF_BBOX, datetime_range=TIME_RANGE
        )
    finally:
        await server.aclose()
    assert [i.id for i in items] == ["a", "b"]


async def test_planet_order_requires_confirm() -> None:
    """A Planet order has no draft; confirm=False is refused (no order placed)."""
    seen: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        seen.append(str(request.url))
        return httpx.Response(202, json={"id": "x", "state": "queued"})

    server = _server(handler)
    try:
        with pytest.raises(GeoError) as exc_info:
            await server.order_scene(
                provider="PLANET", scene_ids=[SCENE_ID], confirm=False
            )
    finally:
        await server.aclose()
    assert exc_info.value.category is ErrorCategory.VALIDATION
    assert exc_info.value.detail.get("parameter") == "confirm"
    assert seen == []  # nothing was posted


async def test_planet_order_places_on_confirm() -> None:
    seen: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            202,
            json={"id": "planet-order-1", "state": "queued", "created_on": "2024-06-01T00:00:00Z"},
        )

    server = _server(handler)
    try:
        confirmation = await server.order_scene(
            provider="PLANET",
            scene_ids=["a", "b"],
            item_type="PSScene",
            product_bundle="analytic_udm2",
            confirm=True,
        )
    finally:
        await server.aclose()

    assert confirmation.order_id == "planet-order-1"
    assert confirmation.provider == "PLANET"
    assert confirmation.status == "queued"
    assert confirmation.confirmed is True
    product = seen[0]["products"][0]
    assert product["item_ids"] == ["a", "b"]
    assert product["item_type"] == "PSScene"
    assert product["product_bundle"] == "analytic_udm2"


async def test_planet_401_maps_to_authentication() -> None:
    server = _server(lambda r: httpx.Response(401, json={"general": [{"message": "Invalid API key"}]}))
    try:
        with pytest.raises(AuthenticationError) as exc_info:
            await server.search_commercial(
                provider="PLANET", bbox=SF_BBOX, datetime_range=TIME_RANGE
            )
    finally:
        await server.aclose()
    assert exc_info.value.detail.get("invalid_credentials") == [PLANET_API_KEY_KEY]


async def test_planet_400_surfaces_upstream_message() -> None:
    server = _server(
        lambda r: httpx.Response(
            400, json={"general": [{"message": "Unknown item type"}], "field": {}}
        )
    )
    try:
        with pytest.raises(GeoError) as exc_info:
            await server.search_commercial(
                provider="PLANET", bbox=SF_BBOX, datetime_range=TIME_RANGE
            )
    finally:
        await server.aclose()
    err = exc_info.value
    assert err.category is ErrorCategory.VALIDATION
    assert "Unknown item type" in str(err)
    assert err.detail.get("planet_message") == "Unknown item type"


async def test_planet_api_key_value_never_echoed() -> None:
    secret = "super-secret-planet-key-value"
    server = _server(
        lambda r: httpx.Response(401, json={"general": [{"message": "nope"}]}),
        planet_api_key=secret,
    )
    try:
        with pytest.raises(AuthenticationError) as exc_info:
            await server.search_commercial(
                provider="PLANET", bbox=SF_BBOX, datetime_range=TIME_RANGE
            )
    finally:
        await server.aclose()
    err = exc_info.value
    assert secret not in str(err)
    assert secret not in repr(err)
    assert secret not in str(err.detail)


# ===========================================================================
# Catalog + credential declaration
# ===========================================================================


def test_required_credentials_cover_both_providers() -> None:
    server = GeoCommercialImageryServer()
    keys = {spec.mcp_json_key for spec in server.required_credentials()}
    assert keys == {
        SENTINELHUB_CLIENT_ID_KEY,
        SENTINELHUB_CLIENT_SECRET_KEY,
        PLANET_API_KEY_KEY,
    }


def test_tools_registered() -> None:
    server = GeoCommercialImageryServer()
    for tool in ("search_commercial", "order_scene"):
        assert tool in server.tool_names()

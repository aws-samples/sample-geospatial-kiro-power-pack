"""Unit / smoke tests for ``geo_stac.stac_search`` (Requirements 7.1, 7.12).

These tests drive :func:`geo_stac.search.stac_search` through ``geo-common``'s
:class:`~geo_common.http.HttpClient` wired to an ``httpx.MockTransport`` so the
STAC API round-trip runs deterministically with no real network I/O.

Coverage:

* Req 7.1 - a successful search returns :class:`StacItem` objects, each
  carrying asset references and spatio-temporal metadata; the per-response
  result is capped at the configured maximum of 1,000 items.
* Req 7.12 - malformed parameters (bad bbox shape, out-of-range coordinate,
  inverted extent, start-after-end range, non-positive limit) are rejected with
  a :class:`~geo_common.errors.ValidationError` before any request is sent.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest

from geo_common.errors import ErrorCategory, ValidationError
from geo_common.http import HttpClient

from geo_stac.search import MAX_ITEMS, StacItem, stac_search

VALID_BBOX = (-122.6, 37.6, -122.3, 37.9)  # San Francisco-ish, EPSG:4326
VALID_RANGE = ("2023-01-01T00:00:00Z", "2023-02-01T00:00:00Z")


def _feature(item_id: str) -> Dict[str, Any]:
    """A minimal but realistic STAC feature with assets + datetime + bbox."""
    return {
        "type": "Feature",
        "id": item_id,
        "bbox": [-122.6, 37.6, -122.3, 37.9],
        "geometry": {"type": "Point", "coordinates": [-122.45, 37.75]},
        "properties": {
            "datetime": "2023-01-15T18:30:00Z",
            "eo:cloud_cover": 4.2,
        },
        "assets": {
            "visual": {
                "href": "https://example.com/assets/visual.tif",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            }
        },
    }


def _feature_collection(n: int) -> Dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [_feature(f"item-{i}") for i in range(n)],
    }


def _client_returning(payload: Dict[str, Any]) -> HttpClient:
    """Build an :class:`HttpClient` whose mock transport returns ``payload``."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return HttpClient(transport=httpx.MockTransport(handler))


async def test_returns_items_with_assets_and_spatiotemporal_metadata() -> None:
    """Req 7.1: each returned item has assets + spatial + temporal metadata."""
    client = _client_returning(_feature_collection(3))
    try:
        items = await stac_search(
            bbox=VALID_BBOX,
            datetime_range=VALID_RANGE,
            http=client,
        )
    finally:
        await client.aclose()

    assert len(items) == 3
    for item in items:
        assert isinstance(item, StacItem)
        assert item.id
        # spatial metadata
        assert len(item.bbox) == 4
        # temporal metadata
        assert item.datetime == "2023-01-15T18:30:00Z"
        # asset references (Req 7.1)
        assert "visual" in item.assets
        assert item.assets["visual"]["href"].endswith("visual.tif")


async def test_sends_bbox_datetime_and_limit_to_search_endpoint() -> None:
    """The wrapped STAC POST carries the spatial/temporal/limit parameters."""
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_feature_collection(1))

    client = HttpClient(transport=httpx.MockTransport(handler))
    try:
        await stac_search(
            bbox=VALID_BBOX,
            datetime_range=VALID_RANGE,
            collections=["sentinel-2-l2a"],
            limit=50,
            http=client,
        )
    finally:
        await client.aclose()

    assert captured["url"].endswith("/search")
    body = captured["body"]
    assert body["bbox"] == list(VALID_BBOX)
    assert body["datetime"] == "2023-01-01T00:00:00Z/2023-02-01T00:00:00Z"
    assert body["limit"] == 50
    assert body["collections"] == ["sentinel-2-l2a"]


async def test_caps_results_at_configured_maximum() -> None:
    """Req 7.1: at most MAX_ITEMS (1,000) items are returned per response."""
    # Upstream returns more than the cap; result must be clamped.
    client = _client_returning(_feature_collection(MAX_ITEMS + 25))
    try:
        items = await stac_search(
            bbox=VALID_BBOX,
            datetime_range=VALID_RANGE,
            limit=MAX_ITEMS + 500,  # over-large request clamps to the cap
            http=client,
        )
    finally:
        await client.aclose()

    assert len(items) == MAX_ITEMS


async def test_empty_result_set_is_supported() -> None:
    client = _client_returning(_feature_collection(0))
    try:
        items = await stac_search(
            bbox=VALID_BBOX,
            datetime_range=VALID_RANGE,
            http=client,
        )
    finally:
        await client.aclose()
    assert items == []


# --- Validation (Requirement 7.12) -----------------------------------------


@pytest.mark.parametrize(
    "bad_bbox",
    [
        (-122.6, 37.6, -122.3),  # too few components
        (-122.6, 37.6, -122.3, 37.9, 1.0),  # too many components
        (-200.0, 37.6, -122.3, 37.9),  # longitude out of range
        (-122.6, -91.0, -122.3, 37.9),  # latitude out of range
        (-122.3, 37.6, -122.6, 37.9),  # west > east (inverted)
        (-122.6, 37.9, -122.3, 37.6),  # south > north (inverted)
        ("a", "b", "c", "d"),  # non-numeric
    ],
)
async def test_malformed_bbox_raises_validation_error(bad_bbox: Any) -> None:
    with pytest.raises(ValidationError) as exc_info:
        await stac_search(bbox=bad_bbox, datetime_range=VALID_RANGE)
    err = exc_info.value
    assert err.category is ErrorCategory.VALIDATION
    assert err.detail and err.detail.get("parameter") == "bbox"


async def test_start_after_end_raises_validation_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        await stac_search(
            bbox=VALID_BBOX,
            datetime_range=("2023-02-01T00:00:00Z", "2023-01-01T00:00:00Z"),
        )
    err = exc_info.value
    assert err.category is ErrorCategory.VALIDATION
    assert err.detail and err.detail.get("parameter") == "datetime_range"


async def test_unparseable_datetime_raises_validation_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        await stac_search(
            bbox=VALID_BBOX,
            datetime_range=("not-a-date", "2023-01-01T00:00:00Z"),
        )
    assert exc_info.value.detail.get("parameter") == "datetime_range"


@pytest.mark.parametrize("bad_limit", [0, -5])
async def test_non_positive_limit_raises_validation_error(bad_limit: int) -> None:
    with pytest.raises(ValidationError) as exc_info:
        await stac_search(
            bbox=VALID_BBOX,
            datetime_range=VALID_RANGE,
            limit=bad_limit,
        )
    assert exc_info.value.detail.get("parameter") == "limit"


async def test_validation_happens_before_any_network_call() -> None:
    """A malformed parameter must short-circuit before the transport is hit."""
    calls: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json=_feature_collection(1))

    client = HttpClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ValidationError):
            await stac_search(
                bbox=(-200.0, 0.0, 10.0, 10.0),
                datetime_range=VALID_RANGE,
                http=client,
            )
    finally:
        await client.aclose()
    assert calls == []

"""Smoke / integration tests for ``stac_search`` via :class:`GeoStacServer` (task 5.3).

Where ``test_stac_search.py`` exercises the :func:`geo_stac.search.stac_search`
function in isolation, these tests drive a small number of *representative,
end-to-end* searches through the assembled :class:`~geo_stac.server.GeoStacServer`
- the same path a running MCP server takes: the server owns the shared
:class:`~geo_common.http.HttpClient`, the ``stac_search`` tool is invoked from
the server's tool registry, and the STAC API round-trip is served by an
``httpx.MockTransport`` so the test is deterministic with no real network I/O.

Each scenario asserts the Requirement 7.1 contract holds through the full stack:
every returned item carries its **asset references** and its **spatial and
temporal metadata**.

_Requirements: 7.1_
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx

from geo_common.http import HttpClient

from geo_stac.search import StacItem
from geo_stac.server import GeoStacServer

# Three representative areas/time windows for the end-to-end searches.
SF_BBOX = (-122.6, 37.6, -122.3, 37.9)  # San Francisco Bay, EPSG:4326
SF_RANGE = ("2023-01-01T00:00:00Z", "2023-02-01T00:00:00Z")

ALPS_BBOX = (10.0, 46.0, 11.0, 47.0)  # Eastern Alps
ALPS_RANGE = ("2022-06-01T00:00:00Z", "2022-09-01T00:00:00Z")


def _sentinel_feature(item_id: str) -> Dict[str, Any]:
    """A realistic Sentinel-2 STAC feature with assets + datetime + bbox."""
    return {
        "type": "Feature",
        "id": item_id,
        "collection": "sentinel-2-l2a",
        "bbox": [-122.6, 37.6, -122.3, 37.9],
        "geometry": {"type": "Point", "coordinates": [-122.45, 37.75]},
        "properties": {
            "datetime": "2023-01-15T18:30:00Z",
            "eo:cloud_cover": 7.5,
        },
        "assets": {
            "visual": {
                "href": "https://example.com/s2/visual.tif",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            },
            "B04": {
                "href": "https://example.com/s2/B04.tif",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            },
        },
    }


def _interval_feature(item_id: str) -> Dict[str, Any]:
    """A STAC feature whose temporal metadata is an interval (no ``datetime``)."""
    return {
        "type": "Feature",
        "id": item_id,
        "collection": "landsat-c2-l2",
        "bbox": [10.0, 46.0, 11.0, 47.0],
        "geometry": {"type": "Point", "coordinates": [10.5, 46.5]},
        "properties": {
            "datetime": None,
            "start_datetime": "2022-06-10T09:00:00Z",
            "end_datetime": "2022-06-10T09:05:00Z",
        },
        "assets": {
            "red": {"href": "https://example.com/ls/red.tif", "type": "image/tiff"},
        },
    }


def _collection(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


def _server_returning(payload: Dict[str, Any]) -> GeoStacServer:
    """A :class:`GeoStacServer` whose mock transport returns ``payload``."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    http = HttpClient(transport=httpx.MockTransport(handler))
    return GeoStacServer(http=http)


async def test_server_search_returns_items_with_assets_and_spatiotemporal_metadata() -> None:
    """Req 7.1: an end-to-end search through the server yields items that each
    carry asset references and spatial + temporal metadata."""
    server = _server_returning(_collection([_sentinel_feature("S2-1"), _sentinel_feature("S2-2")]))
    try:
        items = await server.stac_search(bbox=SF_BBOX, datetime_range=SF_RANGE)
    finally:
        await server.aclose()

    assert len(items) == 2
    for item in items:
        assert isinstance(item, StacItem)
        assert item.id
        # spatial metadata (Req 7.1)
        assert len(item.bbox) == 4
        # temporal metadata (Req 7.1)
        assert item.datetime == "2023-01-15T18:30:00Z"
        # asset references (Req 7.1)
        assert set(item.assets) == {"visual", "B04"}
        assert item.assets["visual"]["href"].endswith("visual.tif")


async def test_server_search_invoked_through_tool_registry() -> None:
    """The ``stac_search`` capability is reachable via the server's tool registry
    (the MCP invocation path), and the round-trip carries the search params."""
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_collection([_sentinel_feature("S2-1")]))

    server = GeoStacServer(http=HttpClient(transport=httpx.MockTransport(handler)))
    try:
        assert "stac_search" in server.tool_names()
        tool = server.get_tool("stac_search")
        items = await tool.func(
            bbox=SF_BBOX,
            datetime_range=SF_RANGE,
            collections=["sentinel-2-l2a"],
            limit=25,
        )
    finally:
        await server.aclose()

    assert len(items) == 1
    assert items[0].assets  # asset references present (Req 7.1)
    # The wrapped STAC POST hit /search carrying the spatial/temporal/limit params.
    assert captured["url"].endswith("/search")
    assert captured["body"]["bbox"] == list(SF_BBOX)
    assert captured["body"]["datetime"] == "2023-01-01T00:00:00Z/2023-02-01T00:00:00Z"
    assert captured["body"]["limit"] == 25
    assert captured["body"]["collections"] == ["sentinel-2-l2a"]


async def test_server_search_preserves_interval_temporal_metadata() -> None:
    """Req 7.1: items whose temporal metadata is an interval still surface their
    time (start_datetime) and assets through the end-to-end server path."""
    server = _server_returning(_collection([_interval_feature("LS-1")]))
    try:
        items = await server.stac_search(bbox=ALPS_BBOX, datetime_range=ALPS_RANGE)
    finally:
        await server.aclose()

    assert len(items) == 1
    item = items[0]
    # temporal metadata falls back to the interval start (Req 7.1)
    assert item.datetime == "2022-06-10T09:00:00Z"
    # spatial metadata (Req 7.1)
    assert item.bbox == (10.0, 46.0, 11.0, 47.0)
    # asset references (Req 7.1)
    assert "red" in item.assets

"""Unit tests for ``isochrone`` (reachability polygons via Valhalla).

Drives the isochrone path through a fully wired :class:`GeoGeocodeRouteServer`
whose shared :class:`HttpClient` is backed by an ``httpx.MockTransport``
returning a canned Valhalla isochrone GeoJSON payload. No real network I/O.
"""

from __future__ import annotations

import httpx
import pytest

from geo_common.errors import ErrorCategory, GeoError, ValidationError
from geo_common.http import HttpClient
from geo_common.retry import RetryPolicy

from geo_geocode_route.isochrone import ValhallaIsochroneSource, default_isochrone_sources
from geo_geocode_route.server import GeoGeocodeRouteServer

VALHALLA_HOST = "valhalla1.openstreetmap.de"
DC = {"lon": -77.0365, "lat": 38.8977}

_ISOCHRONE = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"contour": 10, "metric": "time"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-77.05, 38.89], [-77.02, 38.89], [-77.02, 38.91], [-77.05, 38.91], [-77.05, 38.89]]],
            },
        },
        {
            "type": "Feature",
            "properties": {"contour": 20, "metric": "time"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-77.07, 38.87], [-77.00, 38.87], [-77.00, 38.93], [-77.07, 38.93], [-77.07, 38.87]]],
            },
        },
    ],
}


def _server(handler, **kwargs):
    client = HttpClient(RetryPolicy(max_attempts=1), transport=httpx.MockTransport(handler))
    return GeoGeocodeRouteServer(http=client, **kwargs)


async def test_isochrone_returns_geojson_polygons():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == VALHALLA_HOST
        assert "isochrone" in request.url.path
        import json
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_ISOCHRONE)

    server = _server(handler)
    try:
        result = await server.isochrone(origin=DC, contours_minutes=[10, 20], profile="car")
    finally:
        await server.aclose()

    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == 2
    assert result["features"][0]["geometry"]["type"] == "Polygon"
    # The request carried both contours and the car -> auto costing.
    assert captured["body"]["costing"] == "auto"
    assert [c["time"] for c in captured["body"]["contours"]] == [10.0, 20.0]


async def test_isochrone_rejects_empty_contours_before_querying():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        calls["n"] += 1
        return httpx.Response(200, json=_ISOCHRONE)

    server = _server(handler)
    try:
        with pytest.raises(ValidationError) as exc_info:
            await server.isochrone(origin=DC, contours_minutes=[])
    finally:
        await server.aclose()
    # Reaches the tool's domain validator (not a schema "required" error).
    assert exc_info.value.detail.get("parameter") == "contours_minutes"
    assert "time budget" in str(exc_info.value).lower()
    assert calls["n"] == 0


async def test_isochrone_rejects_omitted_contours_with_clear_message():
    """An omitted contours_minutes reaches the tool's validator (it is optional
    in the schema) and gets the clear 'required: provide at least one' message,
    not a bare schema 'required property' error."""
    server = _server(lambda r: httpx.Response(200, json=_ISOCHRONE))
    try:
        with pytest.raises(ValidationError) as exc_info:
            await server.isochrone(origin=DC)
    finally:
        await server.aclose()
    assert exc_info.value.detail.get("parameter") == "contours_minutes"
    assert "at least one positive time budget" in str(exc_info.value).lower()


async def test_isochrone_rejects_out_of_range_contour():
    server = _server(lambda r: httpx.Response(200, json=_ISOCHRONE))
    try:
        with pytest.raises(ValidationError):
            await server.isochrone(origin=DC, contours_minutes=[999])  # > max minutes
    finally:
        await server.aclose()


async def test_isochrone_rejects_bad_profile():
    server = _server(lambda r: httpx.Response(200, json=_ISOCHRONE))
    try:
        with pytest.raises(ValidationError):
            await server.isochrone(origin=DC, contours_minutes=[10], profile="teleport")
    finally:
        await server.aclose()


async def test_isochrone_availability_error_is_retagged():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    server = _server(handler)
    try:
        with pytest.raises(GeoError) as exc:
            await server.isochrone(origin=DC, contours_minutes=[10])
    finally:
        await server.aclose()
    assert exc.value.category in (ErrorCategory.NETWORK, ErrorCategory.UPSTREAM)
    assert exc.value.source == "valhalla"


def test_default_isochrone_sources_is_valhalla():
    sources = default_isochrone_sources()
    assert [s.name for s in sources] == ["valhalla"]
    assert isinstance(sources[0], ValhallaIsochroneSource)

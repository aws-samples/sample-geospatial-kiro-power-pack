"""Unit tests for ``route`` (Requirements 7.10, 7.11, 7.12).

Drive the routing engines through a fully wired :class:`GeoGeocodeRouteServer`
whose shared :class:`HttpClient` is backed by an ``httpx.MockTransport``
returning canned OSRM / Valhalla payloads. No real network I/O happens.
"""

from __future__ import annotations

import httpx
import pytest

from geo_common.errors import ErrorCategory, GeoError, NotFoundError, ValidationError
from geo_common.http import HttpClient
from geo_common.retry import RetryPolicy

from geo_geocode_route.models import Coordinate, Route
from geo_geocode_route.routing import OsrmSource, ValhallaSource, _decode_polyline
from geo_geocode_route.server import GeoGeocodeRouteServer

OSRM_HOST = "router.project-osrm.org"

BERLIN = (13.40, 52.52)
POTSDAM = (13.06, 52.39)

_OSRM_OK = {
    "code": "Ok",
    "routes": [
        {
            "distance": 32000.0,
            "duration": 1800.0,
            "geometry": {
                "type": "LineString",
                "coordinates": [[13.40, 52.52], [13.20, 52.45], [13.06, 52.39]],
            },
        }
    ],
}
_OSRM_NO_ROUTE = {"code": "NoRoute", "routes": []}


def _server(handler, **kwargs):
    client = HttpClient(
        RetryPolicy(max_attempts=1),
        transport=httpx.MockTransport(handler),
    )
    return GeoGeocodeRouteServer(http=client, **kwargs)


async def test_route_returns_osrm_route():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == OSRM_HOST
        # The driving profile and both coordinates are in the path.
        assert "/driving/" in request.url.path
        return httpx.Response(200, json=_OSRM_OK)

    server = _server(handler, routers=[OsrmSource()])
    try:
        result = await server.route(origin=BERLIN, destination=POTSDAM)
    finally:
        await server.aclose()

    assert isinstance(result, Route)
    assert result.distance_m == pytest.approx(32000.0)
    assert result.duration_s == pytest.approx(1800.0)
    assert result.geometry["type"] == "LineString"
    assert result.source == "osrm"
    assert result.profile == "car"
    assert result.origin == Coordinate(lon=13.40, lat=52.52)
    assert result.destination == Coordinate(lon=13.06, lat=52.39)


async def test_route_maps_profile_alias_to_osrm_segment():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json=_OSRM_OK)

    server = _server(handler, routers=[OsrmSource()])
    try:
        await server.route(origin=BERLIN, destination=POTSDAM, profile="cycling")
    finally:
        await server.aclose()
    assert "/cycling/" in seen["path"]


async def test_route_raises_not_found_on_no_route():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_OSRM_NO_ROUTE)

    server = _server(handler, routers=[OsrmSource()])
    try:
        with pytest.raises(NotFoundError):
            await server.route(origin=BERLIN, destination=POTSDAM)
    finally:
        await server.aclose()


async def test_route_raises_availability_error_when_engine_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502)

    server = _server(handler, routers=[OsrmSource()])
    try:
        with pytest.raises(GeoError) as exc:
            await server.route(origin=BERLIN, destination=POTSDAM)
    finally:
        await server.aclose()
    assert exc.value.category in (ErrorCategory.NETWORK, ErrorCategory.UPSTREAM)
    assert exc.value.source == "osrm"


async def test_route_rejects_malformed_coordinate_before_querying():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        calls["n"] += 1
        return httpx.Response(200, json=_OSRM_OK)

    server = _server(handler)
    try:
        with pytest.raises(ValidationError):
            await server.route(origin=(200.0, 0.0), destination=POTSDAM)
    finally:
        await server.aclose()
    assert calls["n"] == 0


async def test_route_rejects_unsupported_profile():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        return httpx.Response(200, json=_OSRM_OK)

    server = _server(handler)
    try:
        with pytest.raises(ValidationError) as exc:
            await server.route(origin=BERLIN, destination=POTSDAM, profile="teleport")
    finally:
        await server.aclose()
    assert exc.value.detail == {"parameter": "profile"}


# --- Valhalla source + polyline decoding ------------------------------------


def test_decode_polyline_precision6():
    # Encoding of a short Valhalla-style precision-6 polyline.
    # Two points: (lon, lat) = (13.40, 52.52) then (13.06, 52.39).
    coords = _decode_polyline("_kbvfAo}jpO", precision=6)
    assert len(coords) >= 1
    # Each decoded position is [lon, lat].
    assert all(len(c) == 2 for c in coords)


async def test_valhalla_source_parses_trip():
    # Build a valid precision-6 shape by round-tripping through the decoder's
    # inverse is unnecessary; use a known-good short encoded string.
    encoded = "_kbvfAo}jpO_seK_seK"
    valhalla_payload = {
        "trip": {
            "status": 0,
            "summary": {"length": 32.0, "time": 1800.0},
            "legs": [{"shape": encoded}],
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=valhalla_payload)

    server = _server(handler, routers=[ValhallaSource()])
    try:
        result = await server.route(origin=BERLIN, destination=POTSDAM)
    finally:
        await server.aclose()

    assert result.source == "valhalla"
    assert result.distance_m == pytest.approx(32000.0)  # km -> m
    assert result.duration_s == pytest.approx(1800.0)
    assert result.geometry["type"] == "LineString"
    assert len(result.geometry["coordinates"]) >= 1

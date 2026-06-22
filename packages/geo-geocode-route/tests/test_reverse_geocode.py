"""Unit tests for ``reverse_geocode`` (coordinate -> address; Req 7.4, 7.11, 7.12).

Drives the reverse path through a fully wired :class:`GeoGeocodeRouteServer`
whose shared :class:`HttpClient` is backed by an ``httpx.MockTransport``
returning canned Nominatim (single object) / Photon (GeoJSON) reverse payloads.
No real network I/O happens.
"""

from __future__ import annotations

import httpx
import pytest

from geo_common.errors import ErrorCategory, GeoError, NotFoundError, ValidationError
from geo_common.http import HttpClient
from geo_common.retry import RetryPolicy

from geo_geocode_route.geocoding import NominatimSource, PhotonSource
from geo_geocode_route.models import Coordinate
from geo_geocode_route.server import GeoGeocodeRouteServer

NOMINATIM_HOST = "nominatim.openstreetmap.org"
PHOTON_HOST = "photon.komoot.io"

# Nominatim reverse returns a SINGLE object (not an array).
_NOMINATIM_REVERSE = {
    "lon": "-77.0365",
    "lat": "38.8977",
    "display_name": "White House, 1600 Pennsylvania Ave NW, Washington, DC",
}
_PHOTON_REVERSE = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [2.2945, 48.8584]},
            "properties": {"name": "Eiffel Tower"},
        }
    ],
}


def _server(handler, **kwargs):
    client = HttpClient(
        RetryPolicy(max_attempts=1),
        transport=httpx.MockTransport(handler),
    )
    return GeoGeocodeRouteServer(http=client, **kwargs)


async def test_reverse_geocode_returns_address_from_nominatim():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == NOMINATIM_HOST
        assert "reverse" in request.url.path
        return httpx.Response(200, json=_NOMINATIM_REVERSE)

    server = _server(handler, geocoders=[NominatimSource()])
    try:
        result = await server.reverse_geocode(lon=-77.0365, lat=38.8977)
    finally:
        await server.aclose()

    assert isinstance(result, Coordinate)
    assert result.source == "nominatim"
    assert "White House" in result.label
    assert result.lon == pytest.approx(-77.0365)


async def test_reverse_geocode_parses_photon_geojson():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_PHOTON_REVERSE)

    server = _server(handler, geocoders=[PhotonSource()])
    try:
        result = await server.reverse_geocode(lon=2.2945, lat=48.8584)
    finally:
        await server.aclose()

    assert result.source == "photon"
    assert result.label == "Eiffel Tower"


async def test_reverse_geocode_falls_back_when_first_source_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == NOMINATIM_HOST:
            return httpx.Response(503)
        return httpx.Response(200, json=_PHOTON_REVERSE)

    server = _server(handler, geocoders=[NominatimSource(), PhotonSource()])
    try:
        result = await server.reverse_geocode(lon=2.2945, lat=48.8584)
    finally:
        await server.aclose()
    assert result.source == "photon"


async def test_reverse_geocode_not_found_when_no_match():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == NOMINATIM_HOST:
            return httpx.Response(200, json={"error": "Unable to geocode"})
        return httpx.Response(200, json={"type": "FeatureCollection", "features": []})

    server = _server(handler, geocoders=[NominatimSource(), PhotonSource()])
    try:
        with pytest.raises(NotFoundError) as exc:
            await server.reverse_geocode(lon=0.0, lat=0.0)
    finally:
        await server.aclose()
    assert exc.value.category is ErrorCategory.NOT_FOUND


async def test_reverse_geocode_rejects_out_of_range_coordinate_before_querying():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        calls["n"] += 1
        return httpx.Response(200, json=_NOMINATIM_REVERSE)

    server = _server(handler)
    try:
        with pytest.raises(ValidationError):
            await server.reverse_geocode(lon=999.0, lat=38.0)
    finally:
        await server.aclose()
    assert calls["n"] == 0  # no source queried


async def test_reverse_geocode_availability_error_when_all_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    server = _server(handler, geocoders=[NominatimSource(), PhotonSource()])
    try:
        with pytest.raises(GeoError) as exc:
            await server.reverse_geocode(lon=2.2945, lat=48.8584)
    finally:
        await server.aclose()
    assert exc.value.category in (ErrorCategory.NETWORK, ErrorCategory.UPSTREAM)
    assert exc.value.source in {"nominatim", "photon"}

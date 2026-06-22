"""Smoke tests for the wired ``geo-geocode-route`` server (Req 7.4, 7.10).

Exercise the whole path - registered tool dispatch -> validation -> source
connector -> parse - for both ``geocode`` and ``route`` through a fully wired
:class:`GeoGeocodeRouteServer` on a recording mock transport, and confirm the
server's catalog/credential declarations and startup guard.
"""

from __future__ import annotations

import httpx

from geo_common.http import HttpClient
from geo_common.models import CredentialClassification, OpennessTier
from geo_common.retry import RetryPolicy

from geo_geocode_route.geocoding import NominatimSource
from geo_geocode_route.models import Coordinate, Route
from geo_geocode_route.routing import OsrmSource
from geo_geocode_route.server import INSTALL_COMMAND, GeoGeocodeRouteServer

NOMINATIM_HOST = "nominatim.openstreetmap.org"
OSRM_HOST = "router.project-osrm.org"

_NOMINATIM_MATCH = [{"lon": "13.40", "lat": "52.52", "display_name": "Berlin"}]
_OSRM_OK = {
    "code": "Ok",
    "routes": [
        {
            "distance": 32000.0,
            "duration": 1800.0,
            "geometry": {"type": "LineString", "coordinates": [[13.40, 52.52], [13.06, 52.39]]},
        }
    ],
}


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.host == NOMINATIM_HOST:
        return httpx.Response(200, json=_NOMINATIM_MATCH)
    if request.url.host == OSRM_HOST:
        return httpx.Response(200, json=_OSRM_OK)
    return httpx.Response(404)  # pragma: no cover


def _server():
    client = HttpClient(RetryPolicy(max_attempts=1), transport=httpx.MockTransport(_handler))
    return GeoGeocodeRouteServer(
        http=client, geocoders=[NominatimSource()], routers=[OsrmSource()]
    )


async def test_geocode_then_route_through_tool_dispatch():
    """Drive ``geocode`` then ``route`` via the registered tool callables."""
    server = _server()
    try:
        geocode_tool = server.get_tool("geocode")
        route_tool = server.get_tool("route")

        origin = await geocode_tool.func(address="Berlin")
        assert isinstance(origin, Coordinate)

        result = await route_tool.func(
            origin=origin, destination=(13.06, 52.39), profile="car"
        )
    finally:
        await server.aclose()

    assert isinstance(result, Route)
    assert result.geometry["type"] == "LineString"
    assert result.source == "osrm"
    assert result.origin == Coordinate(lon=13.40, lat=52.52)


def test_registered_tools_and_metadata():
    server = GeoGeocodeRouteServer()
    assert server.tool_names() == ["geocode", "reverse_geocode", "route", "isochrone"]
    assert server.pillar == "A"
    assert server.server_name == "geo-geocode-route"


def test_catalog_entries_are_open_and_attributed():
    server = GeoGeocodeRouteServer()
    entries = {e.name: e for e in server.catalog_entries()}
    assert set(entries) == {"geocode", "reverse_geocode", "route", "isochrone"}
    for entry in entries.values():
        assert entry.openness_tier is OpennessTier.OPEN
        assert entry.provider_server == "geo-geocode-route"
        assert entry.install_command == INSTALL_COMMAND


def test_optional_credential_declared_and_server_starts():
    server = GeoGeocodeRouteServer()
    specs = server.required_credentials()
    keys = {spec.mcp_json_key for spec in specs}
    assert keys == {"AMAZON_LOCATION_API_KEY"}
    assert all(
        spec.classification is CredentialClassification.OPTIONAL for spec in specs
    )
    # No Required credentials -> startup guard passes with an empty config.
    server.start(configured_keys=[])
    assert server.started is True

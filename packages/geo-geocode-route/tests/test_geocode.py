"""Unit tests for ``geocode`` (Requirements 7.4, 7.11, 7.12).

Drive the geocoders through a fully wired :class:`GeoGeocodeRouteServer` whose
shared :class:`HttpClient` is backed by an ``httpx.MockTransport`` returning
canned Nominatim / Photon payloads. No real network I/O happens.
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

_NOMINATIM_MATCH = [
    {"lon": "-77.0365", "lat": "38.8977", "display_name": "White House, Washington, DC"}
]
_PHOTON_MATCH = {
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


async def test_geocode_returns_first_match_from_nominatim():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == NOMINATIM_HOST
        return httpx.Response(200, json=_NOMINATIM_MATCH)

    server = _server(handler, geocoders=[NominatimSource()])
    try:
        result = await server.geocode(address="1600 Pennsylvania Ave")
    finally:
        await server.aclose()

    assert isinstance(result, Coordinate)
    assert result.lon == pytest.approx(-77.0365)
    assert result.lat == pytest.approx(38.8977)
    assert result.source == "nominatim"
    assert result.label == "White House, Washington, DC"


async def test_geocode_parses_photon_geojson():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_PHOTON_MATCH)

    server = _server(handler, geocoders=[PhotonSource()])
    try:
        result = await server.geocode(address="Eiffel Tower")
    finally:
        await server.aclose()

    assert result.lon == pytest.approx(2.2945)
    assert result.lat == pytest.approx(48.8584)
    assert result.source == "photon"


async def test_geocode_falls_back_when_first_source_unavailable():
    """An unavailable source is recorded and the next source resolves it."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == NOMINATIM_HOST:
            return httpx.Response(503)  # unavailable -> UPSTREAM after retries
        return httpx.Response(200, json=_PHOTON_MATCH)

    server = _server(handler, geocoders=[NominatimSource(), PhotonSource()])
    try:
        result = await server.geocode(address="Eiffel Tower")
    finally:
        await server.aclose()

    assert result.source == "photon"


async def test_geocode_raises_not_found_when_no_source_matches():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == NOMINATIM_HOST:
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={"type": "FeatureCollection", "features": []})

    server = _server(handler, geocoders=[NominatimSource(), PhotonSource()])
    try:
        with pytest.raises(NotFoundError) as exc:
            await server.geocode(address="Nowhere at all 999999")
    finally:
        await server.aclose()
    assert exc.value.category is ErrorCategory.NOT_FOUND


async def test_geocode_raises_availability_error_when_all_sources_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    server = _server(handler, geocoders=[NominatimSource(), PhotonSource()])
    try:
        with pytest.raises(GeoError) as exc:
            await server.geocode(address="Eiffel Tower")
    finally:
        await server.aclose()
    assert exc.value.category in (ErrorCategory.NETWORK, ErrorCategory.UPSTREAM)
    # The error names the logical source, not just the host.
    assert exc.value.source in {"nominatim", "photon"}


async def test_geocode_rejects_unparseable_address_before_querying():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        calls["n"] += 1
        return httpx.Response(200, json=_NOMINATIM_MATCH)

    server = _server(handler)
    try:
        with pytest.raises(ValidationError):
            await server.geocode(address="   ")
    finally:
        await server.aclose()
    assert calls["n"] == 0  # no source was queried


# --- Amazon Location (credentialed geocoder) -------------------------------

from geo_common.errors import AuthenticationError  # noqa: E402
from geo_geocode_route.geocoding import AmazonLocationSource  # noqa: E402

_AMAZON_MATCH = {
    "ResultItems": [
        {
            "Title": "1600 Pennsylvania Ave NW, Washington, DC",
            "Position": [-77.0365, 38.8977],
            "Address": {"Label": "1600 Pennsylvania Ave NW, Washington, DC, USA"},
        }
    ]
}


async def test_amazon_location_geocodes_with_api_key():
    """With an API key, Amazon Location authenticates via the ``key`` query
    param and parses ``ResultItems[].Position`` into a coordinate."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["host"] = request.url.host
        seen["path"] = request.url.path
        seen["key"] = request.url.params.get("key")
        return httpx.Response(200, json=_AMAZON_MATCH)

    server = _server(
        handler, geocoders=[AmazonLocationSource(api_key="test-key", region="us-east-1")]
    )
    try:
        result = await server.geocode(
            address="1600 Pennsylvania Ave", source="amazon-location"
        )
    finally:
        await server.aclose()

    assert seen["host"] == "places.geo.us-east-1.amazonaws.com"
    assert seen["path"] == "/v2/geocode"
    assert seen["key"] == "test-key"  # API key sent as the auth query param
    assert result.source == "amazon-location"
    assert result.lon == pytest.approx(-77.0365)
    assert result.lat == pytest.approx(38.8977)
    assert result.label.startswith("1600 Pennsylvania Ave NW")


async def test_amazon_location_reverse_geocodes_with_api_key():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/reverse-geocode"
        return httpx.Response(200, json=_AMAZON_MATCH)

    server = _server(handler, geocoders=[AmazonLocationSource(api_key="test-key")])
    try:
        result = await server.reverse_geocode(
            lon=-77.0365, lat=38.8977, source="amazon-location"
        )
    finally:
        await server.aclose()
    assert result.source == "amazon-location"
    assert result.label.startswith("1600 Pennsylvania Ave NW")


async def test_amazon_location_without_key_is_authentication_error():
    """`source="amazon-location"` with no key -> AuthenticationError naming the
    mcp.json key, no request made (credential guard)."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        calls["n"] += 1
        return httpx.Response(200, json=_AMAZON_MATCH)

    server = _server(handler, geocoders=[AmazonLocationSource(api_key=None)])
    try:
        with pytest.raises(AuthenticationError) as exc:
            await server.geocode(address="1600 Pennsylvania Ave", source="amazon-location")
    finally:
        await server.aclose()

    err = exc.value
    assert err.category is ErrorCategory.AUTHENTICATION
    assert err.detail.get("mcp_json_key") == "AMAZON_LOCATION_API_KEY"
    assert "AMAZON_LOCATION_API_KEY" in str(err)
    assert calls["n"] == 0  # credential guard fires before any request


async def test_geocode_source_selector_forces_amazon_location():
    """`source="amazon-location"` queries only Amazon Location, even though the
    open geocoders would have answered (and ALS is not in the default chain)."""
    seen = {"hosts": []}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["hosts"].append(request.url.host)
        if request.url.host.endswith("amazonaws.com"):
            return httpx.Response(200, json=_AMAZON_MATCH)
        return httpx.Response(200, json=_NOMINATIM_MATCH)

    server = _server(
        handler,
        geocoders=[NominatimSource(), AmazonLocationSource(api_key="k")],
    )
    try:
        result = await server.geocode(address="1600 Pennsylvania Ave", source="amazon-location")
    finally:
        await server.aclose()
    assert result.source == "amazon-location"
    # Only Amazon Location was contacted — Nominatim was skipped.
    assert all(h.endswith("amazonaws.com") for h in seen["hosts"])


async def test_geocode_unknown_source_is_validation_error():
    server = _server(lambda r: httpx.Response(200, json=_NOMINATIM_MATCH))
    try:
        with pytest.raises(ValidationError) as exc:
            await server.geocode(address="x", source="bogus")
    finally:
        await server.aclose()
    assert exc.value.detail.get("parameter") == "source"


def test_amazon_location_always_selectable_but_not_in_default_chain(monkeypatch):
    """ALS is always present (so it can be selected) but excluded from the
    default chain so open geocodes are never blocked by it."""
    monkeypatch.delenv("AMAZON_LOCATION_API_KEY", raising=False)
    server = GeoGeocodeRouteServer()
    als = [g for g in server.geocoders if g.name == "amazon-location"]
    assert len(als) == 1
    assert als[0].default_chain is False
    assert als[0].api_key is None

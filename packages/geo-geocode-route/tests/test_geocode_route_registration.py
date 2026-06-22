"""Catalog/credential registration + error sweep for ``geo-geocode-route``.

Cross-cutting registration half (task 13.6) for the Pillar A expansion server
``geo-geocode-route``:

* every capability registers as a Resource Catalog entry naming
  ``geo-geocode-route`` as provider, with the ``uvx`` install command for the
  not-installed case (Requirements 2.1, 2.6);
* the server's in-code :meth:`required_credentials` agrees exactly with
  ``bundle-manifest.json`` (Requirement 16.1);
* an unreachable / non-responding source surfaces as an ``Error_Taxonomy``
  availability error (``NETWORK``) identifying the source, with no partial data
  (Requirement 7.11); and
* an authentication failure surfaces as an ``Error_Taxonomy`` authentication
  error, with no partial data (Requirement 7.8).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import httpx
import pytest

from geo_common.errors import AuthenticationError, ErrorCategory, NetworkError
from geo_common.http import HttpClient
from geo_common.models import CredentialClassification, OpennessTier
from geo_common.retry import RetryPolicy

from geo_geocode_route.geocoding import NominatimSource
from geo_geocode_route.routing import OsrmSource
from geo_geocode_route.server import INSTALL_COMMAND, GeoGeocodeRouteServer

_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "bundle-manifest.json"

_OPEN_TIERS = {OpennessTier.OPEN, OpennessTier.FREE_TIER}


def _manifest_entry() -> Dict[str, Any]:
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    servers = {s["name"]: s for s in manifest["servers"]}
    assert "geo-geocode-route" in servers
    return servers["geo-geocode-route"]


async def _no_sleep(_seconds: float) -> None:
    return None


def _server(handler: Any) -> GeoGeocodeRouteServer:
    client = HttpClient(
        RetryPolicy(max_attempts=1),
        transport=httpx.MockTransport(handler),
        sleep=_no_sleep,
    )
    return GeoGeocodeRouteServer(
        http=client, geocoders=[NominatimSource()], routers=[OsrmSource()]
    )


# --- Req 2.1 / 2.6: catalog registration -----------------------------------


def test_catalog_entries_register_geo_geocode_route_as_open_provider() -> None:
    entries = GeoGeocodeRouteServer().catalog_entries()
    assert {e.name for e in entries} == {"geocode", "reverse_geocode", "route", "isochrone"}
    for entry in entries:
        assert entry.pillar == "A"
        assert entry.provider_server == "geo-geocode-route"
        assert entry.openness_tier in _OPEN_TIERS
        assert entry.install_command == INSTALL_COMMAND == "uvx geo-geocode-route"
        assert 1 <= len(entry.capability_description) <= 500


# --- Req 16.1: in-code credentials agree with the manifest ------------------


def test_required_credentials_match_manifest() -> None:
    specs = GeoGeocodeRouteServer().required_credentials()
    spec_pairs = {(s.mcp_json_key, s.classification.value) for s in specs}
    manifest_pairs = {
        (c["key"], c["classification"]) for c in _manifest_entry()["credentials"]
    }
    assert spec_pairs == manifest_pairs
    assert all(
        s.classification is CredentialClassification.OPTIONAL for s in specs
    )
    server = GeoGeocodeRouteServer()
    server.start(configured_keys=[])
    assert server.started is True


def test_manifest_install_command_matches_server() -> None:
    assert _manifest_entry()["uvx"] == INSTALL_COMMAND == "uvx geo-geocode-route"


# --- Req 7.11: unreachable source -> availability error, no partial data ----


async def test_unreachable_source_yields_availability_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    server = _server(handler)
    try:
        with pytest.raises(NetworkError) as exc_info:
            await server.geocode(address="Berlin")
    finally:
        await server.aclose()
    err = exc_info.value
    assert err.category is ErrorCategory.NETWORK
    # The availability error names the logical source, not the raw host.
    assert err.source == "nominatim"


async def test_route_unreachable_source_identifies_engine() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    server = _server(handler)
    try:
        with pytest.raises(NetworkError) as exc_info:
            await server.route(origin=(13.4, 52.5), destination=(13.1, 52.4))
    finally:
        await server.aclose()
    assert exc_info.value.category is ErrorCategory.NETWORK
    assert exc_info.value.source == "osrm"


# --- Req 7.8: authentication failure -> auth error, no partial data ---------


async def test_authentication_failure_yields_authentication_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "unauthorized"})

    server = _server(handler)
    try:
        with pytest.raises(AuthenticationError) as exc_info:
            await server.geocode(address="Berlin")
    finally:
        await server.aclose()
    assert exc_info.value.category is ErrorCategory.AUTHENTICATION

"""The ``geo-geocode-route`` MCP server (Pillar A, expansion).

Wires the geocoding (:mod:`geo_geocode_route.geocoding`) and routing
(:mod:`geo_geocode_route.routing`) connectors into the shared
:class:`~geo_common.server.BaseGeoServer` contract and exposes ``geocode``
(Requirement 7.4) and ``route`` (Requirement 7.10) as MCP tools. Both tools
validate their input and reject malformed parameters - an unparseable address,
a malformed coordinate, or an unsupported profile - with an ``Error_Taxonomy``
validation error (Requirement 7.12). It also registers the server's Resource
Catalog entries and credential specs (Requirements 2.1, 11.3).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from geo_common.http import HttpClient
from geo_common.models import (
    CatalogEntry,
    CredentialClassification,
    CredentialSpec,
    OpennessTier,
)
from geo_common.server import BaseGeoServer

from geo_geocode_route.geocoding import (
    AmazonLocationSource,
    GeocodeSource,
    default_geocoders,
    geocode as _geocode,
    reverse_geocode as _reverse_geocode,
)
from geo_geocode_route.isochrone import (
    IsochroneSource,
    default_isochrone_sources,
    isochrone as _isochrone,
)
from geo_geocode_route.models import Coordinate, Route
from geo_geocode_route.routing import (
    RouteSource,
    default_routers,
    route as _route,
)

__all__ = ["GeoGeocodeRouteServer", "INSTALL_COMMAND", "main"]

#: The ``uvx`` command that installs this server (Requirements 2.6, 16.1). The
#: Resource Catalog surfaces it on entries whose provider is not yet installed.
INSTALL_COMMAND = "uvx geo-geocode-route"


class GeoGeocodeRouteServer(BaseGeoServer):
    """Pillar A (expansion) server exposing geocoding and routing (Req 7.4, 7.10).

    Holds the configurable geocoder set (Nominatim + Photon by default) and
    routing-engine set (OSRM by default) and registers ``geocode`` and
    ``route`` as MCP tools. All outbound calls share the inherited
    :class:`HttpClient` so they get identical retry/backoff and the per-request
    timeout that enforces the 10-second geocode and 30-second route bounds.
    """

    pillar = "A"
    server_name = "geo-geocode-route"
    version = "0.2.0"

    #: The default geocoders/routers (Nominatim/Photon/OSRM/Valhalla) are open
    #: and need no credential. The Amazon Location provider *can* use an optional
    #: API key; this mirrors ``bundle-manifest.json``'s ``geo-geocode-route``
    #: credential block. The key is :attr:`CredentialClassification.OPTIONAL`,
    #: so an absent key never blocks startup (Requirement 16.5).
    _CREDENTIAL_SPECS = (
        CredentialSpec(
            source="Amazon Location Service",
            mcp_json_key="AMAZON_LOCATION_API_KEY",
            classification=CredentialClassification.OPTIONAL,
        ),
    )

    def __init__(
        self,
        *,
        geocoders: Optional[Sequence[GeocodeSource]] = None,
        routers: Optional[Sequence[RouteSource]] = None,
        isochrone_sources: Optional[Sequence[IsochroneSource]] = None,
        http: Optional[HttpClient] = None,
    ) -> None:
        super().__init__(http=http)
        self.geocoders: List[GeocodeSource] = (
            list(geocoders) if geocoders is not None else self._default_geocoders()
        )
        self.routers: List[RouteSource] = (
            list(routers) if routers is not None else default_routers()
        )
        self.isochrone_sources: List[IsochroneSource] = (
            list(isochrone_sources)
            if isochrone_sources is not None
            else default_isochrone_sources()
        )
        self.register_tool("geocode", self.geocode)
        self.register_tool("reverse_geocode", self.reverse_geocode)
        self.register_tool("route", self.route)
        self.register_tool("isochrone", self.isochrone)

    @staticmethod
    def _default_geocoders() -> List[GeocodeSource]:
        """Open geocoders, plus Amazon Location as a selectable provider.

        The open OpenStreetMap-backed geocoders (Nominatim, Photon) are the
        default chain. Amazon Location is always *selectable* via
        ``source="amazon-location"`` (reading ``AMAZON_LOCATION_API_KEY`` from
        the environment, which may be unset) but is excluded from the default
        chain (``default_chain=False``), so an open, no-credential geocode is
        never blocked by it. Selecting it without a key yields a clean
        ``AuthenticationError`` naming the key.
        """
        api_key = os.environ.get("AMAZON_LOCATION_API_KEY") or None
        geocoders: List[GeocodeSource] = list(default_geocoders())
        geocoders.append(AmazonLocationSource(api_key=api_key))
        return geocoders

    def catalog_entries(self) -> List[CatalogEntry]:
        """Register ``geo-geocode-route``'s capabilities in the Resource Catalog.

        Declares the ``geocode`` and ``route`` capabilities (Requirements 2.1,
        11.3). The default sources - Nominatim/Photon (OpenStreetMap-backed)
        and OSRM - are open data/services, so both entries are tier
        :attr:`OpennessTier.OPEN`, name ``geo-geocode-route`` as the provider,
        and carry the ``uvx`` install command (Requirement 2.6).
        """
        return [
            CatalogEntry(
                name="geocode",
                pillar=self.pillar,
                capability_description=(
                    "Geocode a free-form address to EPSG:4326 coordinates using "
                    "OpenStreetMap-backed geocoders (Nominatim, Photon)."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                install_command=INSTALL_COMMAND,
            ),
            CatalogEntry(
                name="reverse_geocode",
                pillar=self.pillar,
                capability_description=(
                    "Reverse-geocode an EPSG:4326 coordinate to its nearest "
                    "address/place using OpenStreetMap-backed geocoders "
                    "(Nominatim, Photon)."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                install_command=INSTALL_COMMAND,
            ),
            CatalogEntry(
                name="route",
                pillar=self.pillar,
                capability_description=(
                    "Compute a route (GeoJSON geometry, distance, and duration) "
                    "between an origin and destination coordinate using open "
                    "routing engines (OSRM, Valhalla)."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                install_command=INSTALL_COMMAND,
            ),
            CatalogEntry(
                name="isochrone",
                pillar=self.pillar,
                capability_description=(
                    "Compute reachability isochrones (GeoJSON polygons) from an "
                    "origin for one or more travel-time budgets using the open "
                    "Valhalla routing engine."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                install_command=INSTALL_COMMAND,
            ),
        ]

    def required_credentials(self) -> List[CredentialSpec]:
        """Declare the ``mcp.json`` keys this server can use (Req 16.1).

        The default geocoders and routing engine (Nominatim/Photon/OSRM) are
        open and require no credential; the optional Amazon Location provider
        can use ``AMAZON_LOCATION_API_KEY``. That key is
        :attr:`CredentialClassification.OPTIONAL`, so the startup credential
        guard always lets the server start (Requirement 16.5) and the
        declaration matches ``bundle-manifest.json``.
        """
        return list(self._CREDENTIAL_SPECS)

    async def geocode(self, *, address: str, source: Optional[str] = None) -> Coordinate:
        """Return the coordinate for ``address`` within 10s (Requirement 7.4).

        Validates the address before querying any source; an unparseable
        address raises an ``Error_Taxonomy`` ``ValidationError`` (Requirement
        7.12). ``source`` selects the geocoder: omit it to try the open default
        chain (Nominatim, then Photon) and return the first match, or name one
        (e.g. ``"amazon-location"``) to force that provider. A credentialed
        provider selected without its key raises an ``AuthenticationError``
        naming the ``mcp.json`` key.
        """
        return await _geocode(
            address=address, http=self.http, sources=self.geocoders, source=source
        )

    async def reverse_geocode(
        self, *, lon: float, lat: float, source: Optional[str] = None
    ) -> Coordinate:
        """Return the nearest address for ``(lon, lat)`` (reverse geocoding, Req 7.4).

        Validates the coordinate before querying any source; a malformed or
        out-of-range ``lon``/``lat`` raises an ``Error_Taxonomy``
        ``ValidationError`` (Requirement 7.12). ``source`` selects the geocoder
        the same way as :meth:`geocode` (omit for the open default chain; name
        one - e.g. ``"amazon-location"`` - to force it). The first resolved
        address is returned (in the result's ``label``).
        """
        return await _reverse_geocode(
            lon=lon, lat=lat, http=self.http, sources=self.geocoders, source=source
        )

    async def route(
        self,
        *,
        origin: Any,
        destination: Any,
        profile: str = "car",
    ) -> Route:
        """Return a route between ``origin`` and ``destination`` within 30s (Req 7.10).

        Validates both coordinates and the travel profile before querying any
        engine; a malformed coordinate or unsupported profile raises an
        ``Error_Taxonomy`` ``ValidationError`` (Requirement 7.12). On a valid
        request the configured routing engines are tried in order through the
        shared HTTP client and the first route found is returned.
        """
        return await _route(
            origin=origin,
            destination=destination,
            http=self.http,
            sources=self.routers,
            profile=profile,
        )

    async def isochrone(
        self,
        *,
        origin: Any,
        contours_minutes: Optional[Sequence[float]] = None,
        profile: str = "car",
    ) -> Dict[str, Any]:
        """Return reachability isochrone polygons from ``origin`` (GeoJSON).

        Validates the origin, profile, and ``contours_minutes`` (one or more
        positive time budgets in minutes) before querying any engine; malformed
        or missing input - including an omitted or empty ``contours_minutes`` -
        raises an ``Error_Taxonomy`` ``ValidationError`` naming the parameter
        (Requirement 7.12). On a valid request the configured isochrone engines
        (Valhalla by default) are tried in order through the shared HTTP client
        and the result is returned as a GeoJSON ``FeatureCollection`` of
        reachability polygons.
        """
        return await _isochrone(
            origin=origin,
            contours_minutes=contours_minutes,
            http=self.http,
            sources=self.isochrone_sources,
            profile=profile,
        )


def main() -> None:
    """Console entry point: serve geo-geocode-route over MCP stdio.

    Runs the startup credential guard, then serves this server's registered
    tools over stdin/stdout via the shared geo-common MCP runtime until the
    client disconnects.
    """
    GeoGeocodeRouteServer().run()

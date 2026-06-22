"""geo-geocode-route: Pillar A (expansion) geocoding + routing MCP server.

Exposes two tools (design.md "Pillar A — Data Connectors"):

* ``geocode`` — turn a free-form address into EPSG:4326 coordinates within 10
  seconds (Requirement 7.4), using OpenStreetMap-backed geocoders (Nominatim,
  Photon).
* ``route`` — compute a route (GeoJSON geometry, distance, duration) between an
  origin and destination coordinate within 30 seconds (Requirement 7.10),
  using open routing engines (OSRM, Valhalla).

Both tools reject malformed parameters - an unparseable address, a malformed
coordinate, or an unsupported travel profile - with an ``Error_Taxonomy``
validation error before any source is queried (Requirement 7.12).

Task 13.2 delivers ``geocode``/``route``, the source connectors, the data
models, the validation surface, and the server scaffold; task 13.6 registers
the Pillar A expansion catalog entries and availability/authentication error
handling alongside the other expansion connectors.
"""

from __future__ import annotations

from geo_geocode_route.geocoding import (
    AmazonLocationSource,
    GeocodeSource,
    NominatimSource,
    PhotonSource,
    default_geocoders,
    geocode,
    reverse_geocode,
)
from geo_geocode_route.isochrone import (
    IsochroneSource,
    ValhallaIsochroneSource,
    default_isochrone_sources,
    isochrone,
)
from geo_geocode_route.models import Coordinate, Route
from geo_geocode_route.routing import (
    OsrmSource,
    RouteSource,
    ValhallaSource,
    default_routers,
    route,
)
from geo_geocode_route.server import GeoGeocodeRouteServer
from geo_geocode_route.validate import (
    MAX_ADDRESS_LENGTH,
    SUPPORTED_PROFILES,
    normalize_profile,
    validate_address,
    validate_coordinate,
)

__all__ = [
    # Data models
    "Coordinate",
    "Route",
    # Validation
    "MAX_ADDRESS_LENGTH",
    "SUPPORTED_PROFILES",
    "validate_address",
    "validate_coordinate",
    "normalize_profile",
    # Geocoding
    "GeocodeSource",
    "NominatimSource",
    "PhotonSource",
    "AmazonLocationSource",
    "default_geocoders",
    "geocode",
    "reverse_geocode",
    # Routing
    "RouteSource",
    "OsrmSource",
    "ValhallaSource",
    "default_routers",
    "route",
    # Isochrone
    "IsochroneSource",
    "ValhallaIsochroneSource",
    "default_isochrone_sources",
    "isochrone",
    # Server
    "GeoGeocodeRouteServer",
]

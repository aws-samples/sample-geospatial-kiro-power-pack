"""Route computation for ``geo-geocode-route`` (Requirements 7.10, 7.12).

``route`` computes a route between an origin and a destination coordinate
within 30 seconds (Requirement 7.10). It validates both coordinates and the
travel profile first - malformed input is rejected with a ``ValidationError``
before any engine is queried (Requirement 7.12) - then tries the configured
routing engines in order and returns the first route found.

Two open routing engines are supported:

* :class:`OsrmSource` - the OSRM ``route`` service (default). Returns GeoJSON
  geometry directly.
* :class:`ValhallaSource` - the Valhalla ``route`` service. Returns an
  encoded-polyline shape that this module decodes to GeoJSON.

Every outbound call goes through the shared
:class:`~geo_common.http.HttpClient`, so all engines inherit identical
retry/backoff, rate-limit handling, and the per-request timeout that enforces
the 30-second response bound (Requirement 5). An engine that is unreachable
surfaces as a taxonomy-classified availability error re-tagged with the logical
engine name; when no engine can find a route a
:class:`~geo_common.errors.NotFoundError` is raised.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from geo_common.errors import (
    ErrorCategory,
    GeoError,
    NetworkError,
    NotFoundError,
    UpstreamError,
)
from geo_common.http import HttpClient

from geo_geocode_route.models import Coordinate, Route
from geo_geocode_route.validate import (
    SERVER_NAME,
    normalize_profile,
    validate_coordinate,
)

__all__ = [
    "RouteSource",
    "OsrmSource",
    "ValhallaSource",
    "default_routers",
    "route",
]

_AVAILABILITY_CATEGORIES = (ErrorCategory.NETWORK, ErrorCategory.UPSTREAM)

DEFAULT_OSRM_URL = "https://router.project-osrm.org/route/v1"
DEFAULT_VALHALLA_URL = "https://valhalla1.openstreetmap.de/route"

#: Canonical profile -> OSRM URL profile segment.
_OSRM_PROFILES = {"car": "driving", "bike": "cycling", "foot": "walking"}
#: Canonical profile -> Valhalla costing model.
_VALHALLA_COSTING = {"car": "auto", "bike": "bicycle", "foot": "pedestrian"}


class RouteSource:
    """Base class for a routing engine queried by :func:`route`.

    A source exposes a human-readable :attr:`name` and an async :meth:`fetch`
    that returns a :class:`Route` between two validated coordinates, or
    ``None`` when the engine finds no route. Subclasses make all I/O through
    the shared :class:`HttpClient` and raise a taxonomy-classified
    :class:`~geo_common.errors.GeoError` on failure.
    """

    name: str = "route-source"

    async def fetch(
        self,
        http: HttpClient,
        origin: Coordinate,
        destination: Coordinate,
        profile: str,
    ) -> Optional[Route]:  # pragma: no cover - abstract
        raise NotImplementedError


class OsrmSource(RouteSource):
    """Routing via the OSRM ``route`` service (GeoJSON geometry)."""

    name = "osrm"

    def __init__(self, url: str = DEFAULT_OSRM_URL) -> None:
        self.url = url.rstrip("/")

    async def fetch(
        self,
        http: HttpClient,
        origin: Coordinate,
        destination: Coordinate,
        profile: str,
    ) -> Optional[Route]:
        osrm_profile = _OSRM_PROFILES.get(profile, "driving")
        coords = "%g,%g;%g,%g" % (
            origin.lon,
            origin.lat,
            destination.lon,
            destination.lat,
        )
        url = "%s/%s/%s" % (self.url, osrm_profile, coords)
        try:
            response = await http.get(
                url, params={"overview": "full", "geometries": "geojson"}
            )
        except GeoError:
            raise
        if response.status_code >= 400:
            raise UpstreamError(
                "OSRM returned HTTP %d" % response.status_code,
                source=self.name,
                detail={"status_code": response.status_code},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(
                "OSRM returned a non-JSON response",
                source=self.name,
                original=str(exc),
            )
        return _parse_osrm(payload, profile=profile)


class ValhallaSource(RouteSource):
    """Routing via the Valhalla ``route`` service (encoded-polyline shape)."""

    name = "valhalla"

    def __init__(self, url: str = DEFAULT_VALHALLA_URL) -> None:
        self.url = url

    async def fetch(
        self,
        http: HttpClient,
        origin: Coordinate,
        destination: Coordinate,
        profile: str,
    ) -> Optional[Route]:
        costing = _VALHALLA_COSTING.get(profile, "auto")
        body = {
            "locations": [
                {"lon": origin.lon, "lat": origin.lat},
                {"lon": destination.lon, "lat": destination.lat},
            ],
            "costing": costing,
        }
        try:
            response = await http.post(self.url, json=body)
        except GeoError:
            raise
        if response.status_code >= 400:
            raise UpstreamError(
                "Valhalla returned HTTP %d" % response.status_code,
                source=self.name,
                detail={"status_code": response.status_code},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(
                "Valhalla returned a non-JSON response",
                source=self.name,
                original=str(exc),
            )
        return _parse_valhalla(payload, profile=profile)


def default_routers() -> List[RouteSource]:
    """The default routing-engine set: OSRM."""
    return [OsrmSource()]


async def route(
    *,
    origin: Any,
    destination: Any,
    http: HttpClient,
    sources: Sequence[RouteSource],
    profile: str = "car",
) -> Route:
    """Return a route between ``origin`` and ``destination`` within 30s (Req 7.10).

    Validates both coordinates and the travel ``profile`` first (Requirement
    7.12): a malformed coordinate or an unsupported profile raises a
    :class:`~geo_common.errors.ValidationError` before any engine is queried.
    The configured engines are then tried in order through the shared
    :class:`HttpClient` (inheriting its 30-second per-request timeout); the
    first engine that finds a route wins.

    An unavailable engine is recorded and the next engine tried; only when
    **every** engine is unavailable is a re-tagged availability error raised.
    When all engines respond but none can find a route, a
    :class:`~geo_common.errors.NotFoundError` is raised.
    """
    origin_c = validate_coordinate(origin, parameter="origin")
    destination_c = validate_coordinate(destination, parameter="destination")
    canonical_profile = normalize_profile(profile)

    last_unavailable: Optional[GeoError] = None
    for source in sources:
        try:
            found = await source.fetch(http, origin_c, destination_c, canonical_profile)
        except GeoError as exc:
            if exc.category in _AVAILABILITY_CATEGORIES:
                last_unavailable = _identify_source(exc, source_name=source.name)
                continue
            raise
        if found is not None:
            found.source = source.name
            found.origin = origin_c
            found.destination = destination_c
            return found

    if last_unavailable is not None:
        raise last_unavailable
    raise NotFoundError(
        "no routing engine could compute a route between the coordinates",
        source=SERVER_NAME,
        detail={"origin": origin_c.as_position(), "destination": destination_c.as_position()},
    )


def _identify_source(exc: GeoError, *, source_name: str) -> GeoError:
    """Re-tag an availability error with the logical engine name (Req 7.11)."""
    if exc.category not in _AVAILABILITY_CATEGORIES or exc.source == source_name:
        return exc
    cls = NetworkError if exc.category is ErrorCategory.NETWORK else UpstreamError
    return cls(
        exc.message,
        source=source_name,
        detail=exc.detail,
        original=exc.original,
        retry_after=exc.retry_after,
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_osrm(payload: Any, *, profile: str) -> Optional[Route]:
    """Parse an OSRM ``route`` response into a :class:`Route`.

    OSRM returns a ``code`` plus a ``routes`` array; each route carries a
    GeoJSON ``geometry``, a ``distance`` (metres) and a ``duration`` (seconds).
    ``code == "NoRoute"`` (or an empty ``routes`` array) means no route was
    found (``None``); any other non-``"Ok"`` code is an upstream error.
    """
    if not isinstance(payload, dict):
        raise UpstreamError("OSRM response was not a JSON object", source="osrm")
    code = payload.get("code")
    if code == "NoRoute":
        return None
    if code not in (None, "Ok"):
        raise UpstreamError(
            "OSRM returned code %r" % code,
            source="osrm",
            detail={"code": code},
        )
    routes = payload.get("routes")
    if not isinstance(routes, list) or not routes:
        return None
    first = routes[0]
    if not isinstance(first, dict):
        raise UpstreamError("OSRM route was not a JSON object", source="osrm")
    geometry = first.get("geometry")
    if not isinstance(geometry, dict):
        raise UpstreamError("OSRM route had no GeoJSON geometry", source="osrm")
    return Route(
        distance_m=_as_float(first.get("distance"), source="osrm", field="distance"),
        duration_s=_as_float(first.get("duration"), source="osrm", field="duration"),
        geometry=geometry,
        profile=profile,
        source="osrm",
    )


def _parse_valhalla(payload: Any, *, profile: str) -> Optional[Route]:
    """Parse a Valhalla ``route`` response into a :class:`Route`.

    Valhalla returns a ``trip`` with a ``summary`` (``length`` in km,
    ``time`` in seconds) and ``legs`` whose ``shape`` is an encoded polyline
    (precision 6). The shape is decoded to a GeoJSON ``LineString``. A trip
    ``status`` other than 0 means no route (``None``).
    """
    if not isinstance(payload, dict):
        raise UpstreamError("Valhalla response was not a JSON object", source="valhalla")
    trip = payload.get("trip")
    if not isinstance(trip, dict):
        raise UpstreamError("Valhalla response had no trip", source="valhalla")
    status = trip.get("status")
    if status not in (None, 0):
        return None
    legs = trip.get("legs")
    if not isinstance(legs, list) or not legs:
        return None

    coordinates: List[List[float]] = []
    for leg in legs:
        if isinstance(leg, dict) and isinstance(leg.get("shape"), str):
            coordinates.extend(_decode_polyline(leg["shape"], precision=6))
    if not coordinates:
        return None

    summary = trip.get("summary") or {}
    length_km = _as_float(summary.get("length"), source="valhalla", field="length")
    duration_s = _as_float(summary.get("time"), source="valhalla", field="time")
    return Route(
        distance_m=length_km * 1000.0,
        duration_s=duration_s,
        geometry={"type": "LineString", "coordinates": coordinates},
        profile=profile,
        source="valhalla",
    )


def _decode_polyline(encoded: str, *, precision: int = 6) -> List[List[float]]:
    """Decode an encoded polyline into ``[lon, lat]`` positions.

    Implements the standard Google/Valhalla polyline algorithm. Valhalla uses
    precision 6 (1e-6 degree resolution). Coordinates are emitted in GeoJSON
    ``[lon, lat]`` order.
    """
    factor = float(10 ** precision)
    coordinates: List[List[float]] = []
    index = 0
    lat = 0
    lon = 0
    length = len(encoded)
    while index < length:
        for is_longitude in (False, True):
            shift = 0
            result = 0
            while True:
                if index >= length:
                    return coordinates
                byte = ord(encoded[index]) - 63
                index += 1
                result |= (byte & 0x1F) << shift
                shift += 5
                if byte < 0x20:
                    break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            if is_longitude:
                lon += delta
            else:
                lat += delta
        coordinates.append([lon / factor, lat / factor])
    return coordinates


def _as_float(value: Any, *, source: str, field: str) -> float:
    """Coerce an upstream numeric field to float, mapping bad data to UPSTREAM."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise UpstreamError(
            "%s returned a non-numeric %s" % (source, field),
            source=source,
            original=str(exc),
        )

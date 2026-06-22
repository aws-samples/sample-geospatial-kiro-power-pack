"""Isochrone (reachability) computation for ``geo-geocode-route`` (Req 7.10-class).

``isochrone`` computes, for an origin and a travel profile, the area reachable
within one or more time budgets - returned as GeoJSON polygons. Only Valhalla
among the supported engines offers isochrones, so this ships a single
:class:`ValhallaIsochroneSource` against the public Valhalla ``/isochrone``
endpoint; the source set is configurable for mirrors/tests.

Inputs are validated before any engine call (Requirement 7.12): a malformed
origin, an unsupported profile, or an empty/invalid ``contours_minutes`` list
raises a :class:`~geo_common.errors.ValidationError`. Every call goes through
the shared :class:`~geo_common.http.HttpClient`, inheriting retry/backoff and
the per-request timeout; an unavailable engine surfaces as a re-tagged
availability error.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from geo_common.errors import (
    ErrorCategory,
    GeoError,
    NetworkError,
    UpstreamError,
    ValidationError,
)
from geo_common.http import HttpClient

from geo_geocode_route.validate import (
    SERVER_NAME,
    normalize_profile,
    validate_coordinate,
)

__all__ = [
    "IsochroneSource",
    "ValhallaIsochroneSource",
    "default_isochrone_sources",
    "isochrone",
    "DEFAULT_VALHALLA_ISOCHRONE_URL",
    "MAX_CONTOURS",
    "MAX_CONTOUR_MINUTES",
]

_AVAILABILITY_CATEGORIES = (ErrorCategory.NETWORK, ErrorCategory.UPSTREAM)

DEFAULT_VALHALLA_ISOCHRONE_URL = "https://valhalla1.openstreetmap.de/isochrone"

#: Canonical profile -> Valhalla costing model.
_VALHALLA_COSTING = {"car": "auto", "bike": "bicycle", "foot": "pedestrian"}

#: Bounds keeping a single isochrone request (and its response) reasonable.
MAX_CONTOURS = 6
MAX_CONTOUR_MINUTES = 120.0


class IsochroneSource:
    """Base class for an isochrone engine queried by :func:`isochrone`."""

    name: str = "isochrone-source"

    async def fetch(
        self,
        http: HttpClient,
        lon: float,
        lat: float,
        profile: str,
        contours_minutes: Sequence[float],
    ) -> Optional[Dict[str, Any]]:  # pragma: no cover - abstract
        raise NotImplementedError


class ValhallaIsochroneSource(IsochroneSource):
    """Isochrones via the Valhalla ``/isochrone`` service (GeoJSON polygons)."""

    name = "valhalla"

    def __init__(self, url: str = DEFAULT_VALHALLA_ISOCHRONE_URL) -> None:
        self.url = url

    async def fetch(
        self,
        http: HttpClient,
        lon: float,
        lat: float,
        profile: str,
        contours_minutes: Sequence[float],
    ) -> Optional[Dict[str, Any]]:
        body = {
            "locations": [{"lon": lon, "lat": lat}],
            "costing": _VALHALLA_COSTING.get(profile, "auto"),
            "contours": [{"time": float(m)} for m in contours_minutes],
            "polygons": True,
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
        return _parse_isochrone(payload)


def default_isochrone_sources() -> List[IsochroneSource]:
    """The default isochrone engine set: Valhalla."""
    return [ValhallaIsochroneSource()]


def _validate_contours(contours_minutes: Any) -> List[float]:
    """Validate the time budgets (minutes) for an isochrone request (Req 7.12)."""
    if contours_minutes is None:
        raise ValidationError(
            "contours_minutes is required: provide at least one positive time "
            "budget in minutes (e.g. [5, 10])",
            source=SERVER_NAME,
            detail={"parameter": "contours_minutes"},
        )
    if isinstance(contours_minutes, (str, bytes)) or not isinstance(
        contours_minutes, Sequence
    ):
        raise ValidationError(
            "contours_minutes must be a list of positive minute values",
            source=SERVER_NAME,
            detail={"parameter": "contours_minutes"},
        )
    values = list(contours_minutes)
    if not values:
        raise ValidationError(
            "contours_minutes must contain at least one positive time budget "
            "in minutes (e.g. [5, 10])",
            source=SERVER_NAME,
            detail={"parameter": "contours_minutes"},
        )
    if len(values) > MAX_CONTOURS:
        raise ValidationError(
            "contours_minutes may contain at most %d values" % MAX_CONTOURS,
            source=SERVER_NAME,
            detail={"parameter": "contours_minutes", "max": MAX_CONTOURS},
        )
    out: List[float] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValidationError(
                "each contour must be a positive number of minutes",
                source=SERVER_NAME,
                detail={"parameter": "contours_minutes", "value": v},
            )
        fv = float(v)
        if not (0.0 < fv <= MAX_CONTOUR_MINUTES):
            raise ValidationError(
                "each contour must be within (0, %g] minutes" % MAX_CONTOUR_MINUTES,
                source=SERVER_NAME,
                detail={"parameter": "contours_minutes", "value": fv},
            )
        out.append(fv)
    return out


async def isochrone(
    *,
    origin: Any,
    contours_minutes: Optional[Sequence[float]] = None,
    http: HttpClient,
    sources: Sequence[IsochroneSource],
    profile: str = "car",
) -> Dict[str, Any]:
    """Return reachability polygons from ``origin`` for each time budget.

    Validates ``origin``, ``profile``, and ``contours_minutes`` first
    (Requirement 7.12), then queries the configured isochrone engines in order
    through the shared :class:`HttpClient`. Returns a GeoJSON
    ``FeatureCollection`` whose features are the reachability polygons (each
    feature's ``properties.contour`` is its time budget in minutes). An
    unavailable engine is recorded and the next tried; only when every engine
    is unavailable is a re-tagged availability error raised.
    """
    origin_c = validate_coordinate(origin, parameter="origin")
    canonical_profile = normalize_profile(profile)
    contours = _validate_contours(contours_minutes)

    last_unavailable: Optional[GeoError] = None
    for source in sources:
        try:
            result = await source.fetch(
                http, origin_c.lon, origin_c.lat, canonical_profile, contours
            )
        except GeoError as exc:
            if exc.category in _AVAILABILITY_CATEGORIES:
                last_unavailable = _identify_source(exc, source_name=source.name)
                continue
            raise
        if result is not None:
            return result

    if last_unavailable is not None:
        raise last_unavailable
    raise UpstreamError(
        "no isochrone engine returned a result",
        source=SERVER_NAME,
        detail={"origin": origin_c.as_position()},
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


def _parse_isochrone(payload: Any) -> Optional[Dict[str, Any]]:
    """Validate a Valhalla isochrone GeoJSON response and return it.

    Valhalla returns a GeoJSON ``FeatureCollection`` whose polygon features each
    carry a ``contour`` (minutes) in their ``properties``. The payload is passed
    through after a light structural check; ``None`` (no features) signals the
    engine produced nothing.
    """
    if not isinstance(payload, dict):
        raise UpstreamError(
            "Valhalla isochrone response was not a JSON object", source="valhalla"
        )
    features = payload.get("features")
    if not isinstance(features, list):
        raise UpstreamError(
            "Valhalla isochrone response had no 'features' list", source="valhalla"
        )
    if not features:
        return None
    # Normalize to a clean FeatureCollection envelope.
    return {
        "type": payload.get("type", "FeatureCollection"),
        "features": features,
    }

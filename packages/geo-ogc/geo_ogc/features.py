"""OGC API - Features retrieval for ``geo-ogc`` (Pillar A).

``ogc_features`` queries an **OGC API - Features** service for the features in a
spatial extent and returns them as a :class:`FeatureCollectionResult`. OGC API -
Features is a standard (the modern successor to WFS), so this connector works
against any conformant service - pygeoapi, GeoServer's OGC API, ldproxy, etc. -
with the service base URL supplied per call (no credential; open standard).

The items request follows the standard shape::

    GET {endpoint}/collections/{collection}/items?bbox=minx,miny,maxx,maxy&limit=N

returning a GeoJSON ``FeatureCollection``. Inputs are validated before any
network call (a malformed bbox/limit/endpoint/collection raises a
:class:`~geo_common.errors.ValidationError`), and every call goes through the
shared :class:`~geo_common.http.HttpClient`, inheriting retry/backoff and the
per-request timeout; an unreachable/erroring service surfaces as a
taxonomy-classified availability error.
"""

from __future__ import annotations

from numbers import Real
from typing import Any, Dict, List, Optional, Sequence, Tuple

from geo_common.errors import GeoError, UpstreamError, ValidationError
from geo_common.http import HttpClient

from geo_ogc.models import FeatureCollectionResult

__all__ = ["ogc_features", "MAX_FEATURES", "DEFAULT_FEATURES"]

#: Hard cap on the number of features a single request may return, bounding the
#: response size; a larger ``limit`` is clamped to this.
MAX_FEATURES = 1000

#: Default number of features requested when the caller does not specify.
DEFAULT_FEATURES = 100

#: Secret-free server identifier used on errors raised from this module.
_SERVER_NAME = "geo-ogc"


async def ogc_features(
    *,
    endpoint: str,
    collection: str,
    bbox: Sequence[float],
    http: HttpClient,
    limit: int = DEFAULT_FEATURES,
    datetime_range: Optional[str] = None,
) -> FeatureCollectionResult:
    """Return features from an OGC API - Features ``collection`` within ``bbox``.

    Validates every parameter before any network call (Requirement 7.12-style):
    a blank ``endpoint``/``collection``, a malformed bbox or out-of-range
    coordinate, or a non-positive ``limit`` raises a
    :class:`~geo_common.errors.ValidationError`. A ``limit`` above
    :data:`MAX_FEATURES` is clamped down. The service is queried through the
    shared :class:`HttpClient`; a bad HTTP status or non-JSON body surfaces as
    an :class:`~geo_common.errors.UpstreamError`, and a transport failure
    propagates as a taxonomy-classified :class:`~geo_common.errors.GeoError`.
    """
    base = _validate_endpoint(endpoint)
    coll = _validate_collection(collection)
    west, south, east, north = _validate_bbox(bbox)
    effective_limit = _validate_limit(limit)

    params: Dict[str, Any] = {
        "bbox": "%g,%g,%g,%g" % (west, south, east, north),
        "limit": effective_limit,
        "f": "json",
    }
    if datetime_range is not None:
        if not isinstance(datetime_range, str) or not datetime_range.strip():
            raise ValidationError(
                "datetime must be a non-empty ISO-8601 instant or interval",
                source=_SERVER_NAME,
                detail={"parameter": "datetime"},
            )
        params["datetime"] = datetime_range.strip()

    items_url = "%s/collections/%s/items" % (base, coll)
    try:
        response = await http.get(items_url, params=params)
    except GeoError:
        # Already taxonomy-classified (timeout/network/rate-limit/etc.).
        raise
    if response.status_code == 404:
        raise UpstreamError(
            "OGC API collection %r not found at %s (HTTP 404)" % (coll, base),
            source=_SERVER_NAME,
            detail={"status_code": 404, "collection": coll},
        )
    if response.status_code >= 400:
        raise UpstreamError(
            "OGC API service returned HTTP %d" % response.status_code,
            source=_SERVER_NAME,
            detail={"status_code": response.status_code},
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise UpstreamError(
            "OGC API service returned a non-JSON response",
            source=_SERVER_NAME,
            original=str(exc),
        )
    return _parse_feature_collection(payload, endpoint=base, collection=coll)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_endpoint(endpoint: Any) -> str:
    """Validate and normalize the OGC API base URL (trailing slash stripped)."""
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValidationError(
            "endpoint must be a non-empty OGC API base URL",
            source=_SERVER_NAME,
            detail={"parameter": "endpoint"},
        )
    base = endpoint.strip().rstrip("/")
    if not (base.startswith("http://") or base.startswith("https://")):
        raise ValidationError(
            "endpoint must be an http(s) URL",
            source=_SERVER_NAME,
            detail={"parameter": "endpoint"},
        )
    return base


def _validate_collection(collection: Any) -> str:
    """Validate the collection id."""
    if not isinstance(collection, str) or not collection.strip():
        raise ValidationError(
            "collection must be a non-empty OGC API collection id",
            source=_SERVER_NAME,
            detail={"parameter": "collection"},
        )
    return collection.strip()


def _validate_bbox(bbox: Any) -> Tuple[float, float, float, float]:
    """Validate a ``(west, south, east, north)`` EPSG:4326 bbox (Req 7.12)."""
    if isinstance(bbox, (str, bytes)) or not isinstance(bbox, Sequence):
        raise ValidationError(
            "bbox must be a sequence of four numbers (west, south, east, north)",
            source=_SERVER_NAME,
            detail={"parameter": "bbox"},
        )
    values = list(bbox)
    if len(values) != 4:
        raise ValidationError(
            "bbox must contain exactly four numbers (west, south, east, north), "
            "got %d" % len(values),
            source=_SERVER_NAME,
            detail={"parameter": "bbox"},
        )
    nums: List[float] = []
    for label, value in zip(("west", "south", "east", "north"), values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValidationError(
                "bbox %s must be a number, got %r" % (label, value),
                source=_SERVER_NAME,
                detail={"parameter": "bbox"},
            )
        nums.append(float(value))
    west, south, east, north = nums
    if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
        raise ValidationError(
            "bbox longitudes must be within [-180, 180]",
            source=_SERVER_NAME,
            detail={"parameter": "bbox"},
        )
    if not (-90.0 <= south <= 90.0 and -90.0 <= north <= 90.0):
        raise ValidationError(
            "bbox latitudes must be within [-90, 90]",
            source=_SERVER_NAME,
            detail={"parameter": "bbox"},
        )
    if west > east:
        raise ValidationError(
            "bbox west (%g) must not exceed east (%g)" % (west, east),
            source=_SERVER_NAME,
            detail={"parameter": "bbox"},
        )
    if south > north:
        raise ValidationError(
            "bbox south (%g) must not exceed north (%g)" % (south, north),
            source=_SERVER_NAME,
            detail={"parameter": "bbox"},
        )
    return west, south, east, north


def _validate_limit(limit: Any) -> int:
    """Validate ``limit`` as a positive integer, clamped to :data:`MAX_FEATURES`."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValidationError(
            "limit must be a positive integer",
            source=_SERVER_NAME,
            detail={"parameter": "limit"},
        )
    if limit < 1:
        raise ValidationError(
            "limit must be a positive integer, got %d" % limit,
            source=_SERVER_NAME,
            detail={"parameter": "limit"},
        )
    return min(limit, MAX_FEATURES)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_feature_collection(
    payload: Any, *, endpoint: str, collection: str
) -> FeatureCollectionResult:
    """Convert an OGC API - Features GeoJSON payload into a result."""
    if not isinstance(payload, dict):
        raise UpstreamError(
            "OGC API response was not a JSON object", source=_SERVER_NAME
        )
    features = payload.get("features")
    if not isinstance(features, list):
        raise UpstreamError(
            "OGC API response had no 'features' list", source=_SERVER_NAME
        )
    clean = [f for f in features if isinstance(f, dict)]
    number_returned = payload.get("numberReturned")
    if not isinstance(number_returned, int):
        number_returned = len(clean)
    number_matched = payload.get("numberMatched")
    if not isinstance(number_matched, int):
        number_matched = None
    return FeatureCollectionResult(
        features=clean,
        number_returned=number_returned,
        number_matched=number_matched,
        endpoint=endpoint,
        collection=collection,
    )

"""Bounding-box validation and area computation for ``geo-vector``.

``vector_features`` accepts a spatial extent as a bounding box and must:

* reject a malformed bbox - wrong arity, non-numeric, out-of-range longitude or
  latitude, or an inverted (min greater than max) extent - with an
  ``Error_Taxonomy`` ``ValidationError`` naming the offending parameter
  (Requirement 7.12); and
* reject a bbox whose area exceeds the configured maximum (default 2,500 km²)
  with the same validation error (Requirements 7.3, 7.12).

The area of a longitude/latitude bounding box is the area of the spherical
quadrangle it bounds - the region between two meridians and two parallels on a
sphere of Earth's mean radius. That area is exactly

    R² · |Δλ| · |sin φ₂ − sin φ₁|

with Δλ the longitude span in radians and φ₁, φ₂ the latitudes in radians. This
is a closed-form, deterministic value (no projection or external dependency),
which keeps the area gate easy to reason about and test.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Sequence, Tuple

from geo_common.errors import ValidationError

from geo_vector.models import BBox

__all__ = [
    "EARTH_RADIUS_KM",
    "DEFAULT_MAX_AREA_KM2",
    "validate_bbox",
    "bbox_area_km2",
]

#: Earth's mean radius in kilometres (IUGG mean radius R₁), used for the
#: spherical-quadrangle area computation.
EARTH_RADIUS_KM = 6371.0088

#: The default maximum bounding-box area accepted by ``vector_features``
#: (Requirement 7.3). Configurable per call / per server.
DEFAULT_MAX_AREA_KM2 = 2500.0

_SERVER_NAME = "geo-vector"


def validate_bbox(bbox: Sequence[float]) -> BBox:
    """Validate ``bbox`` and return it as a 4-tuple of floats.

    ``bbox`` must be ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326
    decimal degrees with ``-180 <= lon <= 180`` and ``-90 <= lat <= 90`` and
    ``min <= max`` on each axis. On any violation a
    :class:`~geo_common.errors.ValidationError` (taxonomy ``validation``) is
    raised, naming ``bbox`` as the invalid parameter (Requirement 7.12).
    """
    # Arity: exactly four ordinates. A string is a Sequence too, so reject it
    # explicitly to avoid iterating its characters.
    if isinstance(bbox, (str, bytes)) or not isinstance(bbox, Sequence):
        raise ValidationError(
            "bbox must be a sequence of four numbers "
            "(min_lon, min_lat, max_lon, max_lat)",
            source=_SERVER_NAME,
            detail={"parameter": "bbox"},
        )
    values = list(bbox)
    if len(values) != 4:
        raise ValidationError(
            "bbox must contain exactly four numbers "
            "(min_lon, min_lat, max_lon, max_lat), got %d" % len(values),
            source=_SERVER_NAME,
            detail={"parameter": "bbox"},
        )

    # Numeric: bool is an int subclass; reject it as a non-coordinate value.
    coords = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValidationError(
                "bbox ordinates must be numbers, got %r" % (value,),
                source=_SERVER_NAME,
                detail={"parameter": "bbox"},
            )
        coords.append(float(value))

    min_lon, min_lat, max_lon, max_lat = coords

    if not (math.isfinite(min_lon) and math.isfinite(max_lon)):
        raise ValidationError(
            "bbox longitudes must be finite",
            source=_SERVER_NAME,
            detail={"parameter": "bbox"},
        )
    if not (math.isfinite(min_lat) and math.isfinite(max_lat)):
        raise ValidationError(
            "bbox latitudes must be finite",
            source=_SERVER_NAME,
            detail={"parameter": "bbox"},
        )
    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
        raise ValidationError(
            "bbox longitudes must be within [-180, 180]",
            source=_SERVER_NAME,
            detail={"parameter": "bbox"},
        )
    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        raise ValidationError(
            "bbox latitudes must be within [-90, 90]",
            source=_SERVER_NAME,
            detail={"parameter": "bbox"},
        )
    if min_lon > max_lon:
        raise ValidationError(
            "bbox min_lon (%g) must not exceed max_lon (%g)" % (min_lon, max_lon),
            source=_SERVER_NAME,
            detail={"parameter": "bbox"},
        )
    if min_lat > max_lat:
        raise ValidationError(
            "bbox min_lat (%g) must not exceed max_lat (%g)" % (min_lat, max_lat),
            source=_SERVER_NAME,
            detail={"parameter": "bbox"},
        )

    return (min_lon, min_lat, max_lon, max_lat)


def bbox_area_km2(bbox: BBox) -> float:
    """Return the area in km² of the spherical quadrangle bounded by ``bbox``.

    ``bbox`` is assumed already validated by :func:`validate_bbox`. The area is
    ``R² · |Δλ| · |sin φ₂ − sin φ₁|`` with angles in radians - exact for a
    sphere of radius :data:`EARTH_RADIUS_KM`.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_span = math.radians(max_lon - min_lon)
    lat_term = math.sin(math.radians(max_lat)) - math.sin(math.radians(min_lat))
    return (EARTH_RADIUS_KM ** 2) * abs(lon_span) * abs(lat_term)

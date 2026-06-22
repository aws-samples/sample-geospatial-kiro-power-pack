"""Parameter validation for ``geo-biodiversity`` (Requirement 7.12).

``species_occurrences`` must reject malformed parameters *before* any network
call and raise an ``Error_Taxonomy`` :class:`~geo_common.errors.ValidationError`
naming the offending parameter:

* a malformed bounding box - wrong arity, non-numeric, non-finite, out-of-range
  longitude/latitude, or an inverted (min greater than max) extent;
* a non-positive or non-integer ``limit``; and
* a ``taxon`` filter that is present but not a non-blank string.

The record cap is also applied here: a requested ``limit`` larger than the
configured maximum (:data:`DEFAULT_MAX_RECORDS`, 10,000) is clamped down to it
so a response never exceeds the cap (Requirement 7.7).
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Optional, Sequence

from geo_common.errors import ValidationError

from geo_biodiversity.models import BBox

__all__ = [
    "DEFAULT_MAX_RECORDS",
    "validate_bbox",
    "validate_and_cap_limit",
    "validate_taxon",
]

#: The default maximum number of occurrence records returned per response
#: (Requirement 7.7). A requested ``limit`` larger than this is clamped to it.
DEFAULT_MAX_RECORDS = 10000

_SOURCE_NAME = "geo-biodiversity"


def validate_bbox(bbox: Sequence[float]) -> BBox:
    """Validate ``bbox`` and return it as a 4-tuple of floats (Requirement 7.12).

    ``bbox`` must be ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326
    decimal degrees with ``-180 <= lon <= 180`` and ``-90 <= lat <= 90`` and
    ``min <= max`` on each axis. On any violation a
    :class:`~geo_common.errors.ValidationError` (taxonomy ``validation``) is
    raised, naming ``bbox`` as the invalid parameter.
    """
    # Arity: exactly four ordinates. A string is a Sequence too, so reject it
    # explicitly to avoid iterating its characters.
    if isinstance(bbox, (str, bytes)) or not isinstance(bbox, Sequence):
        raise ValidationError(
            "bbox must be a sequence of four numbers "
            "(min_lon, min_lat, max_lon, max_lat)",
            source=_SOURCE_NAME,
            detail={"parameter": "bbox"},
        )
    values = list(bbox)
    if len(values) != 4:
        raise ValidationError(
            "bbox must contain exactly four numbers "
            "(min_lon, min_lat, max_lon, max_lat), got %d" % len(values),
            source=_SOURCE_NAME,
            detail={"parameter": "bbox"},
        )

    # Numeric: bool is an int subclass; reject it as a non-coordinate value.
    coords = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValidationError(
                "bbox ordinates must be numbers, got %r" % (value,),
                source=_SOURCE_NAME,
                detail={"parameter": "bbox"},
            )
        coords.append(float(value))

    min_lon, min_lat, max_lon, max_lat = coords

    if not (math.isfinite(min_lon) and math.isfinite(max_lon)):
        raise ValidationError(
            "bbox longitudes must be finite",
            source=_SOURCE_NAME,
            detail={"parameter": "bbox"},
        )
    if not (math.isfinite(min_lat) and math.isfinite(max_lat)):
        raise ValidationError(
            "bbox latitudes must be finite",
            source=_SOURCE_NAME,
            detail={"parameter": "bbox"},
        )
    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
        raise ValidationError(
            "bbox longitudes must be within [-180, 180]",
            source=_SOURCE_NAME,
            detail={"parameter": "bbox"},
        )
    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        raise ValidationError(
            "bbox latitudes must be within [-90, 90]",
            source=_SOURCE_NAME,
            detail={"parameter": "bbox"},
        )
    if min_lon > max_lon:
        raise ValidationError(
            "bbox min_lon (%g) must not exceed max_lon (%g)" % (min_lon, max_lon),
            source=_SOURCE_NAME,
            detail={"parameter": "bbox"},
        )
    if min_lat > max_lat:
        raise ValidationError(
            "bbox min_lat (%g) must not exceed max_lat (%g)" % (min_lat, max_lat),
            source=_SOURCE_NAME,
            detail={"parameter": "bbox"},
        )

    return (min_lon, min_lat, max_lon, max_lat)


def validate_and_cap_limit(limit: int) -> int:
    """Validate ``limit`` and clamp it to :data:`DEFAULT_MAX_RECORDS`.

    A non-integer or non-positive ``limit`` raises a
    :class:`~geo_common.errors.ValidationError` naming ``limit`` (Requirement
    7.12). A valid ``limit`` larger than the configured maximum is clamped down
    to it so a response is capped at the maximum (Requirement 7.7).
    """
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValidationError(
            "limit must be an integer",
            source=_SOURCE_NAME,
            detail={"parameter": "limit"},
        )
    if limit < 1:
        raise ValidationError(
            "limit must be a positive integer",
            source=_SOURCE_NAME,
            detail={"parameter": "limit"},
        )
    return min(limit, DEFAULT_MAX_RECORDS)


def validate_taxon(taxon: Optional[str]) -> Optional[str]:
    """Validate the optional ``taxon`` filter (Requirement 7.12).

    ``None`` (no filter) is accepted. When present, ``taxon`` must be a
    non-blank string; anything else raises a
    :class:`~geo_common.errors.ValidationError` naming ``taxon``. The returned
    value is stripped of surrounding whitespace.
    """
    if taxon is None:
        return None
    if not isinstance(taxon, str) or not taxon.strip():
        raise ValidationError(
            "taxon, when provided, must be a non-empty string",
            source=_SOURCE_NAME,
            detail={"parameter": "taxon"},
        )
    return taxon.strip()

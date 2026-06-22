"""Location and time-range validation for ``geo-weather-climate``.

``observations`` must validate every user-supplied parameter *before* any
network call and reject a malformed request with an ``Error_Taxonomy``
``ValidationError`` naming the offending parameter (Requirement 7.12). The two
gates this module provides are:

* :func:`validate_location` - a point ``(lon, lat)`` in EPSG:4326 decimal
  degrees, longitude in ``[-180, 180]`` and latitude in ``[-90, 90]``; and
* :func:`validate_time_range` - a ``(start, end)`` pair of ISO-8601 timestamps
  whose ``start`` is not later than its ``end`` (the start-after-end rejection
  called out by task 13.4).

Both raise before any source is contacted so no partial work is performed on an
invalid request.
"""

from __future__ import annotations

import math
from datetime import datetime
from numbers import Real
from typing import Any, Sequence, Tuple

from geo_common.errors import ValidationError

from geo_weather_climate.models import Coordinate

__all__ = ["validate_location", "validate_time_range"]

#: The secret-free source identifier used on errors raised from this module.
_SERVER_NAME = "geo-weather-climate"


def validate_location(location: Any) -> Coordinate:
    """Validate a point location and return it as a :class:`Coordinate`.

    Accepts an existing :class:`Coordinate`, a mapping with ``lon``/``lat``
    keys, or a two-element ``(lon, lat)`` sequence. Longitudes must lie in
    ``[-180, 180]`` and latitudes in ``[-90, 90]`` and both must be finite. On
    any violation a :class:`~geo_common.errors.ValidationError` (taxonomy
    ``validation``) is raised naming ``location`` (Requirement 7.12).
    """
    if isinstance(location, Coordinate):
        lon, lat = location.lon, location.lat
    elif isinstance(location, dict):
        if "lon" not in location or "lat" not in location:
            raise ValidationError(
                "location mapping must provide 'lon' and 'lat'",
                source=_SERVER_NAME,
                detail={"parameter": "location"},
            )
        lon, lat = location["lon"], location["lat"]
    elif isinstance(location, (str, bytes)) or not isinstance(location, Sequence):
        raise ValidationError(
            "location must be a (lon, lat) pair or a {'lon', 'lat'} mapping",
            source=_SERVER_NAME,
            detail={"parameter": "location"},
        )
    else:
        values = list(location)
        if len(values) != 2:
            raise ValidationError(
                "location must contain exactly two numbers (lon, lat), got %d"
                % len(values),
                source=_SERVER_NAME,
                detail={"parameter": "location"},
            )
        lon, lat = values

    coords = []
    for name, value in (("lon", lon), ("lat", lat)):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValidationError(
                "location %s must be a number, got %r" % (name, value),
                source=_SERVER_NAME,
                detail={"parameter": "location"},
            )
        fvalue = float(value)
        if not math.isfinite(fvalue):
            raise ValidationError(
                "location %s must be finite" % name,
                source=_SERVER_NAME,
                detail={"parameter": "location"},
            )
        coords.append(fvalue)

    lon_f, lat_f = coords
    if not (-180.0 <= lon_f <= 180.0):
        raise ValidationError(
            "location longitude must be within [-180, 180]",
            source=_SERVER_NAME,
            detail={"parameter": "location"},
        )
    if not (-90.0 <= lat_f <= 90.0):
        raise ValidationError(
            "location latitude must be within [-90, 90]",
            source=_SERVER_NAME,
            detail={"parameter": "location"},
        )
    return Coordinate(lon=lon_f, lat=lat_f)


def validate_time_range(time_range: Sequence[str]) -> Tuple[str, str]:
    """Validate a ``(start, end)`` ISO-8601 time range (Requirement 7.12).

    Raises a :class:`~geo_common.errors.ValidationError` naming ``time_range``
    for a malformed pair, an unparseable timestamp, or - the case task 13.4
    calls out explicitly - a ``start`` that is later than its ``end``. Returns
    the original ``(start, end)`` strings on success so the caller forwards the
    exact timestamps the user supplied.
    """
    if (
        isinstance(time_range, (str, bytes))
        or not isinstance(time_range, Sequence)
        or len(time_range) != 2
    ):
        raise ValidationError(
            "time_range must be a (start, end) pair of ISO-8601 timestamps",
            source=_SERVER_NAME,
            detail={"parameter": "time_range"},
        )

    start_raw, end_raw = time_range
    start_dt = _parse_iso(start_raw, field="start")
    end_dt = _parse_iso(end_raw, field="end")
    if start_dt > end_dt:
        raise ValidationError(
            "time_range start must not be later than end",
            source=_SERVER_NAME,
            detail={"parameter": "time_range"},
        )
    return str(start_raw), str(end_raw)


def _parse_iso(value: Any, *, field: str) -> datetime:
    """Parse an ISO-8601 timestamp, tolerating a trailing ``Z`` (UTC)."""
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            "time_range %s must be a non-empty ISO-8601 string" % field,
            source=_SERVER_NAME,
            detail={"parameter": "time_range"},
        )
    text = value.strip()
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValidationError(
            "time_range %s is not a valid ISO-8601 timestamp: %r" % (field, text),
            source=_SERVER_NAME,
            detail={"parameter": "time_range"},
        ) from exc

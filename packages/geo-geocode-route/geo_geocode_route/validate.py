"""Input validation for ``geo-geocode-route`` (Requirement 7.12).

Both tools reject malformed parameters with an ``Error_Taxonomy``
``ValidationError`` *before* any source is queried, naming the offending
parameter (Requirement 7.12):

* :func:`validate_address` rejects an **unparseable address** - one that is not
  a string, is empty/blank, exceeds the length cap, or carries no
  alphanumeric content (e.g. ``"!!!"`` / ``"---"``) and so cannot name a
  place. The normalized (whitespace-collapsed) address is returned for the
  geocoder to use.
* :func:`validate_coordinate` rejects a malformed origin/destination - wrong
  shape, non-numeric, non-finite, or out-of-range longitude/latitude - and
  returns a :class:`~geo_geocode_route.models.Coordinate`.
* :func:`normalize_profile` rejects an unsupported travel mode and returns the
  canonical profile name.

Keeping validation closed-form and dependency-free (mirroring
``geo_vector.bbox``) makes the rejection paths easy to reason about and test.
"""

from __future__ import annotations

import math
import re
from numbers import Real
from typing import Mapping, Sequence

from geo_common.errors import ValidationError

from geo_geocode_route.models import Coordinate

__all__ = [
    "SERVER_NAME",
    "MAX_ADDRESS_LENGTH",
    "SUPPORTED_PROFILES",
    "validate_address",
    "validate_coordinate",
    "normalize_profile",
]

SERVER_NAME = "geo-geocode-route"

#: Upper bound on an address length. Anything longer is treated as malformed
#: input rather than a place name (Requirement 7.12).
MAX_ADDRESS_LENGTH = 512

#: Canonical travel profiles mapped from any accepted alias. ``route`` rejects
#: a profile outside this map as a validation error (Requirement 7.12).
SUPPORTED_PROFILES = {
    "car": "car",
    "driving": "car",
    "drive": "car",
    "auto": "car",
    "bike": "bike",
    "bicycle": "bike",
    "cycling": "bike",
    "foot": "foot",
    "walk": "foot",
    "walking": "foot",
    "pedestrian": "foot",
}

#: An address must contain at least one letter or digit to be parseable.
_ALPHANUMERIC = re.compile(r"[^\W_]", re.UNICODE)
_WHITESPACE_RUN = re.compile(r"\s+")


def validate_address(address: object) -> str:
    """Validate ``address`` and return its normalized form (Requirement 7.12).

    Rejects an unparseable address with a :class:`ValidationError` naming the
    ``address`` parameter: a non-string, an empty/whitespace-only string, a
    string longer than :data:`MAX_ADDRESS_LENGTH`, or a string with no
    alphanumeric content (which therefore names no place). On success the
    address is returned with surrounding whitespace stripped and internal
    whitespace runs collapsed to single spaces.
    """
    if not isinstance(address, str):
        raise ValidationError(
            "address must be a string, got %s" % type(address).__name__,
            source=SERVER_NAME,
            detail={"parameter": "address"},
        )
    normalized = _WHITESPACE_RUN.sub(" ", address).strip()
    if not normalized:
        raise ValidationError(
            "address must not be empty",
            source=SERVER_NAME,
            detail={"parameter": "address"},
        )
    if len(normalized) > MAX_ADDRESS_LENGTH:
        raise ValidationError(
            "address must be at most %d characters, got %d"
            % (MAX_ADDRESS_LENGTH, len(normalized)),
            source=SERVER_NAME,
            detail={"parameter": "address"},
        )
    if not _ALPHANUMERIC.search(normalized):
        raise ValidationError(
            "address is unparseable: it contains no alphanumeric characters",
            source=SERVER_NAME,
            detail={"parameter": "address"},
        )
    return normalized


def validate_coordinate(value: object, *, parameter: str) -> Coordinate:
    """Validate a coordinate and return it as a :class:`Coordinate`.

    Accepts an existing :class:`Coordinate`, a mapping with ``lon``/``lat``
    (or ``longitude``/``latitude``) keys, or a 2-item ``(lon, lat)`` sequence.
    Rejects a malformed value with a :class:`ValidationError` naming
    ``parameter``: a wrong shape, a non-numeric or non-finite ordinate, a
    longitude outside ``[-180, 180]``, or a latitude outside ``[-90, 90]``
    (Requirement 7.12).
    """
    lon, lat = _extract_lon_lat(value, parameter=parameter)
    lon_f = _as_finite_number(lon, parameter=parameter, axis="longitude")
    lat_f = _as_finite_number(lat, parameter=parameter, axis="latitude")

    if not (-180.0 <= lon_f <= 180.0):
        raise ValidationError(
            "%s longitude %g is outside [-180, 180]" % (parameter, lon_f),
            source=SERVER_NAME,
            detail={"parameter": parameter, "axis": "longitude"},
        )
    if not (-90.0 <= lat_f <= 90.0):
        raise ValidationError(
            "%s latitude %g is outside [-90, 90]" % (parameter, lat_f),
            source=SERVER_NAME,
            detail={"parameter": parameter, "axis": "latitude"},
        )
    return Coordinate(lon=lon_f, lat=lat_f)


def normalize_profile(profile: object) -> str:
    """Return the canonical travel profile for ``profile`` (Requirement 7.12).

    Maps any accepted alias (e.g. ``"driving"`` -> ``"car"``) to its canonical
    name; rejects an unsupported or non-string profile with a
    :class:`ValidationError` naming the ``profile`` parameter.
    """
    if not isinstance(profile, str):
        raise ValidationError(
            "profile must be a string, got %s" % type(profile).__name__,
            source=SERVER_NAME,
            detail={"parameter": "profile"},
        )
    key = profile.strip().lower()
    canonical = SUPPORTED_PROFILES.get(key)
    if canonical is None:
        raise ValidationError(
            "unsupported profile %r; supported: %s"
            % (profile, ", ".join(sorted(set(SUPPORTED_PROFILES.values())))),
            source=SERVER_NAME,
            detail={"parameter": "profile"},
        )
    return canonical


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_lon_lat(value: object, *, parameter: str):
    """Pull a ``(lon, lat)`` pair out of the accepted coordinate shapes."""
    if isinstance(value, Coordinate):
        return value.lon, value.lat
    if isinstance(value, Mapping):
        lon = value.get("lon", value.get("longitude"))
        lat = value.get("lat", value.get("latitude"))
        if lon is None or lat is None:
            raise ValidationError(
                "%s mapping must provide 'lon'/'lat' (or 'longitude'/'latitude')"
                % parameter,
                source=SERVER_NAME,
                detail={"parameter": parameter},
            )
        return lon, lat
    # A string is a Sequence too; reject it explicitly so its characters are
    # never iterated as ordinates.
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(
            "%s must be a Coordinate, a {lon, lat} mapping, or a (lon, lat) pair"
            % parameter,
            source=SERVER_NAME,
            detail={"parameter": parameter},
        )
    items = list(value)
    if len(items) != 2:
        raise ValidationError(
            "%s must be a (lon, lat) pair, got %d value(s)" % (parameter, len(items)),
            source=SERVER_NAME,
            detail={"parameter": parameter},
        )
    return items[0], items[1]


def _as_finite_number(value: object, *, parameter: str, axis: str) -> float:
    """Coerce ``value`` to a finite float or raise a validation error."""
    # bool is an int subclass; reject it as a non-coordinate value.
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(
            "%s %s must be a number, got %r" % (parameter, axis, value),
            source=SERVER_NAME,
            detail={"parameter": parameter, "axis": axis},
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(
            "%s %s must be finite" % (parameter, axis),
            source=SERVER_NAME,
            detail={"parameter": parameter, "axis": axis},
        )
    return number

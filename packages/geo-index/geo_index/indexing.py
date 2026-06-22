"""H3 / S2 spatial-index cell lookups for ``geo-index`` (Requirements 8.9, 8.10).

This module holds the pure, local cell-lookup logic the ``geo-index`` server
exposes as the ``index_cell`` MCP tool (task 14.9):

* :func:`index_cell` - return the H3 cell identifier (resolutions 0-15) or the
  S2 cell identifier (levels 0-30) for a coordinate at a requested resolution
  (Requirement 8.9). The function is synchronous and side-effect free; the
  server (see :mod:`geo_index.server`) wraps it as an ``async`` MCP tool.

The lookup is **deterministic** over the valid input space (design Property 15;
Requirement 8.9): the same ``(lon, lat, scheme, resolution)`` always yields the
same identifier, and that identifier is well-formed for the selected scheme
(an H3 cell token for H3, an S2 cell token for S2).

Out-of-range input raises a ``geo_common``
:class:`~geo_common.errors.ValidationError` so failures surface on the shared
``Error_Taxonomy`` (Requirement 8.10; design Property 4) rather than as raw
library exceptions, and no partial output is produced. Specifically:

* an unknown indexing ``scheme`` (not ``"h3"`` or ``"s2"``),
* a ``resolution`` outside the valid range for the selected scheme
  (H3 0-15, S2 0-30), or
* a coordinate outside longitude ``[-180, 180]`` or latitude ``[-90, 90]``
  (which also rejects ``NaN`` / non-finite coordinates),

each raise a :class:`ValidationError` whose message and ``detail`` identify the
out-of-range parameter.

Python 3.10+ (matching ``pyproject.toml``); uses ``from __future__ import
annotations`` so the type hints resolve lazily.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

from geo_common.errors import ValidationError

__all__ = [
    "H3",
    "S2",
    "SCHEMES",
    "H3_MIN_RESOLUTION",
    "H3_MAX_RESOLUTION",
    "S2_MIN_LEVEL",
    "S2_MAX_LEVEL",
    "LON_MIN",
    "LON_MAX",
    "LAT_MIN",
    "LAT_MAX",
    "resolution_range",
    "index_cell",
]

#: Canonical scheme identifiers accepted by :func:`index_cell`.
H3 = "h3"
S2 = "s2"
SCHEMES: Tuple[str, str] = (H3, S2)

#: Inclusive H3 resolution range (Requirement 8.9).
H3_MIN_RESOLUTION = 0
H3_MAX_RESOLUTION = 15

#: Inclusive S2 level range (Requirement 8.9).
S2_MIN_LEVEL = 0
S2_MAX_LEVEL = 30

#: Inclusive coordinate bounds (Requirement 8.10).
LON_MIN, LON_MAX = -180.0, 180.0
LAT_MIN, LAT_MAX = -90.0, 90.0

#: Per-scheme inclusive resolution range, used for validation and messaging.
_RESOLUTION_RANGE: Dict[str, Tuple[int, int]] = {
    H3: (H3_MIN_RESOLUTION, H3_MAX_RESOLUTION),
    S2: (S2_MIN_LEVEL, S2_MAX_LEVEL),
}


def _normalize_scheme(scheme: str, *, source: str) -> str:
    """Return the canonical scheme id (``"h3"``/``"s2"``) or raise.

    Accepts any case (e.g. ``"H3"``, ``"S2"``). An unknown scheme raises a
    taxonomy :class:`ValidationError` naming ``scheme`` as the offending
    parameter (Requirement 8.10).
    """
    if not isinstance(scheme, str):
        raise ValidationError(
            "scheme must be a string, one of %s" % (list(SCHEMES),),
            source=source,
            detail={"parameter": "scheme", "value_type": type(scheme).__name__},
        )
    normalized = scheme.strip().lower()
    if normalized not in _RESOLUTION_RANGE:
        raise ValidationError(
            "unknown indexing scheme %r; expected one of %s"
            % (scheme, list(SCHEMES)),
            source=source,
            detail={"parameter": "scheme", "value": scheme, "allowed": list(SCHEMES)},
        )
    return normalized


def resolution_range(scheme: str) -> Tuple[int, int]:
    """Return the inclusive ``(min, max)`` resolution range for ``scheme``.

    ``scheme`` is matched case-insensitively. Raises :class:`ValidationError`
    for an unknown scheme.
    """
    return _RESOLUTION_RANGE[_normalize_scheme(scheme, source="geo-index")]


def _validate_resolution(resolution: int, scheme: str, *, source: str) -> int:
    """Validate ``resolution`` against the selected ``scheme``'s range.

    ``resolution`` must be an integer within the scheme's inclusive range
    (H3 0-15, S2 0-30). A ``bool`` is rejected (it is an ``int`` subclass but
    not a meaningful resolution). Out-of-range or non-integer values raise a
    taxonomy :class:`ValidationError` identifying ``resolution`` (Req 8.10).
    """
    if isinstance(resolution, bool) or not isinstance(resolution, int):
        raise ValidationError(
            "resolution must be an integer",
            source=source,
            detail={"parameter": "resolution", "value_type": type(resolution).__name__},
        )
    low, high = _RESOLUTION_RANGE[scheme]
    if resolution < low or resolution > high:
        raise ValidationError(
            "resolution %d is outside the valid range for scheme %r (%d-%d)"
            % (resolution, scheme, low, high),
            source=source,
            detail={
                "parameter": "resolution",
                "value": resolution,
                "scheme": scheme,
                "min": low,
                "max": high,
            },
        )
    return resolution


def _validate_coordinate(lon: float, lat: float, *, source: str) -> Tuple[float, float]:
    """Validate ``lon``/``lat`` against the WGS84 coordinate bounds (Req 8.10).

    Longitude must lie in ``[-180, 180]`` and latitude in ``[-90, 90]``.
    ``NaN`` / infinite coordinates fail the range check (a comparison against
    ``NaN`` is always false) and raise a taxonomy :class:`ValidationError`
    naming the offending coordinate.
    """
    for name, value, low, high in (
        ("lon", lon, LON_MIN, LON_MAX),
        ("lat", lat, LAT_MIN, LAT_MAX),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(
                "%s must be a number" % name,
                source=source,
                detail={"parameter": name, "value_type": type(value).__name__},
            )
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < low or numeric > high:
            raise ValidationError(
                "%s %r is outside the valid range [%g, %g]" % (name, value, low, high),
                source=source,
                detail={"parameter": name, "value": value, "min": low, "max": high},
            )
    return float(lon), float(lat)


def _h3_cell(lon: float, lat: float, resolution: int) -> str:
    """Return the H3 cell token for ``(lat, lon)`` at ``resolution`` (0-15)."""
    import h3

    # h3 v4 takes latitude first, then longitude, then resolution, and returns
    # the canonical H3 index string (e.g. "8928341aec3ffff").
    return h3.latlng_to_cell(lat, lon, resolution)


def _s2_cell(lon: float, lat: float, level: int) -> str:
    """Return the S2 cell token for ``(lat, lon)`` at ``level`` (0-30).

    The leaf cell (level 30) for the coordinate is computed, then truncated to
    the requested ``level`` via ``parent(level)``; ``to_token()`` yields the
    canonical compact S2 token string.
    """
    import s2sphere

    latlng = s2sphere.LatLng.from_degrees(lat, lon)
    leaf = s2sphere.CellId.from_lat_lng(latlng)
    cell = leaf.parent(level)
    return cell.to_token()


def index_cell(
    *,
    lon: float,
    lat: float,
    scheme: str,
    resolution: int,
    source: str = "geo-index",
) -> str:
    """Return the spatial-index cell id for a coordinate (Req 8.9).

    For ``scheme == "h3"`` returns the H3 cell identifier at the requested
    ``resolution`` (0-15); for ``scheme == "s2"`` returns the S2 cell
    identifier at the requested level (0-30). ``scheme`` is matched
    case-insensitively. The result is well-formed for the selected scheme and
    deterministic: the same input always yields the same identifier (design
    Property 15).

    Raises :class:`~geo_common.errors.ValidationError` (taxonomy
    ``validation``) - producing no output - when ``scheme`` is unknown, when
    ``resolution`` is outside the scheme's valid range, or when the coordinate
    is outside longitude ``[-180, 180]`` / latitude ``[-90, 90]``
    (Requirement 8.10). The error identifies the out-of-range parameter.
    """
    normalized = _normalize_scheme(scheme, source=source)
    _validate_resolution(resolution, normalized, source=source)
    _validate_coordinate(lon, lat, source=source)

    if normalized == H3:
        return _h3_cell(lon, lat, resolution)
    return _s2_cell(lon, lat, resolution)

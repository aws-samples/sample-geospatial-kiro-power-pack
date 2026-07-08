"""Terrain derivatives - slope and hillshade - over an elevation grid (Req 7.5).

``geo-terrain`` fetches a sampled elevation grid for an extent (see
:func:`geo_terrain.elevation.elevation`); this module turns that grid into the
two standard first-order derivatives:

* :func:`compute_slope` - per-cell slope in **degrees**, from the gradient
  magnitude of the surface (Horn-style, via :func:`numpy.gradient`).
* :func:`compute_hillshade` - per-cell shaded-relief value in ``[0, 255]`` for a
  given sun ``azimuth``/``altitude``, the standard illumination model used by
  GDAL ``hillshade``.

Both are pure functions of the grid plus the ground cell size (metres), so they
are deterministic and unit-testable without any network I/O. Ground cell size
is derived from the geographic extent with :func:`meters_per_degree` (a local
equirectangular approximation, adequate for the small extents these tools
serve). Cells with no elevation data (``None``) propagate as ``None`` in the
output rather than being silently treated as zero.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

__all__ = [
    "meters_per_degree",
    "compute_slope",
    "compute_aspect",
    "compute_hillshade",
]

#: Aspect value for a flat cell (undefined downslope direction), matching GDAL.
FLAT_ASPECT = -1.0

#: Metres per degree of latitude (mean; WGS84 is ~110.57 km at the equator to
#: ~111.69 km at the poles - this mean is adequate for small extents).
_M_PER_DEG_LAT = 110_540.0
#: Metres per degree of longitude at the equator (scaled by cos(lat)).
_M_PER_DEG_LON_EQUATOR = 111_320.0


def meters_per_degree(lat_deg: float) -> "tuple[float, float]":
    """Return ``(metres_per_deg_lon, metres_per_deg_lat)`` at ``lat_deg``.

    Longitude metres shrink with ``cos(latitude)``; latitude metres are treated
    as constant. This is a local equirectangular approximation - accurate enough
    for the bounded extents ``slope``/``hillshade`` operate on, and keeps the
    computation dependency-light (no pyproj needed).
    """
    lon_m = _M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat_deg))
    return (abs(lon_m), _M_PER_DEG_LAT)


def _to_array(values: List[List[Optional[float]]]) -> np.ndarray:
    """Convert a row-major grid (with optional ``None``) to a float array (NaN holes)."""
    return np.array(
        [[(np.nan if v is None else float(v)) for v in row] for row in values],
        dtype=np.float64,
    )


def _gradients(
    grid: np.ndarray, cellsize_x_m: float, cellsize_y_m: float
) -> "tuple[np.ndarray, np.ndarray]":
    """Return ``(dz/dx, dz/dy)`` in metres-per-metre (east+, north+).

    Rows run north→south, so the north-positive gradient is the negative of the
    row gradient. Columns run west→east, so the east-positive gradient is the
    column gradient.
    """
    dz_drow, dz_dcol = np.gradient(grid)
    dz_dx = dz_dcol / cellsize_x_m  # east-positive
    dz_dy = -dz_drow / cellsize_y_m  # north-positive (row increases southward)
    return dz_dx, dz_dy


def compute_slope(
    values: List[List[Optional[float]]],
    *,
    cellsize_x_m: float,
    cellsize_y_m: float,
) -> List[List[Optional[float]]]:
    """Per-cell slope in **degrees** for an elevation grid.

    Slope is ``atan(sqrt((dz/dx)^2 + (dz/dy)^2))`` over the ground gradients.
    A flat surface yields ``0``; a vertical cliff approaches ``90``. Cells whose
    gradient involves a no-data neighbour come back as ``None``.
    """
    grid = _to_array(values)
    dz_dx, dz_dy = _gradients(grid, cellsize_x_m, cellsize_y_m)
    slope_rad = np.arctan(np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy))
    slope_deg = np.degrees(slope_rad)
    return _to_grid(slope_deg)


def compute_aspect(
    values: List[List[Optional[float]]],
    *,
    cellsize_x_m: float,
    cellsize_y_m: float,
) -> List[List[Optional[float]]]:
    """Per-cell aspect in **compass degrees** ``[0, 360)`` for an elevation grid.

    Aspect is the compass bearing of the **downslope** direction (the way water
    would flow), measured clockwise from north (0 = north-facing, 90 = east,
    180 = south, 270 = west). With east-positive ``dz/dx`` and north-positive
    ``dz/dy``, the downslope vector is ``(-dz/dx, -dz/dy)`` in (east, north), so
    ``aspect = atan2(-dz/dx, -dz/dy)`` normalized to ``[0, 360)``.

    Flat cells (no gradient) have no defined aspect and come back as
    :data:`FLAT_ASPECT` (``-1``), matching GDAL. Cells whose gradient involves a
    no-data neighbour come back as ``None``.
    """
    grid = _to_array(values)
    dz_dx, dz_dy = _gradients(grid, cellsize_x_m, cellsize_y_m)
    aspect_deg = np.degrees(np.arctan2(-dz_dx, -dz_dy)) % 360.0
    flat = (np.abs(dz_dx) < 1e-12) & (np.abs(dz_dy) < 1e-12)
    aspect_deg = np.where(flat, FLAT_ASPECT, aspect_deg)
    return _to_grid(aspect_deg)


def compute_hillshade(
    values: List[List[Optional[float]]],
    *,
    cellsize_x_m: float,
    cellsize_y_m: float,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
) -> List[List[Optional[int]]]:
    """Per-cell shaded-relief value in ``[0, 255]`` (standard hillshade model).

    Uses the GDAL ``hillshade`` illumination model: with sun zenith
    ``90 - altitude`` and the surface slope/aspect from the ground gradients,
    ``255 * (cos(zenith)cos(slope) + sin(zenith)sin(slope)cos(azimuth - aspect))``,
    clamped to ``[0, 255]``. ``azimuth_deg`` is the sun direction clockwise from
    north (default 315 = NW); ``altitude_deg`` is its height above the horizon
    (default 45). Cells with no-data neighbours come back as ``None``.
    """
    grid = _to_array(values)
    dz_dx, dz_dy = _gradients(grid, cellsize_x_m, cellsize_y_m)
    slope_rad = np.arctan(np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy))
    # Aspect: compass direction the slope faces, clockwise from north.
    aspect_rad = np.arctan2(dz_dy, -dz_dx)
    zenith_rad = math.radians(90.0 - altitude_deg)
    azimuth_rad = math.radians(360.0 - azimuth_deg + 90.0)
    shaded = (
        math.cos(zenith_rad) * np.cos(slope_rad)
        + math.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
    )
    hillshade = np.clip(255.0 * shaded, 0.0, 255.0)
    out: List[List[Optional[int]]] = []
    for row in hillshade:
        out.append([(None if np.isnan(v) else int(round(v))) for v in row])
    return out


def _to_grid(arr: np.ndarray) -> List[List[Optional[float]]]:
    """Convert a float array back to a row-major grid, NaN → ``None``."""
    return [
        [(None if np.isnan(v) else float(v)) for v in row]
        for row in arr
    ]

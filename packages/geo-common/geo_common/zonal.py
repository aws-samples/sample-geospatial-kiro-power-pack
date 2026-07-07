"""Zonal statistics over a raster and a set of vector zones (Req 8.7, 8.8).

This module implements the heart of ``geo-raster``:

* :func:`compute_zonal_statistics` — a pure, reference-grade computation over an
  in-memory :class:`~geo_raster.models.RasterGrid`. For each zone it gathers the
  cells whose **center** falls inside the zone polygon (the standard rasterstats
  "centroid" rule), excludes nodata cells, and computes the requested statistics
  (Requirement 8.7). A zone with no overlapping cells gets a no-data indication —
  every requested statistic ``None`` — while the remaining zones still receive
  statistics (Requirement 8.8). This function performs no I/O, so a test can
  match it against a straightforward reference implementation (design Property
  14).
* :func:`zonal_statistics` — the public capability. It validates the requested
  statistics, reads only the raster window overlapping the union of the zones
  (via :func:`~geo_raster.reader.read_raster_grid`, which reads directly from S3
  byte ranges and aborts leaving local storage unchanged on an S3 read failure —
  Requirement 12.6), then delegates to :func:`compute_zonal_statistics`.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from geo_common.errors import ValidationError
from geo_common.geometry import geometry_bbox, iter_polygons, point_in_geometry
from geo_common.raster_models import (
    DEFAULT_STATISTICS,
    SUPPORTED_STATISTICS,
    FeatureCollection,
    RasterGrid,
    ZoneStat,
)

__all__ = ["compute_zonal_statistics", "_zones_bbox"]

BBox = Tuple[float, float, float, float]


def _validate_stats(stats: Optional[Sequence[str]], *, source: str) -> List[str]:
    """Validate and normalize the requested statistics list (Requirement 8.7)."""
    if stats is None:
        return list(DEFAULT_STATISTICS)
    requested = list(stats)
    if not requested:
        raise ValidationError(
            "stats must name at least one statistic; supported: "
            + ", ".join(SUPPORTED_STATISTICS),
            source=source,
            detail={"parameter": "stats"},
        )
    seen: List[str] = []
    for name in requested:
        if name not in SUPPORTED_STATISTICS:
            raise ValidationError(
                f"unsupported statistic {name!r}; supported: "
                + ", ".join(SUPPORTED_STATISTICS),
                source=source,
                detail={"parameter": "stats"},
            )
        if name not in seen:
            seen.append(name)
    return seen


def _zone_id(properties: dict, index: int) -> str:
    """Resolve a zone's identifier from its properties, falling back to index."""
    for key in ("zone_id", "id", "ID", "name"):
        if key in properties and properties[key] is not None:
            return str(properties[key])
    return str(index)


def _statistic_value(name: str, values: List[float]) -> Optional[float]:
    """Compute one statistic over the (non-empty) overlapping cell ``values``."""
    if name == "count":
        return float(len(values))
    if name == "sum":
        return float(sum(values))
    if name == "min":
        return float(min(values))
    if name == "max":
        return float(max(values))
    if name == "mean":
        return float(sum(values) / len(values))
    if name == "std":
        # Population standard deviation (ddof=0), matching rasterstats/NumPy.
        # A single-cell zone has zero spread, so this returns 0.0.
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return float(variance ** 0.5)
    raise ValidationError(  # pragma: no cover - guarded by _validate_stats
        f"unsupported statistic {name!r}", detail={"parameter": "stats"}
    )


def _numpy_available() -> bool:
    """Return whether the optional ``numpy`` fast-path dependency is importable."""
    try:
        import numpy  # noqa: F401
    except Exception:  # pragma: no cover - environment-dependent
        return False
    return True


def compute_zonal_statistics(
    grid: RasterGrid,
    zones: FeatureCollection,
    stats: Optional[Sequence[str]] = None,
    *,
    source: str = "geo-raster",
    backend: str = "auto",
) -> List[ZoneStat]:
    """Compute per-zone statistics over an in-memory grid (Req 8.7, 8.8).

    A cell contributes to a zone when its center lies inside the zone polygon
    and its value is not the grid's ``nodata`` (and not NaN). A zone with no
    contributing cells yields a no-data indication (every requested statistic
    ``None``); the remaining zones still receive statistics (Requirement 8.8).

    ``backend`` selects the compute engine: ``"pure"`` uses the dependency-free
    Python engine; ``"numpy"`` uses the vectorized fast path (requires the
    optional ``numpy`` extra); ``"auto"`` (default) uses numpy when available and
    otherwise falls back to pure. All backends are semantically identical
    (verified by the differential property test).
    """
    requested = _validate_stats(stats, source=source)
    if backend not in ("auto", "pure", "numpy"):
        raise ValidationError(
            f"unknown backend {backend!r}; use 'auto', 'pure', or 'numpy'",
            source=source,
            detail={"parameter": "backend"},
        )
    use_numpy = backend == "numpy" or (backend == "auto" and _numpy_available())
    if use_numpy:
        if not _numpy_available():
            raise ValidationError(
                "backend='numpy' requires the optional numpy extra "
                "(install geo-raster[fast])",
                source=source,
                detail={"parameter": "backend"},
            )
        from geo_common.zonal_fast import compute_zonal_statistics_numpy

        return compute_zonal_statistics_numpy(grid, zones, requested, source=source)
    return _compute_zonal_statistics_pure(grid, zones, requested, source=source)


def _compute_zonal_statistics_pure(
    grid: RasterGrid,
    zones: FeatureCollection,
    requested: List[str],
    *,
    source: str = "geo-raster",
) -> List[ZoneStat]:
    """Dependency-free reference engine for :func:`compute_zonal_statistics`."""
    nodata = grid.nodata

    # Pre-compute every cell's center coordinate and value once.
    centers: List[Tuple[float, float, float]] = []
    for row in range(grid.height):
        for col in range(grid.width):
            value = grid.values[row * grid.width + col]
            if nodata is not None and value == nodata:
                continue
            if value != value:  # NaN nodata
                continue
            x, y = grid.cell_center(col, row)
            centers.append((x, y, value))

    results: List[ZoneStat] = []
    for index, feature in enumerate(zones.features):
        zone_id = _zone_id(feature.properties or {}, index)
        if feature.geometry is None:
            raise ValidationError(
                f"zone {zone_id!r} has no geometry",
                source=source,
                detail={"parameter": "zones"},
            )
        polygons = iter_polygons(feature.geometry.to_geojson(), source=source)

        overlapping = [v for x, y, v in centers if point_in_geometry(polygons, x, y)]

        if not overlapping:
            # No overlapping cells -> no-data indication for this zone (Req 8.8).
            results.append(
                ZoneStat(
                    zone_id=zone_id,
                    statistics={name: None for name in requested},
                    no_data=True,
                )
            )
            continue

        results.append(
            ZoneStat(
                zone_id=zone_id,
                statistics={name: _statistic_value(name, overlapping) for name in requested},
                no_data=False,
            )
        )
    return results


def _zones_bbox(zones: FeatureCollection, *, source: str) -> Optional[BBox]:
    """Return the union bbox of all zone geometries (None if no zones)."""
    boxes: List[BBox] = []
    for index, feature in enumerate(zones.features):
        if feature.geometry is None:
            raise ValidationError(
                f"zone at index {index} has no geometry",
                source=source,
                detail={"parameter": "zones"},
            )
        box = geometry_bbox(feature.geometry.to_geojson(), source=source)
        if box is not None:
            boxes.append(box)
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )

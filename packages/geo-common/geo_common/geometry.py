"""Self-contained planar geometry helpers for zonal statistics.

``zonal_statistics`` assigns a raster cell to a vector zone when the cell's
**center** falls inside the zone polygon (the standard rasterstats "centroid"
rule). To stay dependency-free (``geo-raster`` depends only on ``geo-common``),
this module implements the small amount of computational geometry that requires:

* :func:`polygon_bbox` / :func:`geometry_bbox` — the axis-aligned bounding box
  of a polygon ring set or a GeoJSON geometry, used to pick the raster window
  to read.
* :func:`point_in_polygon` — even-odd ray-casting point-in-polygon for a single
  GeoJSON ``Polygon`` (exterior ring minus holes).
* :func:`point_in_geometry` — point membership for a ``Polygon`` or
  ``MultiPolygon`` geometry.

The geometries are plain GeoJSON coordinate arrays (lists of ``[x, y]``), matching
:class:`~geo_raster.models.GeoJSONGeometry`. Coordinates are interpreted in the
raster's own coordinate reference system; aligning zone and raster CRS is the
caller's responsibility (see the ``zonal-statistics`` steering workflow).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from geo_common.errors import ValidationError

__all__ = [
    "AREAL_GEOMETRY_TYPES",
    "polygon_bbox",
    "geometry_bbox",
    "point_in_polygon",
    "point_in_geometry",
    "iter_polygons",
]

#: The GeoJSON geometry types ``zonal_statistics`` can summarize (areal zones).
AREAL_GEOMETRY_TYPES = ("Polygon", "MultiPolygon")

BBox = Tuple[float, float, float, float]
Ring = Sequence[Sequence[float]]


def _coerce_ring(ring: Any, *, source: str) -> List[Tuple[float, float]]:
    """Validate and coerce a coordinate ring to a list of ``(x, y)`` tuples."""
    if not isinstance(ring, (list, tuple)) or len(ring) < 3:
        raise ValidationError(
            "polygon ring must have at least 3 coordinates",
            source=source,
            detail={"parameter": "zones"},
        )
    out: List[Tuple[float, float]] = []
    for pt in ring:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            raise ValidationError(
                "polygon coordinate must be an [x, y] pair",
                source=source,
                detail={"parameter": "zones"},
            )
        try:
            x = float(pt[0])
            y = float(pt[1])
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "polygon coordinate components must be numeric",
                source=source,
                detail={"parameter": "zones"},
            ) from exc
        if x != x or y != y or x in (float("inf"), float("-inf")) or y in (
            float("inf"),
            float("-inf"),
        ):
            raise ValidationError(
                "polygon coordinate components must be finite",
                source=source,
                detail={"parameter": "zones"},
            )
        out.append((x, y))
    return out


def iter_polygons(geometry: Any, *, source: str = "geo-raster") -> List[List[List[Tuple[float, float]]]]:
    """Return a geometry as a list of polygons, each a list of coerced rings.

    A ``Polygon`` yields one polygon; a ``MultiPolygon`` yields several. Each
    polygon is ``[exterior_ring, hole_ring, ...]``. Raises a
    :class:`~geo_common.errors.ValidationError` for a non-areal geometry or a
    malformed coordinate array (Requirement 7.12 / 8.10 validation surface).
    """
    gtype = geometry.get("type") if isinstance(geometry, dict) else None
    coords = geometry.get("coordinates") if isinstance(geometry, dict) else None
    if gtype not in AREAL_GEOMETRY_TYPES:
        raise ValidationError(
            "zonal statistics requires Polygon or MultiPolygon zones; got "
            f"{gtype!r}",
            source=source,
            detail={"parameter": "zones"},
        )
    if not isinstance(coords, (list, tuple)) or not coords:
        raise ValidationError(
            "zone geometry has empty or malformed coordinates",
            source=source,
            detail={"parameter": "zones"},
        )

    polygons: List[List[List[Tuple[float, float]]]] = []
    if gtype == "Polygon":
        raw_polys = [coords]
    else:  # MultiPolygon
        raw_polys = list(coords)
    for poly in raw_polys:
        if not isinstance(poly, (list, tuple)) or not poly:
            raise ValidationError(
                "polygon must have at least an exterior ring",
                source=source,
                detail={"parameter": "zones"},
            )
        rings = [_coerce_ring(ring, source=source) for ring in poly]
        polygons.append(rings)
    return polygons


def polygon_bbox(rings: Sequence[Sequence[Tuple[float, float]]]) -> Optional[BBox]:
    """Return the ``(min_x, min_y, max_x, max_y)`` bbox of a polygon's rings."""
    xs: List[float] = []
    ys: List[float] = []
    for ring in rings:
        for x, y in ring:
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def geometry_bbox(geometry: Any, *, source: str = "geo-raster") -> Optional[BBox]:
    """Return the bbox of a ``Polygon`` / ``MultiPolygon`` geometry."""
    boxes = [polygon_bbox(rings) for rings in iter_polygons(geometry, source=source)]
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _ring_contains(ring: Sequence[Tuple[float, float]], x: float, y: float) -> bool:
    """Even-odd ray-casting test: is ``(x, y)`` inside the closed ``ring``?"""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        # Does the horizontal ray from (x, y) cross edge (j -> i)?
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def point_in_polygon(rings: Sequence[Sequence[Tuple[float, float]]], x: float, y: float) -> bool:
    """Return whether ``(x, y)`` lies inside a polygon (exterior minus holes)."""
    if not rings:
        return False
    if not _ring_contains(rings[0], x, y):
        return False
    for hole in rings[1:]:
        if _ring_contains(hole, x, y):
            return False
    return True


def point_in_geometry(
    polygons: Sequence[Sequence[Sequence[Tuple[float, float]]]], x: float, y: float
) -> bool:
    """Return whether ``(x, y)`` lies inside any polygon of a (multi)polygon."""
    for rings in polygons:
        if point_in_polygon(rings, x, y):
            return True
    return False

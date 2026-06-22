"""Data models for the ``geo-raster`` server (Pillar B, expansion).

These models describe the inputs and output of ``zonal_statistics``
(Requirements 8.7, 8.8):

* :class:`GeoJSONGeometry` / :class:`Feature` / :class:`FeatureCollection` — the
  GeoJSON vector *zones* a raster is summarized over. They are defined locally
  (a minimal, permissive subset) so ``geo-raster`` stays self-contained and
  depends only on ``geo-common`` (its ``pyproject.toml`` declares no other
  server package — installing ``geo-raster`` must not pull other servers).
* :class:`RasterGrid` — a single-band, georeferenced block of raster cells (the
  window read from the asset). Carries the cell values, the cell→world
  ``geotransform`` (GDAL 6-tuple), and the optional ``nodata`` value, and knows
  how to map a ``(col, row)`` cell to its center's world coordinate. The zonal
  engine computes statistics over the cells of a :class:`RasterGrid`.
* :class:`ZoneStat` — the per-zone result: the requested statistics, or a
  no-data indication (every requested statistic ``None``) for a zone with no
  overlapping cells (Requirement 8.8).

Python 3.10+ (matching ``pyproject.toml``): uses ``from __future__ import
annotations`` together with ``typing`` generics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

__all__ = [
    "GeoJSONGeometry",
    "Feature",
    "FeatureCollection",
    "RasterGrid",
    "ZoneStat",
    "SUPPORTED_STATISTICS",
    "DEFAULT_STATISTICS",
]

#: The statistics ``zonal_statistics`` can compute (Requirement 8.7). The set
#: includes at least minimum, maximum, mean, sum, and count.
SUPPORTED_STATISTICS: Tuple[str, ...] = ("min", "max", "mean", "sum", "count")

#: The default statistics returned when a caller does not name a subset.
DEFAULT_STATISTICS: List[str] = ["min", "max", "mean", "sum", "count"]


class GeoJSONGeometry(BaseModel):
    """A single GeoJSON geometry object (a zone boundary).

    ``coordinates`` is typed permissively (``Any``) so a structurally unusual
    geometry can be *constructed* and then *reported on* by the zonal engine
    (which raises a validation error for non-areal geometry) rather than being
    rejected at model-construction time. ``zonal_statistics`` only summarizes
    areal zones (``Polygon`` / ``MultiPolygon``).
    """

    type: str = Field(min_length=1, description="GeoJSON geometry type, e.g. 'Polygon'.")
    coordinates: Optional[Any] = Field(
        default=None,
        description="GeoJSON coordinate array; omitted for GeometryCollection.",
    )
    geometries: Optional[List["GeoJSONGeometry"]] = Field(
        default=None,
        description="Member geometries; present only for GeometryCollection.",
    )

    def to_geojson(self) -> Dict[str, Any]:
        """Return a plain GeoJSON ``dict`` for the geometry."""
        if self.type == "GeometryCollection":
            members = self.geometries or []
            return {
                "type": "GeometryCollection",
                "geometries": [m.to_geojson() for m in members],
            }
        return {"type": self.type, "coordinates": self.coordinates}

    @classmethod
    def from_geojson(cls, mapping: Dict[str, Any]) -> "GeoJSONGeometry":
        """Build a :class:`GeoJSONGeometry` from a GeoJSON mapping ``dict``."""
        gtype = mapping.get("type", "")
        if gtype == "GeometryCollection":
            members = [cls.from_geojson(g) for g in mapping.get("geometries", [])]
            return cls(type=gtype, geometries=members)
        return cls(type=gtype, coordinates=mapping.get("coordinates"))


class Feature(BaseModel):
    """A GeoJSON Feature: a zone geometry plus arbitrary ``properties``.

    A zone's identifier is taken from ``properties['zone_id']`` (or ``'id'``)
    when present; otherwise the zone's position in the collection is used.
    """

    type: str = Field(default="Feature")
    geometry: Optional[GeoJSONGeometry] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class FeatureCollection(BaseModel):
    """A GeoJSON FeatureCollection of vector zones consumed by zonal stats."""

    type: str = Field(default="FeatureCollection")
    features: List[Feature] = Field(default_factory=list)


class RasterGrid(BaseModel):
    """A single-band, georeferenced block of raster cells (Requirement 8.7).

    ``values`` holds ``width * height`` cell values in row-major order (row 0
    first, left to right). ``geotransform`` is a GDAL-style 6-tuple
    ``(origin_x, px_w, row_rot, origin_y, col_rot, px_h)`` mapping a fractional
    ``(col, row)`` to a world coordinate; for a north-up raster ``px_h`` is
    negative. ``nodata``, when set, marks cells excluded from statistics.
    """

    width: int = Field(ge=0)
    height: int = Field(ge=0)
    geotransform: Tuple[float, float, float, float, float, float]
    nodata: Optional[float] = None
    dtype: str = "float64"
    values: List[float] = Field(default_factory=list)

    def value(self, col: int, row: int) -> float:
        """Return the cell value at ``(col, row)`` (0-based)."""
        if not (0 <= col < self.width and 0 <= row < self.height):
            raise IndexError("cell coordinate out of grid bounds")
        return self.values[row * self.width + col]

    def cell_center(self, col: int, row: int) -> Tuple[float, float]:
        """Return the world ``(x, y)`` coordinate of cell ``(col, row)``'s center."""
        origin_x, px_w, row_rot, origin_y, col_rot, px_h = self.geotransform
        fc = col + 0.5
        fr = row + 0.5
        x = origin_x + px_w * fc + row_rot * fr
        y = origin_y + col_rot * fc + px_h * fr
        return x, y


class ZoneStat(BaseModel):
    """Per-zone zonal-statistics result (Requirements 8.7, 8.8).

    ``statistics`` maps each requested statistic name to its value, or to
    ``None`` when the zone has no overlapping raster cells — the no-data
    indication (Requirement 8.8). ``no_data`` is a convenience flag that is
    ``True`` exactly when every requested statistic is ``None``.
    """

    zone_id: str
    statistics: Dict[str, Optional[float]]
    no_data: bool = False

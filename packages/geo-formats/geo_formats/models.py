"""Data models for the ``geo-formats`` server (Pillar B, expansion).

``geo-formats`` converts rasters to Cloud-Optimized GeoTIFF (COG) and vector
datasets to GeoParquet, and validates that an output really is in the claimed
format (design.md "Pillar B — Processing and Compute"). These models describe
the inputs the conversion tools accept and the results they return:

* :class:`GeoJSONGeometry` / :class:`Feature` / :class:`FeatureCollection` —
  the GeoJSON-shaped vector inputs :func:`~geo_formats.geoparquet.to_geoparquet`
  consumes. They mirror the equivalent ``geo-ops`` models so a feature
  collection produced by one server flows into the other unchanged.
* :class:`FormatResult` — what a successful conversion returns: the written
  output ``href``, the produced ``fmt`` and ``driver``, the output size, and a
  small ``detail`` map (band/overview counts for COG, feature/column counts for
  GeoParquet).
* :class:`FormatValidity` — the result of
  :func:`~geo_formats.validate.validate_format`: a ``valid`` flag plus a
  non-empty ``reason`` populated exactly when the output is *not* a valid
  instance of the requested format.

Python 3.10+ (matching ``pyproject.toml``): uses ``from __future__ import
annotations`` together with ``typing`` generics so the pydantic models resolve
their annotations cleanly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

__all__ = [
    "GeoJSONGeometry",
    "Feature",
    "FeatureCollection",
    "FormatResult",
    "FormatValidity",
]


class GeoJSONGeometry(BaseModel):
    """A single GeoJSON geometry object.

    Carries the GeoJSON ``type`` plus either ``coordinates`` (for the seven
    primitive/multi types) or ``geometries`` (for ``GeometryCollection``). The
    ``coordinates`` field is typed permissively (``Any``) so a feature
    collection can be *constructed* from arbitrary GeoJSON; structural problems
    surface when the geometry is materialized for writing rather than at model
    construction time.
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
        """Return a plain GeoJSON ``dict`` suitable for ``shapely.geometry.shape``."""
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
    """A GeoJSON Feature: a geometry plus arbitrary ``properties``."""

    type: str = Field(default="Feature")
    geometry: Optional[GeoJSONGeometry] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class FeatureCollection(BaseModel):
    """A GeoJSON FeatureCollection consumed by :func:`to_geoparquet`."""

    type: str = Field(default="FeatureCollection")
    features: List[Feature] = Field(default_factory=list)


class FormatResult(BaseModel):
    """The result of a successful format conversion.

    ``href`` is the written output location; ``fmt`` is the produced format
    (``"COG"`` or ``"GeoParquet"``) and ``driver`` the underlying writer
    (``"COG"``/``"GTiff"`` for COG, ``"Parquet"`` for GeoParquet).
    ``size_bytes`` is the size of the written output, and ``detail`` carries a
    small, format-specific summary (band/overview counts; feature/column
    counts).
    """

    href: str = Field(min_length=1)
    fmt: str = Field(min_length=1)
    driver: Optional[str] = None
    size_bytes: Optional[int] = None
    detail: Dict[str, Any] = Field(default_factory=dict)


class FormatValidity(BaseModel):
    """The result of :func:`~geo_formats.validate.validate_format`.

    ``valid`` reports whether ``href`` is a well-formed instance of ``fmt``;
    ``reason`` is a non-empty explanation populated **exactly when** ``valid``
    is ``False``. ``detail`` carries the structural facts the check observed
    (e.g. whether the GeoTIFF is internally tiled, how many overviews it has,
    or whether GeoParquet ``geo`` metadata is present).
    """

    valid: bool
    fmt: str = Field(min_length=1)
    reason: Optional[str] = None  # non-empty exactly when valid is False
    detail: Dict[str, Any] = Field(default_factory=dict)

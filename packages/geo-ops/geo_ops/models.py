"""Data models for the ``geo-ops`` server (Pillar B, MVP).

This module defines the GeoJSON-shaped data types the ``geo-ops`` tools accept
and return:

* :class:`GeoJSONGeometry` - a single GeoJSON geometry (``Point``,
  ``LineString``, ``Polygon`` and their ``Multi*`` variants, plus
  ``GeometryCollection``). It is intentionally **permissive** about its
  ``coordinates`` so that a structurally malformed or topologically invalid
  geometry can be *constructed* and then *reported on* by ``validate_geometry``
  (Requirements 8.2, 8.3) rather than rejected at model-construction time.
* :class:`Feature` / :class:`FeatureCollection` - the GeoJSON feature
  containers consumed and produced by ``spatial_join`` and ``overlay``.
* :class:`GeometryValidity` - the result of ``validate_geometry``: a ``valid``
  flag plus a ``reason`` that is populated exactly when the geometry is invalid
  (Requirement 8.3).

The models map cleanly onto Shapely via ``shapely.geometry.shape`` /
``shapely.geometry.mapping`` (see :mod:`geo_ops.geometry`), so the GEOS-backed
operations operate on standard GeoJSON dictionaries.

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
    "GeometryValidity",
]


class GeoJSONGeometry(BaseModel):
    """A single GeoJSON geometry object.

    Carries the GeoJSON ``type`` plus either ``coordinates`` (for the seven
    primitive/multi types) or ``geometries`` (for ``GeometryCollection``). The
    ``coordinates`` field is typed permissively (``Any``) on purpose: geometry
    *validity* is a topological property reported by ``validate_geometry``
    (Req 8.2/8.3), not something the model should reject up front.
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
    """A GeoJSON FeatureCollection consumed/produced by joins and overlays."""

    type: str = Field(default="FeatureCollection")
    features: List[Feature] = Field(default_factory=list)


class GeometryValidity(BaseModel):
    """The result of ``validate_geometry`` (Requirements 8.2, 8.3).

    ``valid`` reports whether the geometry is topologically valid; ``reason``
    is a non-empty explanation populated **exactly when** ``valid`` is
    ``False`` (Requirement 8.3) and ``None`` when the geometry is valid.
    """

    valid: bool
    reason: Optional[str] = None  # non-empty exactly when valid is False (Req 8.3)

"""GeoJSON data models for the ``geo-vector`` server (Pillar A, MVP).

``vector_features`` returns a :class:`FeatureCollection` (Requirement 7.3).
These models are a minimal, RFC 7946-shaped GeoJSON representation - enough to
carry features merged from OpenStreetMap (Overpass) and Overture Maps through
the MCP boundary without pulling in a heavy geometry dependency at this layer.

The geometry of each :class:`Feature` is kept as a plain GeoJSON ``geometry``
mapping (``{"type": ..., "coordinates": ...}``) rather than a typed geometry
union, so the connector can faithfully pass through whatever geometry a source
returns (Point, LineString, Polygon, ...) while remaining lossless.

Python 3.9 compatibility: this module uses ``from __future__ import
annotations`` together with ``typing`` generics so the pydantic models resolve
their annotations on 3.9+.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

__all__ = ["Feature", "FeatureCollection", "BBox"]

#: A geographic bounding box as ``(min_lon, min_lat, max_lon, max_lat)`` in
#: EPSG:4326 decimal degrees - the order used throughout the connector and by
#: GeoJSON's ``bbox`` member (RFC 7946 §5).
BBox = Tuple[float, float, float, float]


class Feature(BaseModel):
    """A single GeoJSON feature (RFC 7946 §3.2).

    ``geometry`` is an untyped GeoJSON geometry mapping so any source geometry
    type passes through unchanged; it may be ``None`` for a feature with no
    geometry. ``properties`` carries the source's tags/attributes; the
    connector adds a ``"source"`` key identifying the originating dataset
    (``"openstreetmap"`` or ``"overture"``) so merged results stay attributable.
    """

    type: str = "Feature"
    geometry: Optional[Dict[str, Any]] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    id: Optional[str] = None


class FeatureCollection(BaseModel):
    """A GeoJSON ``FeatureCollection`` (RFC 7946 §3.3).

    Returned by ``vector_features``; ``features`` holds the merged features
    from every queried source, and ``bbox`` echoes the validated request extent
    as ``(min_lon, min_lat, max_lon, max_lat)``.
    """

    type: str = "FeatureCollection"
    features: List[Feature] = Field(default_factory=list)
    bbox: Optional[BBox] = None

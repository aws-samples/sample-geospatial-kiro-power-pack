"""Data models for the ``geo-terrain`` server (Pillar A, expansion).

``elevation`` answers two shapes of request (Requirement 7.5):

* a single **location** - a :class:`Coordinate` (lon, lat) - for which it
  returns a scalar elevation in metres; and
* an **extent** - a :class:`GeoWindow` (a bounding box plus a sample grid
  size) - for which it returns a :class:`RasterArray` of sampled elevations.

These models are intentionally minimal and ``geo-common``-compatible, matching
the convention used by the other Pillar A connectors (e.g. ``geo-vector``'s
``BBox``/``FeatureCollection``): just enough structure to carry an elevation
query and its result across the MCP boundary without pulling in a heavy raster
dependency at this layer.

Python 3.9 compatibility: this module uses ``from __future__ import
annotations`` together with ``typing`` generics so the pydantic models resolve
their annotations on 3.9+.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

__all__ = ["BBox", "Coordinate", "GeoWindow", "RasterArray"]

#: A geographic bounding box as ``(min_lon, min_lat, max_lon, max_lat)`` in
#: EPSG:4326 decimal degrees - the order used throughout the connector and by
#: GeoJSON's ``bbox`` member (RFC 7946 §5).
BBox = Tuple[float, float, float, float]


class Coordinate(BaseModel):
    """A single geographic location in EPSG:4326 decimal degrees.

    ``elevation`` returns a scalar elevation (metres) for a ``Coordinate``
    (Requirement 7.5). Longitude must lie in ``[-180, 180]`` and latitude in
    ``[-90, 90]``; :func:`geo_terrain.elevation.parse_location` enforces this
    and raises an ``Error_Taxonomy`` validation error otherwise (Requirement
    7.12).
    """

    lon: float
    lat: float


class GeoWindow(BaseModel):
    """A geographic extent plus the sample grid to resolve over it.

    ``elevation`` returns a :class:`RasterArray` for a ``GeoWindow`` by sampling
    a ``width`` × ``height`` grid of points spanning ``bbox`` (Requirement 7.5).
    ``bbox`` is ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326 decimal
    degrees. The grid dimensions are validated and capped at a configurable
    maximum total sample count (Requirement 7.12).
    """

    bbox: BBox
    width: int = 16
    height: int = 16


class RasterArray(BaseModel):
    """A grid of sampled elevations returned for a :class:`GeoWindow`.

    ``values`` is a ``height`` × ``width`` row-major grid of elevations in
    ``units`` (metres). Rows run north→south (``values[0]`` is the northernmost
    row) and columns west→east, matching common raster orientation. A sample
    with no data carries ``None`` so missing coverage is explicit rather than
    silently zero. ``bbox`` echoes the requested extent and ``source`` names
    the terrain dataset the values came from.
    """

    values: List[List[Optional[float]]] = Field(default_factory=list)
    width: int
    height: int
    bbox: BBox
    source: str
    crs: str = "EPSG:4326"
    units: str = "m"

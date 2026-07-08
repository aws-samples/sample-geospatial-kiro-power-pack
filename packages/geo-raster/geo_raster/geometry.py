"""Backwards-compatible re-export of the shared planar-geometry helpers.

The point-in-polygon / bbox helpers now live in :mod:`geo_common.geometry` so
any server (not just geo-raster) can use them without a cross-server
dependency. Re-exported here so existing ``from geo_raster.geometry import ...``
imports keep working.
"""

from __future__ import annotations

from geo_common.geometry import (
    AREAL_GEOMETRY_TYPES,
    geometry_bbox,
    iter_polygons,
    point_in_geometry,
    point_in_polygon,
    polygon_bbox,
)

__all__ = [
    "AREAL_GEOMETRY_TYPES",
    "polygon_bbox",
    "geometry_bbox",
    "point_in_polygon",
    "point_in_geometry",
    "iter_polygons",
]

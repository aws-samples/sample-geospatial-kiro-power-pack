"""geo-ops: Pillar B (MVP) processing MCP server.

Provides local, pure-function geospatial processing built on Shapely/GEOS and
GeoPandas. Task 4.3 delivers the geometry operations - ``validate_geometry``
(validity + reason), ``spatial_join`` (attribute join by spatial predicate),
and ``overlay`` (geometric set operations) - plus the GeoJSON data models they
share. The ``transform_crs`` CRS-reprojection tool (task 4.1) is added to the
same server when its module lands.
"""

from __future__ import annotations

from geo_ops.geometry import (
    OVERLAY_OPS,
    SPATIAL_PREDICATES,
    buffer,
    convex_hull,
    overlay,
    spatial_join,
    validate_geometry,
)
from geo_ops.transform import CRS_ROUNDTRIP_TOLERANCE, transform_crs
from geo_ops.models import (
    Feature,
    FeatureCollection,
    GeoJSONGeometry,
    GeometryValidity,
)
from geo_ops.server import GeoOpsServer

__all__ = [
    # Data models
    "GeoJSONGeometry",
    "Feature",
    "FeatureCollection",
    "GeometryValidity",
    # CRS transform (task 4.1)
    "transform_crs",
    "CRS_ROUNDTRIP_TOLERANCE",
    # Geometry operations (task 4.3)
    "validate_geometry",
    "spatial_join",
    "overlay",
    "buffer",
    "convex_hull",
    "SPATIAL_PREDICATES",
    "OVERLAY_OPS",
    # Server
    "GeoOpsServer",
]

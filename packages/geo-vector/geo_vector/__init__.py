"""geo-vector: Pillar A (MVP) vector-feature MCP server.

Exposes ``vector_features``, which returns features from OpenStreetMap
(Overpass) and Overture Maps for a bounding box whose area does not exceed a
configurable maximum (default 2,500 km²), and rejects malformed or oversized
extents with an ``Error_Taxonomy`` validation error (Requirements 7.3, 7.12).

Task 6.1 delivers ``vector_features``, the source connectors, the GeoJSON
``FeatureCollection`` model, and the server scaffold; task 6.2 adds the
Resource Catalog entry, credential specs, and the availability-error
identification for unreachable sources.
"""

from __future__ import annotations

from geo_vector.bbox import (
    DEFAULT_MAX_AREA_KM2,
    EARTH_RADIUS_KM,
    bbox_area_km2,
    validate_bbox,
)
from geo_vector.features import (
    OverpassSource,
    OvertureSource,
    VectorSource,
    default_sources,
    vector_features,
)
from geo_vector.models import BBox, Feature, FeatureCollection
from geo_vector.server import GeoVectorServer

__all__ = [
    # Data models
    "BBox",
    "Feature",
    "FeatureCollection",
    # bbox validation + area
    "EARTH_RADIUS_KM",
    "DEFAULT_MAX_AREA_KM2",
    "validate_bbox",
    "bbox_area_km2",
    # Sources + connector
    "VectorSource",
    "OverpassSource",
    "OvertureSource",
    "default_sources",
    "vector_features",
    # Server
    "GeoVectorServer",
]

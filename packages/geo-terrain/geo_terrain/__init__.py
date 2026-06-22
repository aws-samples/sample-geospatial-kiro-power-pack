"""geo-terrain: Pillar A (expansion) elevation MCP server.

Exposes ``elevation`` (plus ``slope`` and ``hillshade``), which return values
from a configured terrain source - SRTM (default) or USGS 3DEP - for a single
location (a scalar in metres) or an extent (a sampled elevation grid), within
30 seconds, and reject malformed parameters with an ``Error_Taxonomy``
validation error (Requirements 7.5, 7.12).

Task 13.3 delivers ``elevation``, the terrain-source connectors, the
``Coordinate``/``GeoWindow``/``RasterArray`` models, and the server scaffold;
task 13.6 adds the Resource Catalog registration and availability-error
identification for unreachable sources.
"""

from __future__ import annotations

from geo_terrain.elevation import (
    DEFAULT_MAX_SAMPLES,
    DEFAULT_SOURCE,
    DEFAULT_TERRAIN_API_URL,
    TerrainSource,
    default_sources,
    elevation,
    parse_location,
)
from geo_terrain.models import BBox, Coordinate, GeoWindow, RasterArray
from geo_terrain.server import GeoTerrainServer, INSTALL_COMMAND

__all__ = [
    # Data models
    "BBox",
    "Coordinate",
    "GeoWindow",
    "RasterArray",
    # Sources + connector
    "DEFAULT_TERRAIN_API_URL",
    "DEFAULT_SOURCE",
    "DEFAULT_MAX_SAMPLES",
    "TerrainSource",
    "default_sources",
    "parse_location",
    "elevation",
    # Server
    "GeoTerrainServer",
    "INSTALL_COMMAND",
]

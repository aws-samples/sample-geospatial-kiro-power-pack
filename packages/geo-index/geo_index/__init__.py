"""geo-index: Pillar B (expansion) spatial-indexing MCP server.

Provides local, pure-function spatial indexing built on the H3 and S2
libraries. Task 14.9 delivers ``index_cell`` - the H3 (res 0-15) / S2
(level 0-30) cell lookup for a coordinate at a requested resolution
(Requirement 8.9), with taxonomy ``ValidationError`` for out-of-range
resolution or out-of-bounds coordinates (Requirement 8.10).
"""

from __future__ import annotations

from geo_index.indexing import (
    H3,
    H3_MAX_RESOLUTION,
    H3_MIN_RESOLUTION,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    S2,
    S2_MAX_LEVEL,
    S2_MIN_LEVEL,
    SCHEMES,
    index_cell,
    resolution_range,
)
from geo_index.server import GeoIndexServer

__all__ = [
    # Cell lookup (task 14.9)
    "index_cell",
    "resolution_range",
    # Scheme + range constants
    "H3",
    "S2",
    "SCHEMES",
    "H3_MIN_RESOLUTION",
    "H3_MAX_RESOLUTION",
    "S2_MIN_LEVEL",
    "S2_MAX_LEVEL",
    "LON_MIN",
    "LON_MAX",
    "LAT_MIN",
    "LAT_MAX",
    # Server
    "GeoIndexServer",
]

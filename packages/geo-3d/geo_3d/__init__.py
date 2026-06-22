"""geo-3d: Pillar B MCP server for 3D geospatial formats.

Direction 3 (format inspection/validation) ships first: ``inspect_tileset``
validates an OGC 3D Tiles ``tileset.json`` and summarizes its tile tree, and
``inspect_gltf`` parses a glTF 2.0 / GLB asset and summarizes it. Both are
local, dependency-free inspections; a parseable-but-invalid document yields a
report with ``valid=False`` rather than raising.
"""

from __future__ import annotations

from geo_3d.dem import MAX_DEM_CELLS, dem_to_mesh
from geo_3d.inspect import (
    MAX_CONTENT_URIS,
    MAX_TILES,
    inspect_gltf,
    inspect_tileset,
)
from geo_3d.models import GltfReport, MeshResult, PointTilesetResult, TilesetReport
from geo_3d.server import Geo3DServer, INSTALL_COMMAND, main
from geo_3d.tiling import MAX_POINTS, points_to_3d_tiles

__all__ = [
    "TilesetReport",
    "GltfReport",
    "PointTilesetResult",
    "MeshResult",
    "inspect_tileset",
    "inspect_gltf",
    "points_to_3d_tiles",
    "dem_to_mesh",
    "MAX_CONTENT_URIS",
    "MAX_TILES",
    "MAX_POINTS",
    "MAX_DEM_CELLS",
    "Geo3DServer",
    "INSTALL_COMMAND",
    "main",
]

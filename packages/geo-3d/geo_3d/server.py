"""The ``geo-3d`` MCP server (Pillar B).

``geo-3d`` works with 3D geospatial formats. Its first capabilities *inspect and
validate* the two most common ones - OGC **3D Tiles** tilesets and **glTF/GLB**
assets:

* ``inspect_tileset`` - structurally validate a ``tileset.json`` and summarize
  its tile tree (asset version, root geometric error / bounding volume, tile
  count and depth, content URIs); and
* ``inspect_gltf`` - parse a glTF 2.0 / GLB asset and summarize it (binary vs
  JSON, glTF version, generator, and per-array counts).

Both are pure, local, dependency-free inspections (no native libraries), so the
server is open and credential-free. A *parseable-but-invalid* document yields a
report with ``valid=False`` and ``issues`` rather than an error; a malformed
*request* (bad/missing input) raises an ``Error_Taxonomy`` ``ValidationError``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from geo_common.http import HttpClient
from geo_common.models import CatalogEntry, CredentialSpec, OpennessTier
from geo_common.server import BaseGeoServer

from geo_3d.inspect import inspect_gltf as _inspect_gltf
from geo_3d.inspect import inspect_tileset as _inspect_tileset
from geo_3d.dem import dem_to_mesh as _dem_to_mesh
from geo_3d.models import GltfReport, MeshResult, PointTilesetResult, TilesetReport
from geo_3d.tiling import points_to_3d_tiles as _points_to_3d_tiles

__all__ = ["Geo3DServer", "INSTALL_COMMAND", "main"]

#: The ``uvx`` command that installs this server (Req 2.6; bundle-manifest).
INSTALL_COMMAND = "uvx geo-3d"


class Geo3DServer(BaseGeoServer):
    """Pillar B server exposing 3D-format inspection/validation.

    Registers ``inspect_tileset`` and ``inspect_gltf``. All inspection is local
    and dependency-free, so ``geo-3d`` declares no ``mcp.json`` credentials and
    always starts.
    """

    pillar = "B"
    server_name = "geo-3d"
    version = "0.2.0"

    #: Native capabilities (name, description) for the Resource Catalog (Req 2.1).
    _CAPABILITIES = (
        (
            "inspect_tileset",
            "Validate an OGC 3D Tiles tileset.json and summarize its tile tree "
            "(asset version, root geometric error/bounding volume, tile count "
            "and depth, content URIs).",
        ),
        (
            "inspect_gltf",
            "Parse a glTF 2.0 / GLB asset and summarize it (binary vs JSON, "
            "glTF version, generator, and per-array element counts).",
        ),
        (
            "points_to_3d_tiles",
            "Tile a point cloud into an OGC 3D Tiles tileset: write a "
            "tileset.json plus a binary .pnts point-cloud tile (single root "
            "tile; pairs with geo-pointcloud).",
        ),
        (
            "dem_to_mesh",
            "Mesh a DEM elevation grid into a glTF 2.0 / GLB terrain surface "
            "(two triangles per cell, elevations as vertex heights; pairs with "
            "geo-terrain).",
        ),
    )

    def __init__(self, http: Optional[HttpClient] = None) -> None:
        super().__init__(http=http)
        self.register_tool("inspect_tileset", self.inspect_tileset)
        self.register_tool("inspect_gltf", self.inspect_gltf)
        self.register_tool("points_to_3d_tiles", self.points_to_3d_tiles)
        self.register_tool("dem_to_mesh", self.dem_to_mesh)

    async def inspect_tileset(
        self,
        *,
        tileset: Optional[Any] = None,
        source: Optional[str] = None,
    ) -> TilesetReport:
        """Validate/summarize an OGC 3D Tiles ``tileset.json``.

        Provide the tileset inline (``tileset``, a JSON object **or** a JSON
        string) **or** via ``source`` (a local file path or ``https`` URL);
        exactly one is required, else an ``Error_Taxonomy`` ``ValidationError``
        is raised. A document that parses but is not a structurally valid
        tileset returns a :class:`~geo_3d.models.TilesetReport` with
        ``valid=False`` and explanatory ``issues``.
        """
        return await _inspect_tileset(http=self.http, tileset=tileset, source=source)

    async def inspect_gltf(self, *, source: str) -> GltfReport:
        """Parse and summarize a glTF 2.0 / GLB asset at ``source``.

        ``source`` is a local file path or ``https`` URL to a ``.gltf`` (JSON)
        or ``.glb`` (binary) asset. A blank source raises a ``ValidationError``;
        a file that reads but is not valid glTF/GLB returns a
        :class:`~geo_3d.models.GltfReport` with ``valid=False`` and ``issues``.
        """
        return await _inspect_gltf(http=self.http, source=source)

    async def points_to_3d_tiles(
        self,
        *,
        points: Any,
        output_dir: str,
        colors: Optional[Any] = None,
        geometric_error: Optional[float] = None,
        content_name: str = "points.pnts",
        max_points_per_tile: Optional[int] = None,
    ) -> PointTilesetResult:
        """Tile a point cloud into an OGC 3D Tiles tileset under ``output_dir``.

        ``points`` is a list of ``[x, y, z]`` triples (or ``{x, y, z}``
        mappings) - e.g. read via ``geo-pointcloud``; ``colors`` optionally
        gives a matching ``[r, g, b]`` (0-255) per point. By default all points
        go in a single root tile; pass ``max_points_per_tile`` to build an
        **octree** level-of-detail hierarchy (multiple ``.pnts`` tiles + a
        nested tileset) so large clouds stream coarse-to-fine. Writes
        ``output_dir/tileset.json`` and the ``.pnts`` content file(s) and
        returns a :class:`~geo_3d.models.PointTilesetResult` (with
        ``tile_count``/``max_depth``). A malformed/empty point list, a
        ``colors`` length mismatch, a bad ``max_points_per_tile``, or a
        non-writable ``output_dir`` raises an ``Error_Taxonomy``
        ``ValidationError`` before anything is written. The output validates
        with ``inspect_tileset``.
        """
        return _points_to_3d_tiles(
            points=points,
            output_dir=output_dir,
            colors=colors,
            geometric_error=geometric_error,
            content_name=content_name,
            max_points_per_tile=max_points_per_tile,
        )

    async def dem_to_mesh(
        self,
        *,
        elevations: Any,
        bbox: Any,
        output_path: str,
        vertical_exaggeration: float = 1.0,
    ) -> MeshResult:
        """Mesh a DEM ``elevations`` grid over ``bbox`` into a glTF/GLB surface.

        ``elevations`` is a rectangular 2D grid (rows north→south, columns
        west→east) of numbers with ``None`` for nodata - e.g. a ``geo-terrain``
        ``elevation`` extent result's ``values``. ``bbox`` is
        ``(west, south, east, north)`` in EPSG:4326; ``output_path`` ends in
        ``.glb`` (binary) or ``.gltf``; ``vertical_exaggeration`` (>0) scales
        elevation. Nodata cells leave holes. Returns a
        :class:`~geo_3d.models.MeshResult`. A malformed grid/bbox/exaggeration
        or a non-writable output path raises an ``Error_Taxonomy``
        ``ValidationError`` before anything is written. The output validates
        with ``inspect_gltf``.
        """
        return _dem_to_mesh(
            elevations=elevations,
            bbox=bbox,
            output_path=output_path,
            vertical_exaggeration=vertical_exaggeration,
        )

    def catalog_entries(self) -> "List[CatalogEntry]":
        """Register ``geo-3d``'s capabilities in the Resource Catalog (Req 2.1)."""
        return [
            CatalogEntry(
                name=name,
                pillar=self.pillar,
                capability_description=description,
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                install_command=INSTALL_COMMAND,
            )
            for name, description in self._CAPABILITIES
        ]

    def required_credentials(self) -> "List[CredentialSpec]":
        """``geo-3d`` is open and credential-free, so this is empty (Req 16.5)."""
        return []


def main() -> None:
    """Console entry point: serve geo-3d over MCP stdio."""
    Geo3DServer().run()


if __name__ == "__main__":  # pragma: no cover - module CLI shim
    main()

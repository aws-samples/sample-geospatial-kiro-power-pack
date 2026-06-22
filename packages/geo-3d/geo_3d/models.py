"""Data models for the ``geo-3d`` server (Pillar B).

``geo-3d`` works with 3D geospatial formats. Its first capabilities *inspect and
validate* the two most common ones - OGC **3D Tiles** tilesets and **glTF/GLB**
assets - and return a structured report. Both report types carry a ``valid``
flag plus an ``issues`` list, so a malformed-but-parseable document yields a
clean negative report rather than an exception.

Python 3.10+: ``from __future__ import annotations`` with ``typing`` generics.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

__all__ = ["TilesetReport", "GltfReport", "PointTilesetResult", "MeshResult"]


class TilesetReport(BaseModel):
    """Structural summary of an OGC 3D Tiles ``tileset.json`` (3D Tiles 1.x).

    ``valid`` is ``True`` when the required structure is present (an ``asset``
    with a ``version``, a ``root`` tile, and a numeric root ``geometricError``);
    otherwise ``issues`` explains what is missing or malformed. The remaining
    fields summarize the tile tree: the root bounding volume, the total tile
    count and maximum depth, and the content URIs referenced (capped).
    """

    valid: bool
    asset_version: Optional[str] = None
    geometric_error: Optional[float] = None
    refine: Optional[str] = None
    root_bounding_volume: Optional[Dict[str, object]] = None
    tile_count: int = Field(default=0, ge=0)
    max_depth: int = Field(default=0, ge=0)
    content_count: int = Field(default=0, ge=0)
    content_uris: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)


class GltfReport(BaseModel):
    """Structural summary of a glTF 2.0 / GLB asset.

    ``binary`` distinguishes a ``.glb`` (binary container) from a ``.gltf``
    (JSON). ``valid`` reflects that the document parsed and carries an
    ``asset.version``; ``issues`` lists any structural problems. ``counts``
    holds the size of the major glTF arrays (meshes, nodes, materials, …).
    """

    valid: bool
    binary: bool
    gltf_version: Optional[str] = None
    generator: Optional[str] = None
    byte_length: Optional[int] = None
    counts: Dict[str, int] = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list)


class PointTilesetResult(BaseModel):
    """The output of tiling a point cloud into an OGC 3D Tiles tileset.

    Records where the files were written (``tileset_path`` and the binary
    ``content_path`` ``.pnts`` tile), the ``point_count`` tiled, the root
    ``bounding_volume`` (a 3D Tiles ``box``) and ``geometric_error``, whether
    per-point ``colors`` were embedded, and the ``.pnts`` ``byte_length``.
    """

    tileset_path: str
    content_path: str
    point_count: int = Field(ge=0)
    bounding_volume: Dict[str, object] = Field(default_factory=dict)
    geometric_error: float
    has_colors: bool = False
    byte_length: int = Field(ge=0)
    #: Number of tiles written (1 for a single root tile; >1 for an octree).
    tile_count: int = Field(default=1, ge=1)
    #: Maximum octree depth (0 for a single root tile).
    max_depth: int = Field(default=0, ge=0)


class MeshResult(BaseModel):
    """The output of meshing a DEM grid into a glTF/GLB surface.

    Records where the mesh was written (``output_path``) and its ``format``
    (``glb`` or ``gltf``), the ``vertex_count`` / ``triangle_count``, the file
    ``byte_length``, the mesh's local-space bounding box (``min_xyz`` / ``max_xyz``,
    in metres with elevation on the Y axis), and ``has_holes`` (``True`` when
    some triangles were dropped because the DEM had nodata cells there).
    """

    output_path: str
    format: str
    vertex_count: int = Field(ge=0)
    triangle_count: int = Field(ge=0)
    byte_length: int = Field(ge=0)
    min_xyz: List[float] = Field(default_factory=list)
    max_xyz: List[float] = Field(default_factory=list)
    has_holes: bool = False

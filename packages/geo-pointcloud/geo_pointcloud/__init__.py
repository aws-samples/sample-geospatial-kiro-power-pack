"""geo-pointcloud: Pillar B (expansion) point-cloud MCP server.

Reads and writes point-cloud data (Requirement 8.11): ``read_pointcloud`` and
``write_pointcloud`` plus the point-cloud data models and the pluggable
read/write backend; the format round-trip property (design Property 20) is
covered by the backend round-trip tests.

The read/write engine is pluggable and the default :class:`SmartCopcBackend`
routes per operation: it reads a real ``.copc.laz`` / ``.laz`` (local or remote)
via :class:`LaspyCopcBackend` (the light ``[copc]`` extra, COPC octree windowed
reads) or :class:`PdalCopcBackend` (the ``[pdal]`` extra), and writes
standards-compliant COPC via PDAL when available. With no extra installed it
falls back to :class:`LocalContainerBackend` - a portable, lossless local
container that is explicitly **not** interoperable COPC (labelled
``GEO-POINTCLOUD-LOCAL``), so nothing is mislabelled as COPC.
"""

from __future__ import annotations

from geo_pointcloud.models import (
    OPTIONAL_DIMENSIONS,
    POSITIONAL_DIMENSIONS,
    FormatResult,
    GeoWindow,
    PointCloudChunk,
    PointRecord,
)
from geo_pointcloud.pointcloud import (
    COPC_FORMAT,
    LAZ_FORMAT,
    LOCAL_CONTAINER_FORMAT,
    LaspyCopcBackend,
    LocalContainerBackend,
    LocalCopcBackend,
    PdalCopcBackend,
    PointCloudBackend,
    SmartCopcBackend,
    default_backend,
    read_pointcloud,
    write_pointcloud,
)
from geo_pointcloud.server import GeoPointcloudServer, INSTALL_COMMAND

__all__ = [
    # Data models
    "PointRecord",
    "PointCloudChunk",
    "GeoWindow",
    "FormatResult",
    "POSITIONAL_DIMENSIONS",
    "OPTIONAL_DIMENSIONS",
    # Read/write core
    "COPC_FORMAT",
    "LAZ_FORMAT",
    "LOCAL_CONTAINER_FORMAT",
    "PointCloudBackend",
    "LocalContainerBackend",
    "LocalCopcBackend",
    "LaspyCopcBackend",
    "PdalCopcBackend",
    "SmartCopcBackend",
    "default_backend",
    "read_pointcloud",
    "write_pointcloud",
    # Server
    "GeoPointcloudServer",
    "INSTALL_COMMAND",
]

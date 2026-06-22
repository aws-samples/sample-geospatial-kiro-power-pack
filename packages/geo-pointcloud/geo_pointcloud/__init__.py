"""geo-pointcloud: Pillar B (expansion) point-cloud MCP server.

Reads and writes point-cloud data in Cloud-Optimized Point Cloud (COPC) format
(Requirement 8.11). Task 14.7 delivers ``read_pointcloud`` and
``write_pointcloud`` plus the point-cloud data models and the pluggable
read/write backend; the format round-trip property (design Property 20) is
covered by task 14.8.

The read/write engine is pluggable: the default :class:`LocalCopcBackend` is a
portable, lossless COPC-style container that preserves the point set without
the native PDAL stack, while :class:`PdalCopcBackend` reads/writes standard
``.copc.laz`` via PDAL in production.
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
    LocalCopcBackend,
    PdalCopcBackend,
    PointCloudBackend,
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
    "PointCloudBackend",
    "LocalCopcBackend",
    "PdalCopcBackend",
    "default_backend",
    "read_pointcloud",
    "write_pointcloud",
    # Server
    "GeoPointcloudServer",
    "INSTALL_COMMAND",
]

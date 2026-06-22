"""geo-raster: the Pillar B (expansion) raster-analytics MCP server.

``geo-raster`` computes **zonal statistics** — per-zone minimum, maximum, mean,
sum, and count of a raster over a set of vector zones (Requirement 8.7) — reading
only the overlapping window directly from S3/HTTP byte ranges and never copying
the full asset to local storage. A zone with no overlapping cells gets a no-data
indication while the remaining zones still receive statistics (Requirement 8.8),
and an S3 read failure after the shared client's retry attempts within the
30-second window aborts the operation with local storage unchanged (Requirement
12.6).

It follows the Geospatial Power Pack conventions: it uses ``geo-common``'s single
async :class:`~geo_common.http.HttpClient` for outbound requests and the shared
``Error_Taxonomy`` for failures, and depends on no other server package.

Public surface (design.md "Pillar B — Processing and Compute"):

* :class:`RasterGrid` — a single-band georeferenced block of raster cells.
* :class:`FeatureCollection` / :class:`Feature` / :class:`GeoJSONGeometry` — the
  GeoJSON vector zones a raster is summarized over.
* :class:`ZoneStat` — the per-zone statistics result (or no-data indication).
* :func:`zonal_statistics` — the public capability (reads + computes).
* :func:`compute_zonal_statistics` — the pure, reference-grade computation over
  an in-memory grid.
* :class:`GeoRasterServer` — the :class:`~geo_common.server.BaseGeoServer`
  subclass that owns the shared HTTP client and registers the MCP tool.
"""

from __future__ import annotations

from geo_raster.cog import ByteRangeReader, CogMetadata, CogReader
from geo_raster.models import (
    DEFAULT_STATISTICS,
    SUPPORTED_STATISTICS,
    Feature,
    FeatureCollection,
    GeoJSONGeometry,
    RasterGrid,
    ZoneStat,
)
from geo_raster.reader import DEFAULT_S3_REGION, HttpRangeReader, read_raster_grid
from geo_raster.server import GeoRasterServer, INSTALL_COMMAND, main
from geo_raster.window_models import GeoWindow, PixelWindow, RasterArray
from geo_raster.window_reader import band_math, read_window, resolve_pixel_window
from geo_raster.zonal import compute_zonal_statistics, zonal_statistics

__all__ = [
    "RasterGrid",
    "Feature",
    "FeatureCollection",
    "GeoJSONGeometry",
    "ZoneStat",
    "SUPPORTED_STATISTICS",
    "DEFAULT_STATISTICS",
    "zonal_statistics",
    "compute_zonal_statistics",
    "read_raster_grid",
    "HttpRangeReader",
    "ByteRangeReader",
    "CogReader",
    "CogMetadata",
    "DEFAULT_S3_REGION",
    # Windowed COG read + band math
    "PixelWindow",
    "GeoWindow",
    "RasterArray",
    "read_window",
    "band_math",
    "resolve_pixel_window",
    "GeoRasterServer",
    "INSTALL_COMMAND",
    "main",
]

"""geo-formats: the Pillar B (expansion) format-conversion MCP server.

``geo-formats`` converts geospatial data into cloud-optimized formats and
validates the results (design.md "Pillar B — Processing and Compute"):

* :func:`to_cog` — convert a raster to a Cloud-Optimized GeoTIFF, internally
  tiled and with overviews, preserving pixels exactly including nodata
  (Requirements 8.4, 12.2; design Property 2).
* :func:`to_geoparquet` — convert a vector dataset to GeoParquet, preserving
  feature count, geometries coordinate-for-coordinate, and every attribute
  value (Requirements 8.5, 12.3; design Property 3).
* :func:`validate_format` — check that an output is a well-formed COG or
  GeoParquet.

On a write failure the conversion tools abort, remove any partial output, and
return a ``geo_common`` ``Error_Taxonomy`` error (Requirement 12.7). The server
inherits the shared :class:`~geo_common.server.BaseGeoServer` contract; all
conversion is local and credential-free.
"""

from __future__ import annotations

from geo_formats.cog import to_cog
from geo_formats.geoparquet import to_geoparquet
from geo_formats.models import (
    Feature,
    FeatureCollection,
    FormatResult,
    FormatValidity,
    GeoJSONGeometry,
)
from geo_formats.server import GeoFormatsServer, INSTALL_COMMAND, main
from geo_formats.validate import validate_format

__all__ = [
    # Data models
    "GeoJSONGeometry",
    "Feature",
    "FeatureCollection",
    "FormatResult",
    "FormatValidity",
    # Conversion + validation tools
    "to_cog",
    "to_geoparquet",
    "validate_format",
    # Server
    "GeoFormatsServer",
    "INSTALL_COMMAND",
    "main",
]

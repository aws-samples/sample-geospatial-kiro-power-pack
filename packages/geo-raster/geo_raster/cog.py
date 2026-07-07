"""Backwards-compatible re-export of the shared COG reader.

The byte-range Cloud-Optimized GeoTIFF reader now lives in
:mod:`geo_common.cog` so every server can consume it without depending on
another server package (see that module for the implementation and supported
COG subset). This module re-exports the public names so existing
``from geo_raster.cog import ...`` imports keep working.
"""

from __future__ import annotations

from geo_common.cog import (
    DEFAULT_S3_REGION,
    ByteRangeReader,
    CogMetadata,
    CogReader,
    HttpRangeReader,
)

__all__ = [
    "ByteRangeReader",
    "CogMetadata",
    "CogReader",
    "HttpRangeReader",
    "DEFAULT_S3_REGION",
]

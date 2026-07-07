"""Backwards-compatible re-export of the shared raster window reader.

``read_raster_grid`` now lives in :mod:`geo_common.raster_read` so any server can
read a raster window without depending on geo-raster; ``HttpRangeReader`` and
``DEFAULT_S3_REGION`` live in :mod:`geo_common.cog`. Re-exported here so existing
``from geo_raster.reader import ...`` imports keep working.
"""

from __future__ import annotations

from geo_common.cog import DEFAULT_S3_REGION, HttpRangeReader
from geo_common.raster_read import read_raster_grid

__all__ = ["HttpRangeReader", "read_raster_grid", "DEFAULT_S3_REGION"]

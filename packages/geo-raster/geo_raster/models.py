"""Backwards-compatible re-export of the shared raster/zone models.

``RasterGrid`` and the GeoJSON zone models (plus the zonal-statistics constants)
now live in :mod:`geo_common.raster_models` so the shared zonal reducer and any
server can use them without a cross-server dependency. Re-exported here so
existing ``from geo_raster.models import ...`` imports keep working.
"""

from __future__ import annotations

from geo_common.raster_models import (
    DEFAULT_STATISTICS,
    SUPPORTED_STATISTICS,
    Feature,
    FeatureCollection,
    GeoJSONGeometry,
    RasterGrid,
    ZoneStat,
)

__all__ = [
    "GeoJSONGeometry",
    "Feature",
    "FeatureCollection",
    "RasterGrid",
    "ZoneStat",
    "SUPPORTED_STATISTICS",
    "DEFAULT_STATISTICS",
]

"""geo-raster's ``zonal_statistics`` tool: read a raster window, then reduce.

The **pure** reducer (:func:`compute_zonal_statistics` and its helpers) now lives
in :mod:`geo_common.zonal` so other servers can reuse it without depending on
geo-raster. This module keeps the async ``zonal_statistics`` tool wrapper — read
only the tiles overlapping the union of the zones by byte range, then reduce —
and re-exports the pure reducer names so existing
``from geo_raster.zonal import ...`` imports keep working.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from geo_common.cog import DEFAULT_S3_REGION, ByteRangeReader
from geo_common.http import HttpClient
from geo_common.raster_models import FeatureCollection, ZoneStat
from geo_common.zonal import (
    _validate_stats,
    _zones_bbox,
    compute_zonal_statistics,
)

from geo_raster.reader import read_raster_grid

__all__ = ["compute_zonal_statistics", "zonal_statistics"]


async def zonal_statistics(
    *,
    raster_href: str,
    zones: FeatureCollection,
    stats: Optional[Sequence[str]] = None,
    band: int = 1,
    http: Optional[HttpClient] = None,
    region: str = DEFAULT_S3_REGION,
    reader: Optional[ByteRangeReader] = None,
) -> List[ZoneStat]:
    """Per-zone raster statistics (Requirements 8.7, 8.8, 12.6).

    Validates the requested statistics, reads only the raster window overlapping
    the union of the zones directly from S3 byte ranges, and computes each
    zone's ``min``/``max``/``mean``/``sum``/``count``/``std`` (population standard
    deviation) over the cells whose center falls inside it. Zones with no
    overlapping cells receive a no-data indication while the others still
    receive statistics (Requirement 8.8). On an S3 read failure after the shared
    client's retry attempts within the 30-second window, the operation aborts
    with local storage unchanged and a ``network`` Error_Taxonomy error
    indicating the S3 read failure (Requirement 12.6).
    """
    requested = _validate_stats(
        stats, source=raster_href if isinstance(raster_href, str) else "geo-raster"
    )

    if not zones.features:
        # No zones -> nothing to summarize; no read is issued.
        return []

    # Validate every zone geometry up front and compute the read window.
    window_bbox = _zones_bbox(zones, source="geo-raster")

    grid = await read_raster_grid(
        raster_href=raster_href,
        window_bbox=window_bbox,
        band=band,
        http=http,
        region=region,
        reader=reader,
    )
    return compute_zonal_statistics(grid, zones, requested, source="geo-raster")

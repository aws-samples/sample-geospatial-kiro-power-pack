"""Read a georeferenced raster window into a :class:`RasterGrid` (Req 8.7, 12.6).

This module bridges the COG byte-range engine (:mod:`geo_raster.cog`) and the
zonal-statistics core (:mod:`geo_raster.zonal`):

* :class:`HttpRangeReader` — the production
  :class:`~geo_raster.cog.ByteRangeReader` that issues HTTP ``Range`` requests
  through the shared :class:`~geo_common.http.HttpClient` (so reads inherit the
  Power Pack's retry/backoff and the 30-second per-request timeout) and resolves
  ``s3://`` URIs to their HTTPS virtual-hosted endpoint.
* :func:`read_raster_grid` — open a COG, resolve a georeferenced window to
  pixels via the asset's geotransform, read **only** the overlapping tiles, and
  return a single-band :class:`RasterGrid` carrying the window's cells plus the
  window-local geotransform.

S3 read-failure handling (Requirement 12.6): the shared ``HttpClient`` retries a
failing S3/HTTP read up to its configured maximum (default 3 attempts) with
exponential backoff capped so the whole sequence stays within a 30-second
window, then surfaces an ``Error_Taxonomy`` ``network`` error. When that
happens, :func:`read_raster_grid` aborts and re-raises a
:class:`~geo_common.errors.NetworkError` that explicitly indicates the S3 read
failure. Because the read path **never writes to local storage**, aborting
leaves local storage unchanged.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
from urllib.parse import urlsplit

from geo_common.cog import (
    DEFAULT_S3_REGION,
    ByteRangeReader,
    CogMetadata,
    CogReader,
    HttpRangeReader,
    resolve_window as _resolve_window,
    window_geotransform as _window_geotransform,
)
from geo_common.errors import NetworkError, ValidationError
from geo_common.http import HttpClient
from geo_common.raster_models import RasterGrid

# ``HttpRangeReader`` and ``DEFAULT_S3_REGION`` live in ``geo_common.cog``;
# re-exported so importers can get the whole read surface from one place.
__all__ = [
    "HttpRangeReader",
    "read_raster_grid",
    "DEFAULT_S3_REGION",
]

BBox = Tuple[float, float, float, float]


async def read_raster_grid(
    *,
    raster_href: str,
    window_bbox: Optional[BBox] = None,
    band: int = 1,
    http: Optional[HttpClient] = None,
    region: str = DEFAULT_S3_REGION,
    reader: Optional[ByteRangeReader] = None,
) -> RasterGrid:
    """Read a single-band georeferenced window of a raster into a grid.

    Parameters
    ----------
    raster_href:
        ``s3://<bucket>/<key>`` or ``http(s)://...`` location of the raster asset.
    window_bbox:
        Optional ``(min_x, min_y, max_x, max_y)`` in the asset's CRS; only the
        cells overlapping this box are read. When omitted the whole asset is
        read.
    band:
        1-based band index to summarize (default 1).
    http / region / reader:
        Shared client / S3 region / an explicit pre-built byte-range reader
        (used by tests with an in-memory asset).

    Raises
    ------
    ValidationError
        For a malformed href, an out-of-range band, or an asset lacking the
        georeferencing required to resolve a ``window_bbox``.
    NetworkError
        When the asset cannot be read from S3 after the shared client's retry
        attempts within the 30-second window (Requirement 12.6); local storage
        is left unchanged.
    """
    href = _validate_href(raster_href)
    if isinstance(band, bool) or not isinstance(band, int) or band < 1:
        raise ValidationError(
            "band must be a positive 1-based integer",
            source=href,
            detail={"parameter": "band"},
        )

    owns_client = reader is None and http is None
    if reader is not None:
        source: ByteRangeReader = reader
        close_client = None
    else:
        client = http if http is not None else HttpClient()
        source = HttpRangeReader(href, client, region=region, source_id=href)
        close_client = client if owns_client else None

    try:
        cog = CogReader(source, source_id=href)
        try:
            meta = await cog.open()
            if band > meta.samples_per_pixel:
                raise ValidationError(
                    f"band {band} is out of range; asset has "
                    f"{meta.samples_per_pixel} band(s)",
                    source=href,
                    detail={"parameter": "band"},
                )
            pixel_window = _resolve_window(window_bbox, meta, source_id=href)
            col_off, row_off, width, height = pixel_window
            if width == 0 or height == 0:
                # The window does not overlap the asset: an empty grid. Every
                # zone will fall back to the no-data indication (Req 8.8).
                gt = _window_geotransform(meta, col_off, row_off)
                return RasterGrid(
                    width=0,
                    height=0,
                    geotransform=gt,
                    nodata=meta.nodata,
                    dtype=meta.dtype,
                    values=[],
                )
            band_data = await cog.read_window_pixels(
                col_off=col_off,
                row_off=row_off,
                width=width,
                height=height,
                bands=[band],
            )
        except NetworkError as exc:
            # Reading the asset failed after the shared client's retries within
            # the 30-second window. Abort, leaving local storage unchanged, and
            # return an Error_Taxonomy error indicating the S3 read failure
            # (Requirement 12.6).
            raise NetworkError(
                "failed to read raster asset from S3 after retry attempts within "
                "the 30-second window; operation aborted and local storage left "
                "unchanged",
                source=href,
                detail={"reason": "s3_read_failure", "href": href},
                original=exc.original if exc.original is not None else str(exc),
            ) from exc
    finally:
        if close_client is not None:
            await close_client.aclose()

    gt = _window_geotransform(meta, col_off, row_off)
    return RasterGrid(
        width=width,
        height=height,
        geotransform=gt,
        nodata=meta.nodata,
        dtype=meta.dtype,
        values=band_data[0],
    )


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def _validate_href(raster_href: object) -> str:
    if not isinstance(raster_href, str) or not raster_href.strip():
        raise ValidationError(
            "raster_href must be a non-empty string",
            detail={"parameter": "raster_href"},
        )
    href = raster_href.strip()
    parts = urlsplit(href)
    scheme = parts.scheme.lower()
    if scheme not in ("s3", "http", "https"):
        raise ValidationError(
            "raster_href must be an s3:// or http(s):// URL",
            detail={"parameter": "raster_href"},
        )
    if scheme == "s3" and (not parts.netloc or not parts.path.lstrip("/")):
        raise ValidationError(
            "s3 raster_href must be of the form s3://<bucket>/<key>",
            detail={"parameter": "raster_href"},
        )
    return href

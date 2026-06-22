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
from urllib.parse import quote, urlsplit

from geo_common.errors import NetworkError, ValidationError
from geo_common.http import HttpClient

from geo_raster.cog import ByteRangeReader, CogMetadata, CogReader
from geo_raster.models import RasterGrid

__all__ = [
    "HttpRangeReader",
    "read_raster_grid",
    "DEFAULT_S3_REGION",
]

#: Default AWS region used when resolving an ``s3://`` URI to an HTTPS endpoint.
DEFAULT_S3_REGION = "us-east-1"

BBox = Tuple[float, float, float, float]


class HttpRangeReader:
    """A :class:`~geo_raster.cog.ByteRangeReader` over HTTP ``Range`` requests.

    Reads exact byte ranges from an ``http(s)://`` URL or an ``s3://`` URI
    (resolved to its HTTPS virtual-hosted endpoint) using the shared
    :class:`HttpClient`. Each :meth:`read_range` issues a single ranged GET; a
    ``206 Partial Content`` response carries exactly the requested bytes, while
    a server that ignores ``Range`` (``200``) is sliced down to the range so the
    caller still sees only the requested window.
    """

    def __init__(
        self,
        href: str,
        http: HttpClient,
        *,
        region: str = DEFAULT_S3_REGION,
        source_id: Optional[str] = None,
    ) -> None:
        self._url = _href_to_url(href, region=region)
        self._http = http
        self._source_id = source_id or href

    async def read_range(self, offset: int, length: int) -> bytes:
        if length <= 0:
            return b""
        end = offset + length - 1
        headers = {"Range": f"bytes={offset}-{end}"}
        response = await self._http.get(self._url, headers=headers)
        content = response.content
        if response.status_code == 206:
            return content
        # Server ignored Range (200) or returned the whole object: slice locally
        # so the engine still only *uses* the requested window's bytes.
        return content[offset : offset + length]


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
# Window resolution
# ---------------------------------------------------------------------------


def _resolve_window(
    window_bbox: Optional[BBox], meta: CogMetadata, *, source_id: str
) -> Tuple[int, int, int, int]:
    """Resolve a world bbox to a clamped ``(col_off, row_off, width, height)``.

    Without a bbox the whole asset is the window. With a bbox the asset must be
    georeferenced (Requirement validation surface).
    """
    if window_bbox is None:
        return 0, 0, meta.width, meta.height

    if meta.geotransform is None:
        raise ValidationError(
            "asset has no georeferencing; cannot resolve a geographic window",
            source=source_id,
            detail={"parameter": "zones"},
        )
    min_x, min_y, max_x, max_y = window_bbox
    gt = meta.geotransform
    corners = [
        _world_to_pixel(min_x, min_y, gt),
        _world_to_pixel(min_x, max_y, gt),
        _world_to_pixel(max_x, min_y, gt),
        _world_to_pixel(max_x, max_y, gt),
    ]
    cols = [c for c, _ in corners]
    rows = [r for _, r in corners]
    col_start = max(0, int(math.floor(min(cols))))
    col_end = min(meta.width, int(math.ceil(max(cols))))
    row_start = max(0, int(math.floor(min(rows))))
    row_end = min(meta.height, int(math.ceil(max(rows))))
    if col_end <= col_start or row_end <= row_start:
        return col_start, row_start, 0, 0
    return col_start, row_start, col_end - col_start, row_end - row_start


def _window_geotransform(
    meta: CogMetadata, col_off: int, row_off: int
) -> Tuple[float, float, float, float, float, float]:
    """Return the geotransform of a window whose origin is ``(col_off, row_off)``.

    When the asset is not georeferenced, an identity pixel-space transform is
    used so cell centers map to ``(col + 0.5, row + 0.5)`` in image coordinates.
    """
    if meta.geotransform is None:
        return (float(col_off), 1.0, 0.0, float(row_off), 0.0, 1.0)
    origin_x, px_w, row_rot, origin_y, col_rot, px_h = meta.geotransform
    new_origin_x = origin_x + px_w * col_off + row_rot * row_off
    new_origin_y = origin_y + col_rot * col_off + px_h * row_off
    return (new_origin_x, px_w, row_rot, new_origin_y, col_rot, px_h)


def _world_to_pixel(
    x: float, y: float, gt: Tuple[float, float, float, float, float, float]
) -> Tuple[float, float]:
    """Invert a GDAL-style geotransform to fractional ``(col, row)``."""
    origin_x, px_w, row_rot, origin_y, col_rot, px_h = gt
    det = px_w * px_h - row_rot * col_rot
    if det == 0:
        raise ValidationError("asset geotransform is not invertible")
    dx = x - origin_x
    dy = y - origin_y
    col = (px_h * dx - row_rot * dy) / det
    row = (-col_rot * dx + px_w * dy) / det
    return col, row


# ---------------------------------------------------------------------------
# Parameter validation + s3:// -> HTTPS resolution
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


def _href_to_url(href: str, *, region: str) -> str:
    """Resolve an ``s3://<bucket>/<key>`` URI to its HTTPS virtual-hosted endpoint."""
    parts = urlsplit(href)
    if parts.scheme.lower() != "s3":
        return href
    bucket = parts.netloc
    key = parts.path.lstrip("/")
    if not bucket or not key:
        raise ValidationError(
            "s3 raster_href must be of the form s3://<bucket>/<key>",
            detail={"parameter": "raster_href"},
        )
    host = (
        f"{bucket}.s3.amazonaws.com"
        if region == "us-east-1"
        else f"{bucket}.s3.{region}.amazonaws.com"
    )
    return f"https://{host}/{quote(key)}"

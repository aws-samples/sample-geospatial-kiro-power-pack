"""Cloud-Optimized GeoTIFF (COG) conversion for ``geo-formats`` (Req 8.4, 12.2).

This module holds :func:`to_cog`, the engine behind ``geo-formats``'s ``to_cog``
MCP tool. It reads a source raster at ``src_href`` and writes a
**Cloud-Optimized GeoTIFF** to ``dst_href`` using rasterio's ``COG`` driver:
internally tiled, with overviews when the raster is large enough to warrant
them (Requirement 8.4 — "with overviews").

The conversion is **lossless** (DEFLATE compression, no lossy quantization), so
converting a raster to COG and reading it back yields, for every band, pixel
values exactly equal to the source — including nodata-flagged pixels (design
Property 2 / Requirements 12.2, 15.3).

Failure handling follows the cloud-optimized-formats contract (Requirement
12.7): if writing the output fails for any reason, the conversion **aborts**,
**removes any partial output** it created at ``dst_href`` (so a half-written COG
never lingers), and raises a ``geo_common`` ``Error_Taxonomy`` error. Malformed
inputs (missing/unreadable source) raise a taxonomy
:class:`~geo_common.errors.ValidationError`; write/I-O failures raise an
:class:`~geo_common.errors.UpstreamError` retaining the original detail.
"""

from __future__ import annotations

import os
from typing import Optional

from geo_common.errors import GeoError, UpstreamError, ValidationError

from geo_formats.models import FormatResult

__all__ = ["to_cog", "DEFAULT_COG_BLOCKSIZE", "DEFAULT_COG_COMPRESS"]

#: COG internal tile size. 512 is the rasterio/GDAL COG default; tiling is what
#: makes partial/windowed reads from object storage efficient.
DEFAULT_COG_BLOCKSIZE: int = 512

#: Lossless compression for the output. DEFLATE keeps the round-trip pixel-exact
#: (Property 2) while still shrinking the file.
DEFAULT_COG_COMPRESS: str = "DEFLATE"


def to_cog(
    src_href: str,
    dst_href: str,
    *,
    overviews: bool = True,
    blocksize: int = DEFAULT_COG_BLOCKSIZE,
    compress: str = DEFAULT_COG_COMPRESS,
    source: str = "geo-formats",
) -> FormatResult:
    """Convert the raster at ``src_href`` to a COG at ``dst_href`` (Req 8.4, 12.2).

    Reads every band of the source and writes a Cloud-Optimized GeoTIFF that
    preserves the source dtype, band count, CRS, geotransform and nodata value.
    When ``overviews`` is true the COG driver adds reduced-resolution overviews
    as the raster size warrants (a raster smaller than one tile legitimately
    has none); when false, no overviews are written.

    Returns a :class:`FormatResult` describing the written output. Raises a
    taxonomy :class:`~geo_common.errors.ValidationError` for a missing or
    unreadable source, and (after removing any partial output) an
    :class:`~geo_common.errors.UpstreamError` if writing the COG fails
    (Requirement 12.7).
    """
    import rasterio
    from rasterio.errors import RasterioError, RasterioIOError
    from rasterio.shutil import copy as rio_copy

    if not src_href:
        raise ValidationError(
            "to_cog requires a non-empty src_href", source=source,
            detail={"parameter": "src_href"},
        )
    if not dst_href:
        raise ValidationError(
            "to_cog requires a non-empty dst_href", source=source,
            detail={"parameter": "dst_href"},
        )

    # --- Read the source raster (malformed/unreadable source -> validation). ---
    try:
        src = rasterio.open(src_href)
    except (RasterioIOError, OSError) as exc:
        raise ValidationError(
            "could not open source raster %r: %s"
            % (src_href, str(exc) or type(exc).__name__),
            source=source,
            detail={"parameter": "src_href", "src_href": src_href},
            original=str(exc) or type(exc).__name__,
        ) from exc

    # --- Write the COG, cleaning up any partial output on failure (Req 12.7). ---
    existed_before = os.path.exists(dst_href)
    try:
        with src:
            profile = src.profile.copy()
            profile.update(
                driver="COG",
                blocksize=blocksize,
                compress=compress,
            )
            # The COG driver controls tiling/overviews itself; drop GTiff-only
            # creation keys that would otherwise conflict with the COG driver.
            for key in ("tiled", "blockxsize", "blockysize", "interleave"):
                profile.pop(key, None)
            profile["overviews"] = "AUTO" if overviews else "NONE"

            data = src.read()
            band_count = src.count
            src_nodata = src.nodata

        with rasterio.open(dst_href, "w", **profile) as dst:
            dst.write(data)

        overview_count = _overview_count(dst_href)
    except GeoError:
        _remove_partial(dst_href, existed_before=existed_before)
        raise
    except (RasterioError, RasterioIOError, OSError, ValueError) as exc:
        _remove_partial(dst_href, existed_before=existed_before)
        raise UpstreamError(
            "failed to write Cloud-Optimized GeoTIFF to %r: %s"
            % (dst_href, str(exc) or type(exc).__name__),
            source=source,
            detail={"dst_href": dst_href},
            original=str(exc) or type(exc).__name__,
        ) from exc

    size_bytes = os.path.getsize(dst_href) if os.path.exists(dst_href) else None
    return FormatResult(
        href=dst_href,
        fmt="COG",
        driver="COG",
        size_bytes=size_bytes,
        detail={
            "bands": band_count,
            "overviews": overview_count,
            "compress": compress,
            "blocksize": blocksize,
            "nodata": src_nodata,
        },
    )


def _overview_count(href: str) -> int:
    """Return the number of overview levels on band 1 of the written COG."""
    import rasterio

    try:
        with rasterio.open(href) as ds:
            if ds.count >= 1:
                return len(ds.overviews(1))
    except Exception:  # pragma: no cover - best-effort summary only
        return 0
    return 0


def _remove_partial(dst_href: str, *, existed_before: bool) -> None:
    """Remove a partially written output created by a failed conversion.

    Only removes ``dst_href`` when this call created it (it did not exist before
    the conversion began), so a pre-existing file is never destroyed by a failed
    overwrite attempt. Best-effort: cleanup failures are swallowed so the
    original taxonomy error is the one that surfaces.
    """
    if existed_before:
        return
    try:
        if os.path.exists(dst_href):
            os.remove(dst_href)
    except OSError:  # pragma: no cover - cleanup is best-effort
        pass

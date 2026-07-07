"""Per-pixel multi-asset band math reduced to zonal statistics (Req 8.7, 8.8).

This module closes the gap that single-asset :func:`geo_raster.window_reader.band_math`
cannot: on catalogs such as Earth Search, the bands an index needs (e.g. NDVI's
NIR ``B08`` and Red ``B04``) live in **separate single-band Cloud-Optimized
GeoTIFFs**, so a single ``asset_href`` can never reference both. :func:`zonal_band_math`
binds each ``B<n>`` token in an expression to its own asset href, reads each
band's window directly from S3/HTTP byte ranges over the union of the zones,
evaluates the expression **per pixel** to build a true derived-index grid (NDVI,
NDWI, NBR, ...), and then reduces that grid to per-zone statistics via the pure
:func:`geo_raster.zonal.compute_zonal_statistics`.

The result is a genuine per-pixel index distribution summarized per zone
(``min``/``max``/``mean``/``sum``/``count``) — not a ratio-of-means
approximation — obtained in a single MCP call with no external raster toolchain.

Correctness contract:

* The referenced band assets must share the same pixel grid over the read window
  (identical resolved size and geotransform). For Sentinel-2 this holds for the
  10 m bands (B02/B03/B04/B08). Mixing resolutions (e.g. a 10 m band with a 20 m
  band) is rejected with a :class:`~geo_common.errors.ValidationError` rather
  than silently misaligned (Requirement 7.12 validation surface).
* Zones must already be expressed in the assets' CRS — the same contract as
  :func:`geo_raster.zonal.zonal_statistics` (reproject vector zones first).
* nodata / divide-by-zero propagate to a masked (NaN) pixel that is excluded
  from every zone's statistics, exactly as in single-asset band math.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Dict, List, Optional, Sequence

from geo_common.errors import ValidationError
from geo_common.http import HttpClient

from geo_raster.cog import ByteRangeReader
from geo_raster.models import FeatureCollection, RasterGrid, ZoneStat
from geo_raster.reader import DEFAULT_S3_REGION, read_raster_grid
from geo_raster.window_reader import _eval_node, _parse_expression
from geo_raster.zonal import (
    _validate_stats,
    _zones_bbox,
    compute_zonal_statistics,
)

__all__ = ["zonal_band_math"]

_SOURCE = "geo-raster"
_BAND_TOKEN = re.compile(r"^B(\d+)$")
#: Geotransform elements are compared with this absolute tolerance when checking
#: that two band assets share the same pixel grid over the read window.
_GT_TOL = 1e-6


def _normalize_assets(assets: object) -> Dict[int, str]:
    """Map an ``assets`` argument to ``{band_number: href}`` (Requirement 7.12).

    Accepts keys as band tokens (``"B08"``, ``"B8"``), bare numbers (``"8"``,
    ``8``), and href values as non-empty strings. Leading zeros are normalized
    (``"B08"`` and ``"B8"`` both mean band 8), so a token cannot be declared
    twice under different spellings.
    """
    if not isinstance(assets, dict) or not assets:
        raise ValidationError(
            "assets must be a non-empty mapping of band token -> asset href "
            "(e.g. {\"B08\": \"s3://.../B08.tif\", \"B04\": \"s3://.../B04.tif\"})",
            source=_SOURCE,
            detail={"parameter": "assets"},
        )
    out: Dict[int, str] = {}
    for raw_key, raw_href in assets.items():
        band = _coerce_band_key(raw_key)
        if not isinstance(raw_href, str) or not raw_href.strip():
            raise ValidationError(
                f"asset href for band token {raw_key!r} must be a non-empty string",
                source=_SOURCE,
                detail={"parameter": "assets"},
            )
        if band in out:
            raise ValidationError(
                f"band {band} is declared more than once in assets "
                "(check for duplicate B<n>/B0<n> spellings)",
                source=_SOURCE,
                detail={"parameter": "assets"},
            )
        out[band] = raw_href.strip()
    return out


def _coerce_band_key(key: object) -> int:
    """Coerce an ``assets`` key to a 1-based band number (Requirement 7.12)."""
    if isinstance(key, bool):  # bool is an int subclass; reject explicitly.
        raise ValidationError(
            "assets band token must be a B<n> string or a band number, not a bool",
            source=_SOURCE,
            detail={"parameter": "assets"},
        )
    if isinstance(key, int):
        band = key
    elif isinstance(key, str):
        text = key.strip()
        match = _BAND_TOKEN.match(text)
        if match:
            band = int(match.group(1))
        elif text.isdigit():
            band = int(text)
        else:
            raise ValidationError(
                f"invalid assets band token {key!r}; use B<n> (e.g. B08) or a "
                "band number",
                source=_SOURCE,
                detail={"parameter": "assets"},
            )
    else:
        raise ValidationError(
            f"invalid assets band token {key!r}; use B<n> (e.g. B08) or a band "
            "number",
            source=_SOURCE,
            detail={"parameter": "assets"},
        )
    if band < 1:
        raise ValidationError(
            "assets band tokens must be 1-based (e.g. B1)",
            source=_SOURCE,
            detail={"parameter": "assets"},
        )
    return band


def _assert_aligned(reference: RasterGrid, other: RasterGrid, band: int) -> None:
    """Raise unless ``other`` shares ``reference``'s window grid (Req 7.12)."""
    if (other.width, other.height) != (reference.width, reference.height):
        raise ValidationError(
            "band assets are not on the same pixel grid over the read window "
            f"(band {band} resolved to {other.width}x{other.height}, expected "
            f"{reference.width}x{reference.height}); use bands at a common "
            "resolution or reproject/resample them first",
            source=_SOURCE,
            detail={"parameter": "assets"},
        )
    for a, b in zip(reference.geotransform, other.geotransform):
        if not math.isclose(a, b, abs_tol=_GT_TOL, rel_tol=0.0):
            raise ValidationError(
                f"band {band} does not align with the reference band's "
                "geotransform over the read window; the assets must share the "
                "same grid (same origin and resolution)",
                source=_SOURCE,
                detail={"parameter": "assets"},
            )


async def zonal_band_math(
    *,
    assets: Dict[str, str],
    expression: str,
    zones: FeatureCollection,
    stats: Optional[Sequence[str]] = None,
    http: Optional[HttpClient] = None,
    region: str = DEFAULT_S3_REGION,
    readers: Optional[Dict[int, ByteRangeReader]] = None,
) -> List[ZoneStat]:
    """Compute a per-pixel band-math index across assets, summarized per zone.

    Parameters
    ----------
    assets:
        Mapping of band token to asset href, e.g.
        ``{"B08": ".../B08.tif", "B04": ".../B04.tif"}``. Each ``B<n>`` the
        ``expression`` references must have a matching entry; the first band of
        each asset is read.
    expression:
        Band-math expression referencing the tokens, e.g. NDVI is
        ``"(B08 - B04) / (B08 + B04)"``. Only the safe arithmetic subset shared
        with :func:`geo_raster.window_reader.band_math` is allowed.
    zones:
        GeoJSON vector zones (in the assets' CRS) to summarize the index over.
    stats:
        Subset of ``min``/``max``/``mean``/``sum``/``count`` (default: all).
    http / region / readers:
        Shared client / S3 region / optional per-band pre-built byte-range
        readers (used by tests with in-memory assets; keyed by band number).

    Returns
    -------
    list[ZoneStat]
        Per-zone statistics over the true per-pixel index. Zones with no
        overlapping cells get a no-data indication (Requirement 8.8).

    Raises
    ------
    ValidationError
        For malformed ``assets``/``expression``/``stats``/zones, a referenced
        band with no asset, or band assets that do not share a common pixel grid
        over the read window (Requirement 7.12).
    """
    requested = _validate_stats(stats, source=_SOURCE)
    assets_by_band = _normalize_assets(assets)
    _node, referenced = _parse_expression(expression, source_id=_SOURCE)

    missing = [b for b in referenced if b not in assets_by_band]
    if missing:
        have = ", ".join(f"B{b}" for b in sorted(assets_by_band))
        need = ", ".join(f"B{b}" for b in missing)
        raise ValidationError(
            f"expression references band(s) {need} with no matching asset; "
            f"assets provides: {have}",
            source=_SOURCE,
            detail={"parameter": "assets"},
        )

    if not zones.features:
        # No zones -> nothing to summarize; no read is issued.
        return []

    window_bbox = _zones_bbox(zones, source=_SOURCE)

    owns_client = readers is None and http is None
    client = HttpClient() if owns_client else http
    try:
        grids: Dict[int, RasterGrid] = {}
        for band in referenced:
            reader_b = None if readers is None else readers.get(band)
            grids[band] = await read_raster_grid(
                raster_href=assets_by_band[band],
                window_bbox=window_bbox,
                band=1,
                http=client if reader_b is None else None,
                region=region,
                reader=reader_b,
            )
    finally:
        if owns_client and client is not None:
            await client.aclose()

    index_grid = _evaluate_index(grids, referenced, _node)
    return compute_zonal_statistics(index_grid, zones, requested, source=_SOURCE)


def _evaluate_index(
    grids: Dict[int, RasterGrid],
    referenced: Sequence[int],
    node,
) -> RasterGrid:
    """Evaluate the parsed expression per pixel across aligned band grids.

    Produces a single-band float :class:`RasterGrid`. A pixel is masked (NaN,
    excluded from every zone's statistics) when any contributing band is nodata
    or NaN, or when evaluation hits a zero denominator / domain error.
    """
    reference = grids[referenced[0]]
    for band in referenced[1:]:
        _assert_aligned(reference, grids[band], band)

    width, height = reference.width, reference.height
    if width == 0 or height == 0:
        # Window does not overlap the assets: empty grid -> all zones no-data.
        return RasterGrid(
            width=0,
            height=0,
            geotransform=reference.geotransform,
            nodata=None,
            dtype="float64",
            values=[],
        )

    band_values = {b: grids[b].values for b in referenced}
    band_nodata = {b: grids[b].nodata for b in referenced}
    # Map each literal identifier in the expression (e.g. "B08", "B8") to its
    # band number, so the evaluator can look up env by the exact node.id and
    # both zero-padded and unpadded spellings resolve correctly.
    id_to_band: Dict[str, int] = {}
    for name in {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}:
        match = _BAND_TOKEN.match(name)
        if match:
            id_to_band[name] = int(match.group(1))
    out: List[float] = [math.nan] * (width * height)

    for i in range(width * height):
        env: Dict[str, float] = {}
        masked = False
        for ident, b in id_to_band.items():
            v = band_values[b][i]
            nd = band_nodata[b]
            if (nd is not None and v == nd) or v != v:  # nodata or NaN
                masked = True
                break
            env[ident] = v
        if masked:
            continue
        try:
            out[i] = float(_eval_node(node, env))
        except (ZeroDivisionError, ValueError):
            out[i] = math.nan

    return RasterGrid(
        width=width,
        height=height,
        geotransform=reference.geotransform,
        nodata=None,
        dtype="float64",
        values=out,
    )

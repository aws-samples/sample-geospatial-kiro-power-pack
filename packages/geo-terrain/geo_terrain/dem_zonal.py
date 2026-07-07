"""Zonal statistics over a DEM COG: elevation, slope, or aspect per zone.

This is the terrain analogue of ``geo-raster``'s ``zonal_band_math``: read only
the tiles of a **DEM Cloud-Optimized GeoTIFF** overlapping the zones (by byte
range, via the shared :func:`geo_common.raster_read.read_raster_grid`), optionally
turn the elevation window into a per-cell slope or aspect surface
(:mod:`geo_terrain.derivatives`), then reduce it to per-zone statistics with the
shared :func:`geo_common.zonal.compute_zonal_statistics`. No cross-server
dependency — every shared piece lives in ``geo-common``.

The DEM href is user-supplied (e.g. a Copernicus GLO-30 or 3DEP tile); the zones
must already be in the DEM's own CRS (same contract as ``zonal_statistics``).
Cell size for slope/aspect is taken from the DEM's geotransform, so a
**projected** DEM (metres) yields slope in the expected degrees; for a
geographic DEM pass ``pixel_size_m`` to supply the ground cell size.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from geo_common.errors import ValidationError
from geo_common.raster_models import FeatureCollection, RasterGrid, ZoneStat
from geo_common.raster_read import DEFAULT_S3_REGION, read_raster_grid
from geo_common.zonal import _zones_bbox, compute_zonal_statistics

from geo_terrain.dem_sources import read_dem_grid
from geo_terrain.derivatives import compute_aspect, compute_slope

__all__ = ["dem_zonal", "MEASURES"]

_SOURCE = "geo-terrain"

#: Supported per-zone DEM measures.
MEASURES = ("elevation", "slope", "aspect")


def _grid_to_rows(grid: RasterGrid) -> List[List[Optional[float]]]:
    """Turn a flat :class:`RasterGrid` into a 2D row-major grid, nodata -> None."""
    nodata = grid.nodata
    rows: List[List[Optional[float]]] = []
    for r in range(grid.height):
        row: List[Optional[float]] = []
        for c in range(grid.width):
            v = grid.values[r * grid.width + c]
            if (nodata is not None and v == nodata) or v != v:  # nodata or NaN
                row.append(None)
            else:
                row.append(float(v))
        rows.append(row)
    return rows


def _rows_to_grid(rows: List[List[Optional[float]]], template: RasterGrid) -> RasterGrid:
    """Flatten a 2D derivative grid back into a :class:`RasterGrid` (None -> NaN)."""
    flat: List[float] = []
    for row in rows:
        for v in row:
            flat.append(float("nan") if v is None else float(v))
    return RasterGrid(
        width=template.width,
        height=template.height,
        geotransform=template.geotransform,
        nodata=None,  # None cells are encoded as NaN and excluded by the reducer
        dtype="float64",
        values=flat,
    )


def _cell_sizes_m(
    grid: RasterGrid, pixel_size_m: Optional[Sequence[float]]
) -> Tuple[float, float]:
    """Ground cell size (metres) in x and y for slope/aspect.

    Uses an explicit ``pixel_size_m`` override when given; otherwise the DEM
    geotransform's pixel scale (correct for a projected DEM in metres).
    """
    if pixel_size_m is not None:
        if len(pixel_size_m) != 2 or any(float(s) <= 0 for s in pixel_size_m):
            raise ValidationError(
                "pixel_size_m must be two positive numbers (metres) [x, y]",
                source=_SOURCE,
                detail={"parameter": "pixel_size_m"},
            )
        return float(pixel_size_m[0]), float(pixel_size_m[1])
    _, px_w, _, _, _, px_h = grid.geotransform
    cx, cy = abs(float(px_w)), abs(float(px_h))
    if cx == 0.0 or cy == 0.0:
        raise ValidationError(
            "DEM geotransform has zero pixel size; pass pixel_size_m explicitly",
            source=_SOURCE,
            detail={"parameter": "pixel_size_m"},
        )
    return cx, cy


async def dem_zonal(
    *,
    dem_href: Optional[str] = None,
    dem_source: Optional[str] = None,
    zones: FeatureCollection,
    measure: str = "elevation",
    stats: Optional[Sequence[str]] = None,
    band: int = 1,
    pixel_size_m: Optional[Sequence[float]] = None,
    http=None,
    region: str = DEFAULT_S3_REGION,
    reader=None,
    readers=None,
) -> List[ZoneStat]:
    """Per-zone DEM statistics for ``elevation``, ``slope``, or ``aspect``.

    Supply exactly one of ``dem_href`` (a specific DEM COG) or ``dem_source`` (a
    named source like ``"glo30"``, whose overlapping tiles are resolved and
    mosaicked automatically). Reads only the tiles overlapping the union of
    ``zones`` (which must be in the DEM's CRS), builds the requested surface, and
    returns one :class:`ZoneStat` per zone. ``elevation`` reduces the raw DEM;
    ``slope``/``aspect`` first compute the derivative per cell (cell size from the
    geotransform, or ``pixel_size_m``). Zones with no overlapping cells get a
    no-data indication (Requirement 8.8).
    """
    if measure not in MEASURES:
        raise ValidationError(
            f"unknown measure {measure!r}; use one of {', '.join(MEASURES)}",
            source=_SOURCE,
            detail={"parameter": "measure"},
        )
    if bool(dem_href) == bool(dem_source):
        raise ValidationError(
            "supply exactly one of dem_href or dem_source",
            source=_SOURCE,
            detail={"parameter": "dem_href"},
        )
    if not zones.features:
        return []

    window_bbox = _zones_bbox(zones, source=_SOURCE)
    if dem_source is not None:
        grid = await read_dem_grid(
            source=dem_source,
            bbox=window_bbox,
            band=band,
            http=http,
            region=region,
            readers=readers,
        )
    else:
        grid = await read_raster_grid(
            raster_href=dem_href,
            window_bbox=window_bbox,
            band=band,
            http=http,
            region=region,
            reader=reader,
        )

    if measure == "elevation":
        target = grid
    else:
        if grid.width < 2 or grid.height < 2:
            raise ValidationError(
                "slope/aspect need a DEM window of at least 2x2 cells over the "
                "zones; widen the zones or use measure='elevation'",
                source=_SOURCE,
                detail={"parameter": "measure", "width": grid.width, "height": grid.height},
            )
        cx, cy = _cell_sizes_m(grid, pixel_size_m)
        rows = _grid_to_rows(grid)
        if measure == "slope":
            deriv = compute_slope(rows, cellsize_x_m=cx, cellsize_y_m=cy)
        else:  # aspect
            deriv = compute_aspect(rows, cellsize_x_m=cx, cellsize_y_m=cy)
        target = _rows_to_grid(deriv, grid)

    return compute_zonal_statistics(target, zones, stats, source=_SOURCE)

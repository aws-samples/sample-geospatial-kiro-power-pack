"""Named DEM Cloud-Optimized GeoTIFF sources + tile resolution and mosaicking.

Lets ``dem_zonal`` read a named DEM source (rather than a hand-supplied href) by
resolving which tiles a bounding box overlaps and, when it spans more than one,
**mosaicking** the overlapping windows into a single grid before reducing.

Currently registered:

* ``glo30`` — Copernicus DEM GLO-30 (~30 m global), on the AWS Open Data bucket
  ``copernicus-dem-30m``. EPSG:4326, 1°×1° tiles named by their SW corner, all
  sharing one global pixel grid (so adjacent tiles mosaic exactly).

The resolver (:func:`resolve_tiles`) is a pure function of the bbox and is fully
unit-testable offline; the mosaic read (:func:`read_dem_grid`) reads only the
window of each overlapping tile via the shared byte-range reader and stitches
the results on their common grid.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from geo_common.errors import ValidationError
from geo_common.raster_models import RasterGrid
from geo_common.raster_read import DEFAULT_S3_REGION, read_raster_grid

__all__ = ["DEM_SOURCES", "resolve_tiles", "read_dem_grid"]

_SOURCE = "geo-terrain"
BBox = Tuple[float, float, float, float]

def _glo30_tile_href(lat_sw: int, lon_sw: int) -> str:
    """GLO-30 tile href for the 1° cell with SW corner ``(lon_sw, lat_sw)``.

    Copernicus DEM tiles are named by their **SW corner**, e.g. the cell
    37-38N, 119-120W is ``Copernicus_DSM_COG_10_N37_00_W120_00_DEM``.
    """
    ns = "N" if lat_sw >= 0 else "S"
    ew = "E" if lon_sw >= 0 else "W"
    name = f"Copernicus_DSM_COG_10_{ns}{abs(lat_sw):02d}_00_{ew}{abs(lon_sw):03d}_00_DEM"
    return f"s3://copernicus-dem-30m/{name}/{name}.tif"


def _threedep_tile_href(lat_sw: int, lon_sw: int) -> str:
    """USGS 3DEP 1/3-arc-second tile href for the 1° cell with SW corner ``(lon_sw, lat_sw)``.

    USGS names tiles by their **NW corner** (north edge = ``lat_sw + 1``, west
    edge = ``lon_sw``), e.g. the cell 37-38N, 119-120W is ``n38w120`` at
    ``s3://prd-tnm/StagedProducts/Elevation/13/TIFF/current/n38w120/USGS_13_n38w120.tif``.
    US coverage only (CONUS, HI, territories, partial AK).
    """
    north_edge = lat_sw + 1
    ns = "n" if north_edge >= 0 else "s"
    ew = "e" if lon_sw >= 0 else "w"
    cell = f"{ns}{abs(north_edge):02d}{ew}{abs(lon_sw):03d}"
    return (
        f"s3://prd-tnm/StagedProducts/Elevation/13/TIFF/current/{cell}/USGS_13_{cell}.tif"
    )


#: Registered named DEM sources. Each documents the CRS its zones must be in and
#: supplies the per-cell tile href builder (all are 1°×1° EPSG:4326 grids that
#: share a global pixel grid, so overlapping tiles mosaic exactly).
DEM_SOURCES: Dict[str, Dict[str, object]] = {
    "glo30": {
        "name": "Copernicus DEM GLO-30",
        "crs": "EPSG:4326",
        "resolution": "~30 m (1 arc-second)",
        "bucket": "copernicus-dem-30m",
        "license": "Copernicus DEM open licence",
        "coverage": "global",
        "href": _glo30_tile_href,
    },
    "3dep": {
        "name": "USGS 3DEP seamless 1/3 arc-second",
        "crs": "EPSG:4326",
        "resolution": "~10 m (1/3 arc-second)",
        "bucket": "prd-tnm",
        "license": "public domain (US Government)",
        "coverage": "United States (CONUS, HI, territories, partial AK)",
        "href": _threedep_tile_href,
    },
}


def resolve_tiles(source: str, bbox: Sequence[float]) -> List[str]:
    """Return the hrefs of the ``source`` tiles overlapping ``bbox`` (min-first).

    ``bbox`` is ``(min_lon, min_lat, max_lon, max_lat)`` in the source's CRS
    (EPSG:4326). Both registered sources are 1°×1° degree-tiled: the result is a
    tile per integer-degree cell whose ``[k, k+1)`` extent intersects the box,
    named by that source's corner convention (GLO-30 SW corner, 3DEP NW corner).
    """
    if source not in DEM_SOURCES:
        raise ValidationError(
            f"unknown DEM source {source!r}; available: {', '.join(sorted(DEM_SOURCES))}",
            source=_SOURCE,
            detail={"parameter": "dem_source"},
        )
    if len(bbox) != 4:
        raise ValidationError(
            "bbox must be (min_lon, min_lat, max_lon, max_lat)",
            source=_SOURCE,
            detail={"parameter": "bbox"},
        )
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)
    if min_lon > max_lon or min_lat > max_lat:
        raise ValidationError(
            "bbox min must not exceed max",
            source=_SOURCE,
            detail={"parameter": "bbox"},
        )
    href_of = DEM_SOURCES[source]["href"]
    # Tiles with SW corner k cover [k, k+1); the box [lo, hi] needs k from
    # floor(lo) to ceil(hi)-1 (so a hi exactly on an integer edge adds no tile).
    lat_tiles = range(math.floor(min_lat), math.ceil(max_lat))
    lon_tiles = range(math.floor(min_lon), math.ceil(max_lon))
    hrefs: List[str] = []
    for lat_sw in lat_tiles:
        for lon_sw in lon_tiles:
            hrefs.append(href_of(lat_sw, lon_sw))  # type: ignore[operator]
    return hrefs


def _mosaic(grids: List[RasterGrid]) -> RasterGrid:
    """Stitch grids that share one north-up pixel grid into a single RasterGrid.

    All inputs must have the same pixel size (px_w/px_h) and zero rotation — true
    for a single degree-tiled source like GLO-30. Cells are placed at their
    global pixel offset relative to the union's top-left; gaps and each tile's
    nodata become NaN (excluded by the zonal reducer).
    """
    ref = grids[0]
    _, px_w, row_rot, _, col_rot, px_h = ref.geotransform
    for g in grids:
        _, gw, grr, _, gcr, gh = g.geotransform
        if row_rot or col_rot or grr or gcr:
            raise ValidationError(
                "cannot mosaic rotated grids", source=_SOURCE, detail={"parameter": "dem_source"}
            )
        if not (math.isclose(gw, px_w, rel_tol=0, abs_tol=1e-9) and math.isclose(gh, px_h, rel_tol=0, abs_tol=1e-9)):
            raise ValidationError(
                "cannot mosaic DEM tiles with differing pixel sizes",
                source=_SOURCE,
                detail={"parameter": "dem_source"},
            )
    ref_origin_x = min(g.geotransform[0] for g in grids)
    ref_origin_y = max(g.geotransform[3] for g in grids)
    placements: List[Tuple[int, int, RasterGrid]] = []
    width = height = 0
    for g in grids:
        col_off = int(round((g.geotransform[0] - ref_origin_x) / px_w))
        row_off = int(round((ref_origin_y - g.geotransform[3]) / (-px_h)))
        placements.append((col_off, row_off, g))
        width = max(width, col_off + g.width)
        height = max(height, row_off + g.height)

    values = [math.nan] * (width * height)
    for col_off, row_off, g in placements:
        gnodata = g.nodata
        for r in range(g.height):
            base = (row_off + r) * width + col_off
            gbase = r * g.width
            for c in range(g.width):
                v = g.values[gbase + c]
                if gnodata is not None and v == gnodata:
                    continue  # leave NaN
                values[base + c] = v
    return RasterGrid(
        width=width,
        height=height,
        geotransform=(ref_origin_x, px_w, 0.0, ref_origin_y, 0.0, px_h),
        nodata=None,
        dtype="float64",
        values=values,
    )


async def read_dem_grid(
    *,
    source: str,
    bbox: Sequence[float],
    band: int = 1,
    http=None,
    region: str = DEFAULT_S3_REGION,
    readers: Optional[Dict[str, object]] = None,
) -> RasterGrid:
    """Read a DEM window from a named ``source``, mosaicking across tiles as needed.

    Resolves the tiles overlapping ``bbox``, reads only each tile's overlapping
    window, and returns a single :class:`RasterGrid`. ``readers`` (test-only)
    maps a tile href to a pre-built byte-range reader for in-memory tiles.
    """
    hrefs = resolve_tiles(source, bbox)
    readers = readers or {}
    grids: List[RasterGrid] = []
    for href in hrefs:
        r = readers.get(href)
        # A tile that is missing/unreadable simply contributes nothing; only a
        # window that overlaps returns cells.
        grid = await read_raster_grid(
            raster_href=href,
            window_bbox=tuple(float(v) for v in bbox),
            band=band,
            http=http,
            region=region,
            reader=r,
        )
        if grid.width > 0 and grid.height > 0:
            grids.append(grid)

    if not grids:
        # No tile overlapped: an empty grid (every zone -> no-data).
        return RasterGrid(width=0, height=0, geotransform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
                          nodata=None, dtype="float64", values=[])
    if len(grids) == 1:
        return grids[0]
    return _mosaic(grids)

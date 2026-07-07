"""Tests for named DEM sources: GLO-30 tile resolution and cross-tile mosaicking.

The tile resolver is a pure function (offline). The mosaic read is exercised with
synthetic adjacent in-memory tiles built via ``geo_common.testing`` — no network.
"""

from __future__ import annotations

import math

import pytest

from geo_common.errors import ValidationError
from geo_common.raster_models import Feature, FeatureCollection, GeoJSONGeometry, RasterGrid
from geo_common.testing import RecordingByteRangeReader, build_cog

from geo_terrain.dem_sources import _mosaic, read_dem_grid, resolve_tiles
from geo_terrain.dem_zonal import dem_zonal


# --- resolver (pure) --------------------------------------------------------


def test_resolve_single_tile_naming():
    # A point/box inside 37-38N, 119-120W resolves to the N37/W120 SW-corner tile.
    hrefs = resolve_tiles("glo30", (-119.6, 37.2, -119.4, 37.6))
    assert hrefs == [
        "s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N37_00_W120_00_DEM/"
        "Copernicus_DSM_COG_10_N37_00_W120_00_DEM.tif"
    ]


def test_resolve_spans_two_tiles_across_a_meridian():
    hrefs = resolve_tiles("glo30", (-120.0, 37.0, -118.0, 38.0))
    names = [h.rsplit("/", 1)[1] for h in hrefs]
    assert names == [
        "Copernicus_DSM_COG_10_N37_00_W120_00_DEM.tif",
        "Copernicus_DSM_COG_10_N37_00_W119_00_DEM.tif",
    ]


def test_resolve_integer_top_edge_adds_no_extra_tile():
    # max_lat exactly on an integer edge must not pull in the tile above.
    hrefs = resolve_tiles("glo30", (10.2, 5.0, 10.8, 6.0))
    assert len(hrefs) == 1
    assert "N05_00_E010" in hrefs[0]


def test_resolve_unknown_source_rejected():
    with pytest.raises(ValidationError) as exc:
        resolve_tiles("srtm-cog", (0.0, 0.0, 1.0, 1.0))
    assert exc.value.detail.get("parameter") == "dem_source"


# --- mosaic stitching (direct) ---------------------------------------------


def test_mosaic_stitches_two_adjacent_grids_side_by_side():
    left = RasterGrid(width=2, height=2, geotransform=(-120.0, 0.5, 0.0, 38.0, 0.0, -0.5),
                      nodata=None, values=[1.0, 2.0, 3.0, 4.0])
    right = RasterGrid(width=2, height=2, geotransform=(-119.0, 0.5, 0.0, 38.0, 0.0, -0.5),
                       nodata=None, values=[5.0, 6.0, 7.0, 8.0])
    m = _mosaic([left, right])
    assert (m.width, m.height) == (4, 2)
    # Row-major: row0 = left row0 + right row0, row1 = left row1 + right row1.
    assert m.values == [1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
    assert m.geotransform[0] == pytest.approx(-120.0)  # union top-left x
    assert m.geotransform[3] == pytest.approx(38.0)


def test_mosaic_rejects_mismatched_pixel_sizes():
    a = RasterGrid(width=2, height=2, geotransform=(-120.0, 0.5, 0.0, 38.0, 0.0, -0.5),
                   nodata=None, values=[1.0, 2.0, 3.0, 4.0])
    b = RasterGrid(width=2, height=2, geotransform=(-119.0, 1.0, 0.0, 38.0, 0.0, -1.0),
                   nodata=None, values=[5.0, 6.0, 7.0, 8.0])
    with pytest.raises(ValidationError):
        _mosaic([a, b])


# --- read_dem_grid end-to-end (synthetic tiles) -----------------------------


def _tile(origin_lon, values):
    return build_cog(
        width=2, height=2, tile_width=2, tile_height=2,
        bands=[values], dtype="float32",
        geotransform=(origin_lon, 0.5, 0.0, 38.0, 0.0, -0.5),
    )


async def test_read_dem_grid_mosaics_two_tiles():
    bbox = (-120.0, 37.0, -118.0, 38.0)
    hrefs = resolve_tiles("glo30", bbox)
    readers = {
        hrefs[0]: RecordingByteRangeReader(_tile(-120.0, [1.0, 2.0, 3.0, 4.0])),
        hrefs[1]: RecordingByteRangeReader(_tile(-119.0, [5.0, 6.0, 7.0, 8.0])),
    }
    grid = await read_dem_grid(source="glo30", bbox=bbox, readers=readers)
    assert (grid.width, grid.height) == (4, 2)
    assert grid.values == [1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]


async def test_dem_zonal_with_named_source_over_mosaic():
    bbox = (-120.0, 37.0, -118.0, 38.0)
    hrefs = resolve_tiles("glo30", bbox)
    readers = {
        hrefs[0]: RecordingByteRangeReader(_tile(-120.0, [1.0, 2.0, 3.0, 4.0])),
        hrefs[1]: RecordingByteRangeReader(_tile(-119.0, [5.0, 6.0, 7.0, 8.0])),
    }
    # A zone covering the whole mosaic extent.
    ring = [[-120.0, 37.0], [-118.0, 37.0], [-118.0, 38.0], [-120.0, 38.0], [-120.0, 37.0]]
    zones = FeatureCollection(features=[
        Feature(geometry=GeoJSONGeometry(type="Polygon", coordinates=[ring]),
                properties={"zone_id": "all"})
    ])
    result = await dem_zonal(
        dem_source="glo30", zones=zones, measure="elevation",
        stats=["min", "max", "mean", "count"], readers=readers,
    )
    assert result[0].statistics == {"min": 1.0, "max": 8.0, "mean": 4.5, "count": 8.0}


async def test_dem_zonal_requires_exactly_one_of_href_or_source():
    ring = [[-120.0, 37.0], [-118.0, 37.0], [-118.0, 38.0], [-120.0, 38.0], [-120.0, 37.0]]
    zones = FeatureCollection(features=[
        Feature(geometry=GeoJSONGeometry(type="Polygon", coordinates=[ring]),
                properties={"zone_id": "z"})
    ])
    with pytest.raises(ValidationError):
        await dem_zonal(zones=zones)  # neither
    with pytest.raises(ValidationError):
        await dem_zonal(zones=zones, dem_href="s3://b/x.tif", dem_source="glo30")  # both


import os

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from geo_common.http import HttpClient


# --- 3DEP naming (NW corner) -----------------------------------------------


def test_resolve_3dep_nw_corner_naming():
    # Cell 37-38N, 119-120W -> USGS names by NW corner: n38w120.
    hrefs = resolve_tiles("3dep", (-119.6, 37.2, -119.4, 37.6))
    assert hrefs == [
        "s3://prd-tnm/StagedProducts/Elevation/13/TIFF/current/n38w120/USGS_13_n38w120.tif"
    ]


# --- resolver property ------------------------------------------------------


@pytest.mark.property
@settings(deadline=None)
@given(
    source=st.sampled_from(["glo30", "3dep"]),
    min_lon=st.floats(min_value=-179.0, max_value=176.0, allow_nan=False, allow_infinity=False),
    min_lat=st.floats(min_value=-89.0, max_value=86.0, allow_nan=False, allow_infinity=False),
    w=st.floats(min_value=0.05, max_value=3.0),
    h=st.floats(min_value=0.05, max_value=3.0),
)
def test_resolve_tiles_count_and_center_coverage(source, min_lon, min_lat, w, h):
    max_lon, max_lat = min_lon + w, min_lat + h
    hrefs = resolve_tiles(source, (min_lon, min_lat, max_lon, max_lat))
    # One tile per integer-degree cell the box spans, no duplicates.
    n_lon = math.ceil(max_lon) - math.floor(min_lon)
    n_lat = math.ceil(max_lat) - math.floor(min_lat)
    assert len(hrefs) == n_lon * n_lat
    assert len(set(hrefs)) == len(hrefs)
    # The tile covering the box center is in the resolved set.
    cx, cy = (min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0
    center = resolve_tiles(source, (cx, cy, cx + 1e-6, cy + 1e-6))
    assert center[0] in hrefs


# --- mosaic round-trip property --------------------------------------------


@pytest.mark.property
@settings(deadline=None)
@given(
    width=st.integers(min_value=2, max_value=5),
    height=st.integers(min_value=2, max_value=5),
    px=st.sampled_from([0.25, 0.5, 1.0]),
    ox=st.sampled_from([-120.0, 0.0, 10.0]),
    oy=st.sampled_from([38.0, 0.0, -5.0]),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_mosaic_reassembles_a_split_grid(width, height, px, ox, oy, seed):
    """Splitting a grid into blocks and mosaicking them reconstructs the original."""
    import random

    rng = random.Random(seed)
    master = [float(rng.randint(-1000, 8000)) for _ in range(width * height)]
    gt = (ox, px, 0.0, oy, 0.0, -px)
    csplit = rng.randint(1, width - 1)
    rsplit = rng.randint(1, height - 1)

    blocks = []
    for (r0, r1, c0, c1) in [
        (0, rsplit, 0, csplit), (0, rsplit, csplit, width),
        (rsplit, height, 0, csplit), (rsplit, height, csplit, width),
    ]:
        vals = [master[r * width + c] for r in range(r0, r1) for c in range(c0, c1)]
        sub_gt = (ox + c0 * px, px, 0.0, oy - r0 * px, 0.0, -px)
        blocks.append(RasterGrid(width=c1 - c0, height=r1 - r0, geotransform=sub_gt,
                                 nodata=None, values=vals))
    rng.shuffle(blocks)
    m = _mosaic(blocks)
    assert (m.width, m.height) == (width, height)
    assert m.values == master


# --- live reads (opt-in) ----------------------------------------------------


@pytest.mark.skipif(not os.environ.get("RUN_LIVE_GLO30"), reason="set RUN_LIVE_GLO30=1 for a live Copernicus GLO-30 read")
async def test_live_glo30_reads_sierra_elevation():
    # Tiny box near Mt Whitney (~4,400 m). Real anonymous S3 read.
    grid = await read_dem_grid(source="glo30", bbox=(-118.30, 36.575, -118.28, 36.585), http=HttpClient())
    finite = [v for v in grid.values if v == v]
    assert finite and max(finite) > 2000.0


@pytest.mark.skipif(not os.environ.get("RUN_LIVE_3DEP"), reason="set RUN_LIVE_3DEP=1 for a live USGS 3DEP read")
async def test_live_3dep_reads_sierra_elevation():
    grid = await read_dem_grid(source="3dep", bbox=(-118.30, 36.575, -118.28, 36.585), http=HttpClient())
    finite = [v for v in grid.values if v == v]
    assert finite and max(finite) > 2000.0

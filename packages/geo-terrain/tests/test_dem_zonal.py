"""Tests for dem_zonal: per-zone elevation/slope/aspect over a DEM COG.

Build deterministic in-memory float DEM COGs with ``geo_common.testing`` and read
them through a recording byte-range reader (no network). Zones are polygons in
the DEM's own coordinate space (geotransform (0,1,0,H,0,-1), so world == pixel).
"""

from __future__ import annotations

import math

import pytest

from geo_common.errors import ValidationError
from geo_common.raster_models import Feature, FeatureCollection, GeoJSONGeometry
from geo_common.testing import RecordingByteRangeReader, build_cog

from geo_terrain.dem_zonal import dem_zonal

_GT = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)  # 4x4, 1-unit cells; cell (c,r) center (c+.5, 3.5-r)


def _dem(values):
    return build_cog(
        width=4, height=4, tile_width=4, tile_height=4,
        bands=[values], dtype="float32", geotransform=_GT,
    )


def _zone_top_left_2x2():
    ring = [[0.0, 2.0], [2.0, 2.0], [2.0, 4.0], [0.0, 4.0], [0.0, 2.0]]
    return FeatureCollection(
        features=[Feature(geometry=GeoJSONGeometry(type="Polygon", coordinates=[ring]),
                          properties={"zone_id": "block"})]
    )


async def test_elevation_zonal_matches_reference() -> None:
    values = [float(i) for i in range(16)]  # cells 0,1,4,5 in the zone
    result = await dem_zonal(
        dem_href="s3://amzn-s3-demo-bucket/dem.tif",
        zones=_zone_top_left_2x2(),
        measure="elevation",
        stats=["min", "max", "mean", "sum", "count"],
        reader=RecordingByteRangeReader(_dem(values)),
    )
    assert len(result) == 1
    assert result[0].no_data is False
    assert result[0].statistics == {"min": 0.0, "max": 5.0, "mean": 2.5, "sum": 10.0, "count": 4.0}


async def test_slope_zonal_on_east_ramp() -> None:
    # z increases 10 per column, 1-unit cells -> slope atan(10) everywhere.
    values = [float(c * 10) for _ in range(4) for c in range(4)]
    result = await dem_zonal(
        dem_href="s3://amzn-s3-demo-bucket/dem.tif",
        zones=_zone_top_left_2x2(),
        measure="slope",
        stats=["mean", "min", "max"],
        reader=RecordingByteRangeReader(_dem(values)),
    )
    expected = math.degrees(math.atan(10.0))
    assert result[0].statistics["mean"] == pytest.approx(expected, abs=1e-6)
    assert result[0].statistics["min"] == pytest.approx(expected, abs=1e-6)


async def test_aspect_zonal_on_east_ramp_faces_west() -> None:
    # Elevation rises east -> downslope faces west -> aspect 270.
    values = [float(c * 10) for _ in range(4) for c in range(4)]
    result = await dem_zonal(
        dem_href="s3://amzn-s3-demo-bucket/dem.tif",
        zones=_zone_top_left_2x2(),
        measure="aspect",
        stats=["mean"],
        reader=RecordingByteRangeReader(_dem(values)),
    )
    assert result[0].statistics["mean"] == pytest.approx(270.0, abs=1e-6)


async def test_dem_zonal_rejects_unknown_measure() -> None:
    with pytest.raises(ValidationError) as exc:
        await dem_zonal(
            dem_href="s3://amzn-s3-demo-bucket/dem.tif",
            zones=_zone_top_left_2x2(),
            measure="curvature",
            reader=RecordingByteRangeReader(_dem([float(i) for i in range(16)])),
        )
    assert exc.value.detail.get("parameter") == "measure"


async def test_dem_zonal_empty_zones_returns_empty() -> None:
    result = await dem_zonal(
        dem_href="s3://amzn-s3-demo-bucket/dem.tif",
        zones=FeatureCollection(features=[]),
        measure="elevation",
        reader=RecordingByteRangeReader(_dem([float(i) for i in range(16)])),
    )
    assert result == []


import asyncio

from hypothesis import given, settings
from hypothesis import strategies as st


def _full_grid_zone():
    ring = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]]
    return FeatureCollection(
        features=[Feature(geometry=GeoJSONGeometry(type="Polygon", coordinates=[ring]),
                          properties={"zone_id": "all"})]
    )


@pytest.mark.property
@settings(deadline=None)
@given(
    values=st.lists(
        st.floats(min_value=-500.0, max_value=9000.0, allow_nan=False, allow_infinity=False),
        min_size=16, max_size=16,
    )
)
def test_elevation_zonal_full_grid_matches_reference(values) -> None:
    """Feature: geospatial-power-pack — elevation zonal over a DEM COG.

    A full-grid zone over a random float DEM reduces to the mean/min/max/count of
    all cells, read back through the byte-range reader and shared reducer.
    """
    cog = _dem([float(v) for v in values])

    async def _run():
        return await dem_zonal(
            dem_href="s3://amzn-s3-demo-bucket/dem.tif",
            zones=_full_grid_zone(),
            measure="elevation",
            stats=["min", "max", "mean", "count"],
            reader=RecordingByteRangeReader(cog),
        )

    result = asyncio.run(_run())
    stats = result[0].statistics
    # float32 storage -> compare at float32 tolerance.
    assert stats["count"] == pytest.approx(16.0)
    assert stats["min"] == pytest.approx(min(values), rel=1e-3, abs=1e-2)
    assert stats["max"] == pytest.approx(max(values), rel=1e-3, abs=1e-2)
    assert stats["mean"] == pytest.approx(sum(values) / len(values), rel=1e-3, abs=1e-2)

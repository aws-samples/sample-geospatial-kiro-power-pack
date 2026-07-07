"""Tests for ``zonal_band_math`` — per-pixel multi-asset NDVI reduced to zones.

These exercise the capability that single-asset ``band_math`` cannot: computing
an index (NDVI) whose bands live in **separate** single-band COGs and reducing
the true per-pixel result to per-zone statistics. The synthetic assets are built
fully in memory (no network) and read through recording byte-range readers.
"""

from __future__ import annotations

import math

import pytest

from _cogbuild import RecordingByteRangeReader, build_cog

from geo_common.errors import ValidationError
from geo_raster import FeatureCollection, zonal_band_math

# North-up geotransform: origin (0, 4), 1-unit pixels, so a 4x4 grid covers
# x in [0, 4], y in [0, 4]; cell (col,row) center is (col+0.5, 3.5-row).
_GEOTRANSFORM = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)


def _single_band_cog(values: list, *, width: int = 4, height: int = 4, geotransform=_GEOTRANSFORM):
    return build_cog(
        width=width,
        height=height,
        tile_width=width,
        tile_height=height,
        bands=[values],
        dtype="float32",
        geotransform=geotransform,
    )


def _top_left_2x2_zone() -> FeatureCollection:
    """A zone covering x in [0,2], y in [2,4] -> the top-left 2x2 cell block."""
    return FeatureCollection(
        features=[
            {
                "type": "Feature",
                "properties": {"zone_id": "block"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0.0, 2.0], [2.0, 2.0], [2.0, 4.0], [0.0, 4.0], [0.0, 2.0]]],
                },
            }
        ]
    )


async def test_true_per_pixel_ndvi_zonal_stats() -> None:
    """NDVI is computed per pixel across two COGs, then reduced per zone."""
    # Red all 1.0; NIR varies. Cells in the zone are indices 0,1,4,5.
    red = [1.0] * 16
    nir = [0.0] * 16
    nir[0], nir[1], nir[4], nir[5] = 3.0, 4.0, 9.0, 19.0

    red_cog = _single_band_cog(red)
    nir_cog = _single_band_cog(nir)

    result = await zonal_band_math(
        assets={"B08": "s3://demo/B08.tif", "B04": "s3://demo/B04.tif"},
        expression="(B08 - B04) / (B08 + B04)",
        zones=_top_left_2x2_zone(),
        readers={8: RecordingByteRangeReader(nir_cog), 4: RecordingByteRangeReader(red_cog)},
    )

    assert len(result) == 1
    stats = result[0].statistics
    assert result[0].no_data is False
    # NDVI per cell: 0.5, 0.6, 0.8, 0.9
    assert stats["count"] == pytest.approx(4.0)
    assert stats["min"] == pytest.approx(0.5)
    assert stats["max"] == pytest.approx(0.9)
    assert stats["mean"] == pytest.approx(0.7)
    assert stats["sum"] == pytest.approx(2.8)


async def test_subset_of_stats_is_honored() -> None:
    red = [1.0] * 16
    nir = [0.0] * 16
    nir[0], nir[1], nir[4], nir[5] = 3.0, 4.0, 9.0, 19.0

    result = await zonal_band_math(
        assets={"B08": "s3://demo/B08.tif", "B04": "s3://demo/B04.tif"},
        expression="(B08 - B04) / (B08 + B04)",
        zones=_top_left_2x2_zone(),
        stats=["mean", "count"],
        readers={
            8: RecordingByteRangeReader(_single_band_cog(nir)),
            4: RecordingByteRangeReader(_single_band_cog(red)),
        },
    )
    assert set(result[0].statistics) == {"mean", "count"}
    assert result[0].statistics["mean"] == pytest.approx(0.7)


async def test_missing_asset_for_referenced_band_is_rejected() -> None:
    """An expression band with no matching asset raises before any read."""
    with pytest.raises(ValidationError) as exc:
        await zonal_band_math(
            assets={"B08": "s3://demo/B08.tif"},  # no B04
            expression="(B08 - B04) / (B08 + B04)",
            zones=_top_left_2x2_zone(),
        )
    assert "B4" in str(exc.value)


async def test_misaligned_grids_are_rejected() -> None:
    """Bands at different resolutions over the window are rejected, not fudged."""
    red = [1.0] * 16
    nir_coarse = [5.0, 6.0, 7.0, 8.0]  # 2x2 grid, 2-unit pixels

    red_cog = _single_band_cog(red)  # 4x4, 1-unit pixels
    nir_cog = build_cog(
        width=2,
        height=2,
        tile_width=2,
        tile_height=2,
        bands=[nir_coarse],
        dtype="float32",
        geotransform=(0.0, 2.0, 0.0, 4.0, 0.0, -2.0),
    )

    with pytest.raises(ValidationError) as exc:
        await zonal_band_math(
            assets={"B08": "s3://demo/B08.tif", "B04": "s3://demo/B04.tif"},
            expression="(B08 - B04) / (B08 + B04)",
            zones=_top_left_2x2_zone(),
            readers={8: RecordingByteRangeReader(nir_cog), 4: RecordingByteRangeReader(red_cog)},
        )
    assert "grid" in str(exc.value).lower() or "align" in str(exc.value).lower()


async def test_bad_band_token_is_rejected() -> None:
    with pytest.raises(ValidationError):
        await zonal_band_math(
            assets={"NIR": "s3://demo/B08.tif", "B04": "s3://demo/B04.tif"},
            expression="(B08 - B04) / (B08 + B04)",
            zones=_top_left_2x2_zone(),
        )


async def test_no_zones_returns_empty_without_reading() -> None:
    result = await zonal_band_math(
        assets={"B08": "s3://demo/B08.tif", "B04": "s3://demo/B04.tif"},
        expression="(B08 - B04) / (B08 + B04)",
        zones=FeatureCollection(features=[]),
    )
    assert result == []


async def test_nodata_pixels_excluded_from_stats() -> None:
    """Masked (nodata) NIR pixels drop out of the per-zone reduction."""
    red = [1.0] * 16
    nir = [0.0] * 16
    nir[0], nir[1], nir[4], nir[5] = 3.0, -9999.0, 9.0, 19.0

    nir_cog = build_cog(
        width=4,
        height=4,
        tile_width=4,
        tile_height=4,
        bands=[nir],
        dtype="float32",
        nodata=-9999.0,
        geotransform=_GEOTRANSFORM,
    )
    red_cog = _single_band_cog(red)

    result = await zonal_band_math(
        assets={"B08": "s3://demo/B08.tif", "B04": "s3://demo/B04.tif"},
        expression="(B08 - B04) / (B08 + B04)",
        zones=_top_left_2x2_zone(),
        readers={8: RecordingByteRangeReader(nir_cog), 4: RecordingByteRangeReader(red_cog)},
    )
    # One of the four cells is nodata -> count drops to 3.
    assert result[0].statistics["count"] == pytest.approx(3.0)
    assert result[0].statistics["max"] == pytest.approx(0.9)
    assert result[0].statistics["min"] == pytest.approx(0.5)

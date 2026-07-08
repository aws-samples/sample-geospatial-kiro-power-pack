"""Tests for ``band_math`` over windowed COG reads (Requirements 7.2, 12.1).

``band_math`` evaluates an NDVI/NDWI/NBR-style expression over a windowed read,
fetching only the referenced bands and only the overlapping tiles. These tests
verify the per-pixel arithmetic, the nodata / divide-by-zero handling, and that
only the referenced bands are read.
"""

from __future__ import annotations

import math

import pytest

from _cogbuild import RecordingByteRangeReader, build_cog

from geo_raster import band_math


def _const_band(width: int, height: int, value: float) -> list:
    return [value] * (width * height)


async def test_ndvi_expression_per_pixel() -> None:
    """NDVI = (B2 - B1) / (B2 + B1) is computed per pixel."""
    width = height = 16
    # B1 = red, B2 = nir. Use distinct per-pixel values.
    red = [float(1 + (r * width + c) % 5) for r in range(height) for c in range(width)]
    nir = [float(10 + (r * width + c) % 7) for r in range(height) for c in range(width)]
    data = build_cog(
        width=width,
        height=height,
        tile_width=8,
        tile_height=8,
        bands=[red, nir],
        dtype="float32",
    )

    reader = RecordingByteRangeReader(data)
    result = await band_math(
        asset_href="s3://amzn-s3-demo-bucket/scene.tif",
        expression="(B2 - B1) / (B2 + B1)",
        window={"col_off": 2, "row_off": 3, "width": 4, "height": 4},
        reader=reader,
    )

    assert result.band_indices == [0]
    assert (result.width, result.height) == (4, 4)
    for r in range(4):
        for c in range(4):
            i = (3 + r) * width + (2 + c)
            expected = (nir[i] - red[i]) / (nir[i] + red[i])
            assert result.value(0, r, c) == pytest.approx(expected)


async def test_division_by_zero_yields_nan_when_no_nodata() -> None:
    """A zero denominator yields NaN when the asset declares no nodata value."""
    width = height = 8
    zeros = _const_band(width, height, 0.0)
    data = build_cog(
        width=width,
        height=height,
        tile_width=8,
        tile_height=8,
        bands=[zeros, zeros],
        dtype="float32",
    )

    reader = RecordingByteRangeReader(data)
    result = await band_math(
        asset_href="s3://amzn-s3-demo-bucket/zero.tif",
        expression="(B2 - B1) / (B2 + B1)",
        window={"col_off": 0, "row_off": 0, "width": 2, "height": 2},
        reader=reader,
    )
    for v in result.band(0):
        assert math.isnan(v)


async def test_nodata_input_propagates_to_nodata_output() -> None:
    """A pixel whose input equals the asset nodata yields nodata, not arithmetic."""
    width = height = 8
    red = _const_band(width, height, -9999.0)  # all nodata
    nir = _const_band(width, height, 5.0)
    data = build_cog(
        width=width,
        height=height,
        tile_width=8,
        tile_height=8,
        bands=[red, nir],
        dtype="float32",
        nodata=-9999.0,
    )

    reader = RecordingByteRangeReader(data)
    result = await band_math(
        asset_href="s3://amzn-s3-demo-bucket/nd.tif",
        expression="(B2 - B1) / (B2 + B1)",
        window={"col_off": 0, "row_off": 0, "width": 3, "height": 3},
        reader=reader,
    )
    assert result.nodata == -9999.0
    for v in result.band(0):
        assert v == -9999.0


async def test_band_math_only_reads_referenced_bands() -> None:
    """An expression referencing one band of a 3-band asset reads only that band.

    With planar-separate layout, each band is stored in its own tiles, so a
    single-band expression must not fetch the other bands' tile byte ranges
    (Requirements 7.2, 12.1).
    """
    width = height = 32
    b1 = [float(i) for i in range(width * height)]
    b2 = [float(i + 1) for i in range(width * height)]
    b3 = [float(i + 2) for i in range(width * height)]
    data = build_cog(
        width=width,
        height=height,
        tile_width=16,
        tile_height=16,
        bands=[b1, b2, b3],
        dtype="float32",
        planar_config=2,
    )

    # Tile index layout for planar-separate: plane * tiles_per_plane + tile.
    from geo_raster import CogReader

    meta = await CogReader(RecordingByteRangeReader(data), source_id="m").open()
    tiles_per_plane = meta.tiles_per_plane
    band2_tile_offsets = {
        meta.tile_offsets[1 * tiles_per_plane + t] for t in range(tiles_per_plane)
    }

    reader = RecordingByteRangeReader(data)
    result = await band_math(
        asset_href="s3://amzn-s3-demo-bucket/multi.tif",
        expression="B2 * 2",
        window={"col_off": 0, "row_off": 0, "width": 4, "height": 4},
        reader=reader,
    )

    fetched = set(reader.fetched_offsets())
    band1_tile_offsets = {meta.tile_offsets[t] for t in range(tiles_per_plane)}
    band3_tile_offsets = {
        meta.tile_offsets[2 * tiles_per_plane + t] for t in range(tiles_per_plane)
    }
    # Only band 2's overlapping tile(s) fetched; bands 1 and 3 untouched.
    assert band2_tile_offsets & fetched
    assert not (band1_tile_offsets & fetched)
    assert not (band3_tile_offsets & fetched)
    assert result.value(0, 0, 1) == pytest.approx(b2[1] * 2)


async def test_zero_padded_band_tokens_resolve() -> None:
    """Zero-padded tokens (B01/B02, as Sentinel-2 uses B04/B08) evaluate correctly.

    The evaluator looks up the literal identifier, so the per-pixel env must be
    keyed by the exact token spelling in the expression, not a normalized
    ``B<int>``. Regression guard for the identifier-keyed env.
    """
    width = height = 8
    red = [float(1 + (r * width + c) % 5) for r in range(height) for c in range(width)]
    nir = [float(10 + (r * width + c) % 7) for r in range(height) for c in range(width)]
    data = build_cog(
        width=width,
        height=height,
        tile_width=8,
        tile_height=8,
        bands=[red, nir],
        dtype="float32",
    )

    reader = RecordingByteRangeReader(data)
    # Band 1 = red -> B01, band 2 = nir -> B02 (both zero-padded).
    result = await band_math(
        asset_href="s3://amzn-s3-demo-bucket/scene.tif",
        expression="(B02 - B01) / (B02 + B01)",
        window={"col_off": 0, "row_off": 0, "width": 4, "height": 4},
        reader=reader,
    )
    for r in range(4):
        for c in range(4):
            i = r * width + c
            expected = (nir[i] - red[i]) / (nir[i] + red[i])
            assert result.value(0, r, c) == pytest.approx(expected)

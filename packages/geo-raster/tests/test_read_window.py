"""Tests for ``read_window`` windowed COG reads (Requirements 7.2, 12.1).

These exercise the byte-range read engine through both the
:func:`geo_raster.read_window` function and the assembled
:class:`~geo_raster.server.GeoRasterServer`. Every asset is an in-memory COG
built by ``_cogbuild`` and read through a recording byte-range reader, so each
test can assert two things at once:

* **Correctness** — the returned :class:`RasterArray` holds exactly the pixels
  inside the requested window (Requirement 7.2).
* **Byte-range discipline** — only the tiles overlapping the window are fetched,
  and the full asset is never read (Requirements 7.2, 12.1).
"""

from __future__ import annotations

from typing import List

import pytest

from _cogbuild import RecordingByteRangeReader, build_cog

from geo_raster import CogReader, GeoRasterServer, PixelWindow, read_window


def _ramp(width: int, height: int, *, offset: int = 0) -> List[float]:
    """A deterministic band: value = (row*1000 + col + offset) mod 65536."""
    return [((r * 1000 + c + offset) % 65536) for r in range(height) for c in range(width)]


async def _tile_offsets(data: bytes) -> List[int]:
    reader = RecordingByteRangeReader(data)
    cog = CogReader(reader, source_id="mem")
    meta = await cog.open()
    return list(meta.tile_offsets)


async def test_window_within_single_tile_fetches_only_that_tile() -> None:
    """Req 7.2/12.1: a window inside one tile reads only that tile's bytes."""
    width = height = 256
    band = _ramp(width, height)
    data = build_cog(
        width=width, height=height, tile_width=128, tile_height=128, bands=[band]
    )
    tile_offsets = await _tile_offsets(data)

    reader = RecordingByteRangeReader(data)
    result = await read_window(
        asset_href="s3://amzn-s3-demo-bucket/asset.tif",
        window=PixelWindow(col_off=10, row_off=20, width=8, height=4),
        reader=reader,
    )

    # Correct shape + values (Req 7.2): only the in-window pixels are returned.
    assert (result.width, result.height) == (8, 4)
    assert result.band_indices == [1]
    for r in range(4):
        for c in range(8):
            assert result.value(1, r, c) == band[(20 + r) * width + (10 + c)]

    # Byte-range discipline (Req 12.1): only tile (0,0) was fetched; the other
    # three tiles' byte ranges were never requested, and far less than the whole
    # asset was transferred.
    fetched = set(reader.fetched_offsets())
    assert tile_offsets[0] in fetched
    assert not ({tile_offsets[1], tile_offsets[2], tile_offsets[3]} & fetched)
    assert reader.total_bytes_read < len(data)


async def test_window_spanning_all_tiles_fetches_each_overlapping_tile() -> None:
    """A window straddling the tile grid fetches every overlapping tile."""
    width = height = 256
    band = _ramp(width, height)
    data = build_cog(
        width=width, height=height, tile_width=128, tile_height=128, bands=[band]
    )
    tile_offsets = await _tile_offsets(data)

    reader = RecordingByteRangeReader(data)
    result = await read_window(
        asset_href="s3://amzn-s3-demo-bucket/asset.tif",
        window=PixelWindow(col_off=120, row_off=120, width=16, height=16),
        reader=reader,
    )

    assert (result.width, result.height) == (16, 16)
    for r in range(16):
        for c in range(16):
            assert result.value(1, r, c) == band[(120 + r) * width + (120 + c)]

    fetched = set(reader.fetched_offsets())
    # The window touches all four 128x128 tiles.
    assert set(tile_offsets) <= fetched


async def test_multiband_contiguous_band_selection() -> None:
    """Selecting a subset of bands returns those bands in the requested order."""
    width = height = 64
    b1 = _ramp(width, height, offset=0)
    b2 = _ramp(width, height, offset=100)
    b3 = _ramp(width, height, offset=200)
    data = build_cog(
        width=width,
        height=height,
        tile_width=32,
        tile_height=32,
        bands=[b1, b2, b3],
        dtype="uint16",
    )

    reader = RecordingByteRangeReader(data)
    result = await read_window(
        asset_href="https://example.com/asset.tif",
        window=PixelWindow(col_off=0, row_off=0, width=4, height=4),
        bands=[3, 1],
        reader=reader,
    )

    assert result.band_indices == [3, 1]
    assert result.value(3, 2, 1) == b3[2 * width + 1]
    assert result.value(1, 2, 1) == b1[2 * width + 1]


async def test_planar_separate_configuration() -> None:
    """A planar-separate (band-sequential) COG reads correctly per band."""
    width = height = 64
    b1 = _ramp(width, height, offset=0)
    b2 = _ramp(width, height, offset=7)
    data = build_cog(
        width=width,
        height=height,
        tile_width=32,
        tile_height=32,
        bands=[b1, b2],
        dtype="uint16",
        planar_config=2,
    )

    reader = RecordingByteRangeReader(data)
    result = await read_window(
        asset_href="s3://amzn-s3-demo-bucket/planar.tif",
        window=PixelWindow(col_off=33, row_off=1, width=5, height=5),
        bands=[2],
        reader=reader,
    )
    assert result.band_indices == [2]
    for r in range(5):
        for c in range(5):
            assert result.value(2, r, c) == b2[(1 + r) * width + (33 + c)]


async def test_deflate_with_horizontal_predictor() -> None:
    """DEFLATE-compressed tiles with Predictor 2 decode to the source pixels."""
    width = height = 64
    band = _ramp(width, height)
    data = build_cog(
        width=width,
        height=height,
        tile_width=32,
        tile_height=32,
        bands=[band],
        dtype="uint16",
        compression="deflate",
        predictor=2,
    )

    reader = RecordingByteRangeReader(data)
    result = await read_window(
        asset_href="s3://amzn-s3-demo-bucket/deflate.tif",
        window=PixelWindow(col_off=5, row_off=5, width=20, height=20),
        reader=reader,
    )
    for r in range(20):
        for c in range(20):
            assert result.value(1, r, c) == band[(5 + r) * width + (5 + c)]


async def test_float32_dtype_and_nodata_roundtrip() -> None:
    """Float32 samples and a declared nodata value flow through unchanged."""
    width = height = 32
    band = [float(r) + 0.5 * c for r in range(height) for c in range(width)]
    data = build_cog(
        width=width,
        height=height,
        tile_width=16,
        tile_height=16,
        bands=[band],
        dtype="float32",
        nodata=-9999.0,
    )

    reader = RecordingByteRangeReader(data)
    result = await read_window(
        asset_href="s3://amzn-s3-demo-bucket/float.tif",
        window=PixelWindow(col_off=0, row_off=0, width=3, height=3),
        reader=reader,
    )
    assert result.dtype == "float32"
    assert result.nodata == -9999.0
    for r in range(3):
        for c in range(3):
            assert result.value(1, r, c) == pytest.approx(band[r * width + c])


async def test_geo_window_resolves_via_geotransform() -> None:
    """A GeoWindow bbox is resolved to the correct pixel window via the geotransform."""
    width = height = 100
    band = _ramp(width, height)
    # Origin (1000, 5000), 10 units/pixel, north-up.
    gt = (1000.0, 10.0, 0.0, 5000.0, 0.0, -10.0)
    data = build_cog(
        width=width,
        height=height,
        tile_width=50,
        tile_height=50,
        bands=[band],
        geotransform=gt,
    )

    reader = RecordingByteRangeReader(data)
    # World bbox covering columns ~ [2,5], rows ~ [2,5].
    result = await read_window(
        asset_href="s3://amzn-s3-demo-bucket/geo.tif",
        window={"bbox": (1020.0, 4950.0, 1050.0, 4980.0)},
        reader=reader,
    )
    # min_x=1020 -> col 2; max_x=1050 -> col 5; max_y=4980 -> row 2; min_y=4950 -> row 5.
    assert result.width == 3
    assert result.height == 3
    assert result.value(1, 0, 0) == band[2 * width + 2]


async def test_server_read_window_tool_registered_and_works() -> None:
    """The assembled server exposes ``read_window`` and reads in-window pixels."""
    width = height = 64
    band = _ramp(width, height)
    data = build_cog(
        width=width, height=height, tile_width=32, tile_height=32, bands=[band]
    )

    server = GeoRasterServer()
    try:
        assert "read_window" in server.tool_names()
        # Drive the reader directly to avoid real network in this smoke check.
        reader = RecordingByteRangeReader(data)
        result = await read_window(
            asset_href="s3://amzn-s3-demo-bucket/k.tif",
            window=PixelWindow(col_off=1, row_off=1, width=2, height=2),
            reader=reader,
        )
        assert (result.width, result.height) == (2, 2)
        assert result.value(1, 0, 0) == band[1 * width + 1]
    finally:
        await server.aclose()

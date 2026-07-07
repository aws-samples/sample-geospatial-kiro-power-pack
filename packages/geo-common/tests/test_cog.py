"""Tests for the shared byte-range COG reader now living in geo-common.

These give the moved reader its own coverage (it is also exercised via
``geo-raster``'s re-export). They build deterministic in-memory COGs with
``geo_common.testing`` and read windows through :class:`geo_common.cog.CogReader`,
including a floating-point predictor (3) asset.
"""

from __future__ import annotations

import pytest

from geo_common.cog import CogReader, HttpRangeReader
from geo_common.testing import RecordingByteRangeReader, build_cog

_GT = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)


async def test_reads_uint16_cog_window_across_tiles() -> None:
    values = [i for i in range(16)]
    cog = build_cog(
        width=4, height=4, tile_width=2, tile_height=2,
        bands=[values], dtype="uint16", geotransform=_GT,
    )
    reader = CogReader(RecordingByteRangeReader(cog), source_id="mem")
    meta = await reader.open()
    assert (meta.width, meta.height) == (4, 4)
    band = await reader.read_window_pixels(col_off=0, row_off=0, width=4, height=4, bands=[1])
    assert band[0] == pytest.approx(values)


async def test_reads_float32_predictor3_cog() -> None:
    values = [float(i) * 0.5 - 3.0 for i in range(16)]
    cog = build_cog(
        width=4, height=4, tile_width=4, tile_height=4,
        bands=[values], dtype="float32", geotransform=_GT,
        compression="deflate", predictor=3,
    )
    reader = CogReader(RecordingByteRangeReader(cog), source_id="mem")
    await reader.open()
    band = await reader.read_window_pixels(col_off=0, row_off=0, width=4, height=4, bands=[1])
    assert band[0] == pytest.approx(values, rel=1e-3, abs=1e-3)


def test_http_range_reader_resolves_s3_to_https() -> None:
    """The shared HttpRangeReader resolves s3:// to a virtual-hosted HTTPS host."""
    r = HttpRangeReader("s3://amzn-s3-demo-bucket/path/to/asset.tif", http=None, region="us-west-2")
    assert r._url == "https://amzn-s3-demo-bucket.s3.us-west-2.amazonaws.com/path/to/asset.tif"

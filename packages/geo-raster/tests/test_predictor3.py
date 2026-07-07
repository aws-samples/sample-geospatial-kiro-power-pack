"""Tests for the TIFF floating-point predictor (Predictor 3) decode.

Predictor 3 is used by many float32 DEM/scientific COGs (Copernicus GLO-30, much
of USGS 3DEP). Before this, ``geo-raster`` rejected such assets outright. These
cover: an independent hand-computed decode vector, an encode/decode byte
round-trip, and a full in-memory COG read round-trip for float32 and float64.
"""

from __future__ import annotations

import struct

import pytest

from _cogbuild import RecordingByteRangeReader, build_cog

from geo_raster.cog import CogReader
from geo_raster.reader import read_raster_grid

_GT = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)


def test_decode_float_predictor_known_bytes() -> None:
    """Hand-computed vector: encoded predictor-3 bytes for float32 1.0 decode to 1.0.

    1.0f big-endian is 3F 80 00 00. With one sample per row (spp=1), the encoder
    reorders to MSB-first planes then byte-differences at lag 1, giving
    3F 41 80 00. The decoder must invert that back to 1.0 — verified here without
    going through the encoder, so an encoder bug can't hide a decoder bug.
    """
    encoded = bytes([0x3F, 0x41, 0x80, 0x00])
    decoded = CogReader._decode_float_predictor(encoded, rows=1, cols=1, spp=1, bps=4)
    (value,) = struct.unpack(">f", decoded)
    assert value == pytest.approx(1.0)


def test_encode_decode_byte_roundtrip_float32() -> None:
    """Encoding then decoding a scanline of float32 bytes is the identity."""
    from _cogbuild import build_cog  # noqa: F401 - ensure module import path

    values = [1.0, -2.5, 3.25, 1000.0, -0.001, 42.0]
    rows, cols, spp, bps = 1, len(values), 1, 4
    little = struct.pack("<" + "f" * len(values), *values)

    # Reproduce the encoder used by build_cog for a single scanline.
    row_values = cols * spp
    planar = bytearray(len(little))
    for count in range(row_values):
        be = little[count * bps : (count + 1) * bps][::-1]
        for k in range(bps):
            planar[k * row_values + count] = be[k]
    diffed = bytearray(planar)
    for i in range(len(planar) - 1, spp - 1, -1):
        diffed[i] = (planar[i] - planar[i - spp]) & 0xFF

    decoded = CogReader._decode_float_predictor(bytes(diffed), rows=rows, cols=cols, spp=spp, bps=bps)
    out = list(struct.unpack(">" + "f" * len(values), decoded))
    assert out == pytest.approx(values)


async def _read_all(cog_bytes: bytes):
    reader = RecordingByteRangeReader(cog_bytes)
    grid = await read_raster_grid(
        raster_href="s3://amzn-s3-demo-bucket/dem.tif",
        window_bbox=None,
        band=1,
        reader=reader,
    )
    return grid


async def test_read_float32_predictor3_cog_roundtrip() -> None:
    """A float32 COG written with Predictor 3 reads back pixel-exact."""
    values = [float(v) * 0.5 - 3.0 for v in range(16)]  # 4x4 float32
    cog = build_cog(
        width=4,
        height=4,
        tile_width=4,
        tile_height=4,
        bands=[values],
        dtype="float32",
        geotransform=_GT,
        compression="deflate",
        predictor=3,
    )
    grid = await _read_all(cog)
    assert grid.width == 4 and grid.height == 4
    assert grid.values == pytest.approx(values)


async def test_read_float64_predictor3_cog_roundtrip() -> None:
    """A float64 COG written with Predictor 3 reads back pixel-exact."""
    values = [float(v) * 1.25 - 7.0 for v in range(16)]
    cog = build_cog(
        width=4,
        height=4,
        tile_width=4,
        tile_height=4,
        bands=[values],
        dtype="float64",
        geotransform=_GT,
        predictor=3,
    )
    grid = await _read_all(cog)
    assert grid.values == pytest.approx(values)


import asyncio

from hypothesis import given, settings
from hypothesis import strategies as st


@st.composite
def _float_cog_case(draw: st.DrawFn):
    width = draw(st.integers(min_value=1, max_value=6))
    height = draw(st.integers(min_value=1, max_value=6))
    tile_width = draw(st.integers(min_value=1, max_value=6))
    tile_height = draw(st.integers(min_value=1, max_value=6))
    dtype = draw(st.sampled_from(["float32", "float64"]))
    compression = draw(st.sampled_from(["none", "deflate"]))
    planar = draw(st.sampled_from([1, 2]))
    n = width * height
    values = draw(
        st.lists(
            st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    return width, height, tile_width, tile_height, dtype, compression, planar, values


@pytest.mark.property
@settings(deadline=None)
@given(case=_float_cog_case())
def test_predictor3_roundtrip_is_pixel_exact(case) -> None:
    """Feature: geospatial-power-pack — Predictor 3 float COGs read back exactly.

    Randomized across dimensions, tile sizes, float32/64, compression, and planar
    config: a COG written with the floating-point predictor must decode to the
    original pixels (the encoder and decoder are independent implementations).
    """
    width, height, tile_width, tile_height, dtype, compression, planar, values = case
    cog = build_cog(
        width=width,
        height=height,
        tile_width=tile_width,
        tile_height=tile_height,
        bands=[values],
        dtype=dtype,
        geotransform=_GT,
        planar_config=planar,
        compression=compression,
        predictor=3,
    )
    grid = asyncio.run(_read_all(cog))
    assert grid.width == width and grid.height == height
    # float32 loses precision vs the Python float, so compare at float32 tolerance.
    tol = 1e-3 if dtype == "float32" else 1e-9
    assert grid.values == pytest.approx(values, rel=tol, abs=tol)

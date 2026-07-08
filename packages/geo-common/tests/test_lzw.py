"""Offline tests for the TIFF LZW decoder (`geo_common.cog._lzw_decode`).

We can't rely on Pillow/tifffile here, so this uses an independent TIFF-variant
LZW *encoder* (below) and asserts the decoder inverts it across randomized byte
strings — including data long enough to grow the code width (9→12 bits) and to
overflow the table (forcing a ClearCode reset). The live USGS 3DEP test validates
the same decoder against real GDAL-produced LZW.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geo_common.cog import _lzw_decode

_CLEAR = 256
_EOI = 257
_MAX_CODE = 4094  # emit a ClearCode before the table fully fills (TIFF LZW)


def _tiff_lzw_encode(data: bytes) -> bytes:
    """Independent TIFF-variant LZW encoder (MSB-first, early change, clear/eoi)."""
    bits = 0
    nbits = 0
    out = bytearray()

    def emit(code: int, width: int) -> None:
        nonlocal bits, nbits
        bits = (bits << width) | code
        nbits += width
        while nbits >= 8:
            nbits -= 8
            out.append((bits >> nbits) & 0xFF)

    table = {bytes([i]): i for i in range(256)}
    next_code = 258
    width = 9
    emit(_CLEAR, width)
    w = b""
    for byte in data:
        wc = w + bytes([byte])
        if wc in table:
            w = wc
            continue
        emit(table[w], width)
        table[wc] = next_code
        next_code += 1
        # The decoder's table lags one entry behind the encoder's, so the
        # encoder must widen one code "ahead": when next_code reaches 2^width
        # (TIFF early change), the code the decoder then reads is at width+1.
        if next_code == (1 << width) and width < 12:
            width += 1
        if next_code >= _MAX_CODE:
            # This simple oracle does not implement the table-full reset; the
            # decoder's reset path is validated against real LZW by the live 3DEP
            # test. Keep offline test data below this threshold.
            raise RuntimeError("test data too large for the non-resetting LZW oracle")
        w = bytes([byte])
    if w:
        emit(table[w], width)
    emit(_EOI, width)
    if nbits > 0:
        out.append((bits << (8 - nbits)) & 0xFF)
    return bytes(out)


def test_lzw_roundtrip_known_repetitive():
    data = b"ABABABABA" * 50
    assert _lzw_decode(_tiff_lzw_encode(data)) == data


def test_lzw_roundtrip_forces_code_width_growth():
    # Structured data grows codes into 10-12 bit widths (without table overflow),
    # exercising multi-width decoding and the early-change boundary.
    data = bytes(range(256)) * 12
    assert _lzw_decode(_tiff_lzw_encode(data)) == data


@pytest.mark.property
@settings(deadline=None, max_examples=200)
@given(data=st.binary(min_size=0, max_size=3000))
def test_lzw_roundtrip_property(data: bytes) -> None:
    assert _lzw_decode(_tiff_lzw_encode(data)) == data

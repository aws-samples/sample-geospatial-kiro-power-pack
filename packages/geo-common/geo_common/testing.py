"""In-memory Cloud-Optimized GeoTIFF builder + recording reader for tests.

These helpers let the ``geo-raster`` windowed-read tests construct deterministic,
fully in-memory tiled GeoTIFFs (no real files, no network) and read them through a
:class:`RecordingByteRangeReader` that records every byte range requested. That
recording is what lets the tests assert the core windowed-read contract: only
the tiles overlapping the requested window are fetched, and the full asset is
never read (Requirements 7.2, 12.1).

The builder produces a classic little-endian TIFF with the tags
``geo_raster.cog.CogReader`` consumes. It supports the cases the reader
supports: tiled layout, uncompressed or DEFLATE tiles (optionally with the
horizontal predictor), contiguous or separate planar configuration, and
8/16/32-bit integer or 32/64-bit float samples.
"""

from __future__ import annotations

import math
import struct
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

# dtype name -> (struct char, bits, sample_format)
_DTYPE = {
    "uint8": ("B", 8, 1),
    "uint16": ("H", 16, 1),
    "uint32": ("I", 32, 1),
    "int16": ("h", 16, 2),
    "int32": ("i", 32, 2),
    "float32": ("f", 32, 3),
    "float64": ("d", 64, 3),
}

# TIFF type -> byte size for the types this builder emits.
_TYPE_SIZE = {2: 1, 3: 2, 4: 4, 11: 4, 12: 8}


def build_cog(
    *,
    width: int,
    height: int,
    tile_width: int,
    tile_height: int,
    bands: Sequence[Sequence[float]],
    dtype: str = "uint16",
    nodata: Optional[float] = None,
    geotransform: Optional[Tuple[float, float, float, float, float, float]] = None,
    planar_config: int = 1,
    compression: str = "none",
    predictor: int = 1,
) -> bytes:
    """Build a tiled GeoTIFF as bytes.

    ``bands`` is one flat row-major sequence (length ``width*height``) per band.
    """
    char, bits, sample_format = _DTYPE[dtype]
    byte_size = bits // 8
    spp = len(bands)
    endian = "<"

    tiles_across = math.ceil(width / tile_width)
    tiles_down = math.ceil(height / tile_height)
    per_tile = tile_width * tile_height

    def sample(band: int, row: int, col: int) -> float:
        if row >= height or col >= width:
            return 0.0  # tile padding
        return bands[band][row * width + col]

    # --- Encode tile payloads (raw, then optional predictor + compression) ---
    def pack_tile(values: List[float]) -> bytes:
        raw = struct.pack(endian + char * len(values), *values)
        return raw

    def apply_predictor(values: List[float], samples: int) -> List[float]:
        if predictor != 2:
            return values
        mask = (1 << bits) - 1 if sample_format != 3 else None
        out = list(values)
        for r in range(tile_height):
            base = r * tile_width * samples
            for s in range(samples):
                for c in range(tile_width - 1, 0, -1):
                    cur = base + c * samples + s
                    prev = base + (c - 1) * samples + s
                    diff = out[cur] - out[prev]
                    if mask is not None:
                        diff &= mask
                    out[cur] = diff
        return out

    def encode_float_predictor(raw: bytes, samples: int) -> bytes:
        """Encode TIFF floating-point predictor (3): plane reorder + byte diff.

        Inverse of ``CogReader._decode_float_predictor``: reorder each sample's
        bytes into byte planes (plane 0 = most-significant byte) then apply
        horizontal byte differencing at lag ``samples``, per scanline.
        """
        row_values = tile_width * samples
        row_bytes = row_values * byte_size
        out = bytearray(len(raw))
        for r in range(tile_height):
            start = r * row_bytes
            row = raw[start : start + row_bytes]
            planar = bytearray(row_bytes)
            for count in range(row_values):
                sample_bytes = row[count * byte_size : (count + 1) * byte_size]
                be = sample_bytes[::-1] if endian == "<" else bytes(sample_bytes)
                for k in range(byte_size):
                    planar[k * row_values + count] = be[k]
            diffed = bytearray(planar)
            for i in range(row_bytes - 1, samples - 1, -1):
                diffed[i] = (planar[i] - planar[i - samples]) & 0xFF
            out[start : start + row_bytes] = diffed
        return bytes(out)

    def maybe_compress(raw: bytes) -> bytes:
        if compression == "deflate":
            return zlib.compress(raw)
        return raw

    def finish_tile(vals: List[float], samples: int) -> bytes:
        vals = apply_predictor(vals, samples)  # predictor 2 (integer) only
        raw = pack_tile(vals)
        if predictor == 3:
            raw = encode_float_predictor(raw, samples)
        return maybe_compress(raw)

    tile_payloads: List[bytes] = []
    if planar_config == 1:
        for tr in range(tiles_down):
            for tc in range(tiles_across):
                vals: List[float] = []
                for pr in range(tile_height):
                    for pc in range(tile_width):
                        for s in range(spp):
                            vals.append(
                                sample(s, tr * tile_height + pr, tc * tile_width + pc)
                            )
                tile_payloads.append(finish_tile(vals, spp))
    else:  # planar separate
        for s in range(spp):
            for tr in range(tiles_down):
                for tc in range(tiles_across):
                    vals = []
                    for pr in range(tile_height):
                        for pc in range(tile_width):
                            vals.append(
                                sample(s, tr * tile_height + pr, tc * tile_width + pc)
                            )
                    tile_payloads.append(finish_tile(vals, 1))

    n_tiles = len(tile_payloads)
    compression_code = 8 if compression == "deflate" else 1

    # --- Assemble IFD tags (must be in ascending tag order) ------------------
    # Each entry: (tag, type, count, data_bytes). data_bytes is the packed
    # array of values; if <= 4 bytes it is stored inline, else externally.
    tags: List[Tuple[int, int, int, bytes]] = []

    def arr(typ: int, values: Sequence[int]) -> bytes:
        if typ == 3:
            return struct.pack(endian + "H" * len(values), *values)
        if typ == 4:
            return struct.pack(endian + "I" * len(values), *values)
        if typ == 12:
            return struct.pack(endian + "d" * len(values), *values)
        raise ValueError(typ)

    tags.append((256, 4, 1, arr(4, [width])))
    tags.append((257, 4, 1, arr(4, [height])))
    tags.append((258, 3, spp, arr(3, [bits] * spp)))
    tags.append((259, 3, 1, arr(3, [compression_code])))
    tags.append((277, 3, 1, arr(3, [spp])))
    tags.append((284, 3, 1, arr(3, [planar_config])))
    if predictor != 1:
        tags.append((317, 3, 1, arr(3, [predictor])))
    tags.append((322, 4, 1, arr(4, [tile_width])))
    tags.append((323, 4, 1, arr(4, [tile_height])))
    # TileOffsets (324) + TileByteCounts (325): filled after layout below.
    tags.append((324, 4, n_tiles, b"\x00" * (4 * n_tiles)))
    tags.append((325, 4, n_tiles, arr(4, [len(p) for p in tile_payloads])))
    tags.append((339, 3, spp, arr(3, [sample_format] * spp)))
    if geotransform is not None:
        origin_x, px_w, _r, origin_y, _c, px_h = geotransform
        tags.append((33550, 12, 3, arr(12, [px_w, abs(px_h), 0.0])))
        tags.append((33922, 12, 6, arr(12, [0.0, 0.0, 0.0, origin_x, origin_y, 0.0])))
    if nodata is not None:
        text = f"{nodata}".encode("ascii") + b"\x00"
        tags.append((42113, 2, len(text), text))

    tags.sort(key=lambda t: t[0])
    n_tags = len(tags)

    ifd_offset = 8
    ifd_size = 2 + 12 * n_tags + 4
    ext_start = ifd_offset + ifd_size

    # Assign external offsets for tags whose data exceeds 4 bytes.
    ext_offsets: Dict[int, int] = {}
    cursor = ext_start
    for tag, _typ, _count, data in tags:
        if len(data) > 4:
            ext_offsets[tag] = cursor
            cursor += len(data)
            if cursor % 2:  # word-align external data
                cursor += 1
    tile_data_start = cursor

    # Compute tile offsets now that the data region start is known.
    tile_offsets: List[int] = []
    pos = tile_data_start
    for payload in tile_payloads:
        tile_offsets.append(pos)
        pos += len(payload)
    tile_offsets_data = arr(4, tile_offsets)

    # --- Serialize -----------------------------------------------------------
    out = bytearray()
    out += b"II"
    out += struct.pack(endian + "H", 42)
    out += struct.pack(endian + "I", ifd_offset)
    out += struct.pack(endian + "H", n_tags)

    ext_blocks: List[Tuple[int, bytes]] = []
    for tag, typ, count, data in tags:
        if tag == 324:
            data = tile_offsets_data
        out += struct.pack(endian + "H", tag)
        out += struct.pack(endian + "H", typ)
        out += struct.pack(endian + "I", count)
        if len(data) <= 4:
            out += data + b"\x00" * (4 - len(data))
        else:
            out += struct.pack(endian + "I", ext_offsets[tag])
            ext_blocks.append((ext_offsets[tag], data))
    out += struct.pack(endian + "I", 0)  # next IFD offset = 0

    # External data blocks, in their assigned offset order.
    for offset, data in sorted(ext_blocks, key=lambda b: b[0]):
        if len(out) < offset:
            out += b"\x00" * (offset - len(out))
        out += data

    # Tile data.
    if len(out) < tile_data_start:
        out += b"\x00" * (tile_data_start - len(out))
    for payload in tile_payloads:
        out += payload

    return bytes(out)


class RecordingByteRangeReader:
    """An in-memory :class:`~geo_raster.cog.ByteRangeReader` that records reads.

    Backed by a ``bytes`` buffer; :meth:`read_range` returns exactly the
    requested slice and appends ``(offset, length)`` to :attr:`reads` so tests
    can assert which byte ranges (and therefore which tiles) were transferred.
    """

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.reads: List[Tuple[int, int]] = []

    async def read_range(self, offset: int, length: int) -> bytes:
        self.reads.append((offset, length))
        return self._data[offset : offset + length]

    @property
    def total_bytes_read(self) -> int:
        return sum(length for _, length in self.reads)

    def fetched_offsets(self) -> List[int]:
        return [offset for offset, _ in self.reads]

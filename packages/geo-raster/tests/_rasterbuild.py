"""In-memory tiled GeoTIFF builder + recording reader for geo-raster tests.

These helpers let the ``geo-raster`` tests construct deterministic, fully
in-memory georeferenced GeoTIFFs (no real files, no network) and read them
through a :class:`RecordingByteRangeReader` that records every byte range
requested. That lets the read+compute test assert the byte-range discipline
(only the tiles overlapping the zones' window are fetched) alongside statistic
correctness.

The builder produces a classic little-endian tiled TIFF with the tags
:class:`geo_raster.cog.CogReader` consumes (uncompressed, contiguous, single or
multi band, with an optional GDAL nodata and a model-pixel-scale / tiepoint
geotransform).
"""

from __future__ import annotations

import math
import struct
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


def build_cog(
    *,
    width: int,
    height: int,
    tile_width: int,
    tile_height: int,
    bands: Sequence[Sequence[float]],
    dtype: str = "float64",
    nodata: Optional[float] = None,
    geotransform: Optional[Tuple[float, float, float, float, float, float]] = None,
) -> bytes:
    """Build a tiled, uncompressed, contiguous GeoTIFF as bytes.

    ``bands`` is one flat row-major sequence (length ``width*height``) per band.
    ``geotransform`` is a GDAL-style ``(origin_x, px_w, 0, origin_y, 0, px_h)``.
    """
    char, bits, sample_format = _DTYPE[dtype]
    spp = len(bands)
    endian = "<"

    tiles_across = math.ceil(width / tile_width)
    tiles_down = math.ceil(height / tile_height)

    def sample(band: int, row: int, col: int) -> float:
        if row >= height or col >= width:
            return 0.0  # tile padding
        return bands[band][row * width + col]

    tile_payloads: List[bytes] = []
    for tr in range(tiles_down):
        for tc in range(tiles_across):
            vals: List[float] = []
            for pr in range(tile_height):
                for pc in range(tile_width):
                    for s in range(spp):
                        vals.append(
                            sample(s, tr * tile_height + pr, tc * tile_width + pc)
                        )
            tile_payloads.append(struct.pack(endian + char * len(vals), *vals))

    n_tiles = len(tile_payloads)

    tags: List[Tuple[int, int, int, bytes]] = []

    def arr(typ: int, values: Sequence[float]) -> bytes:
        if typ == 3:
            return struct.pack(endian + "H" * len(values), *[int(v) for v in values])
        if typ == 4:
            return struct.pack(endian + "I" * len(values), *[int(v) for v in values])
        if typ == 12:
            return struct.pack(endian + "d" * len(values), *[float(v) for v in values])
        raise ValueError(typ)

    tags.append((256, 4, 1, arr(4, [width])))
    tags.append((257, 4, 1, arr(4, [height])))
    tags.append((258, 3, spp, arr(3, [bits] * spp)))
    tags.append((259, 3, 1, arr(3, [1])))  # compression none
    tags.append((277, 3, 1, arr(3, [spp])))
    tags.append((284, 3, 1, arr(3, [1])))  # planar contiguous
    tags.append((322, 4, 1, arr(4, [tile_width])))
    tags.append((323, 4, 1, arr(4, [tile_height])))
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

    ext_offsets: Dict[int, int] = {}
    cursor = ext_start
    for tag, _typ, _count, data in tags:
        if len(data) > 4:
            ext_offsets[tag] = cursor
            cursor += len(data)
            if cursor % 2:
                cursor += 1
    tile_data_start = cursor

    tile_offsets: List[int] = []
    pos = tile_data_start
    for payload in tile_payloads:
        tile_offsets.append(pos)
        pos += len(payload)
    tile_offsets_data = arr(4, tile_offsets)

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
    out += struct.pack(endian + "I", 0)

    for offset, data in sorted(ext_blocks, key=lambda b: b[0]):
        if len(out) < offset:
            out += b"\x00" * (offset - len(out))
        out += data

    if len(out) < tile_data_start:
        out += b"\x00" * (tile_data_start - len(out))
    for payload in tile_payloads:
        out += payload

    return bytes(out)


class RecordingByteRangeReader:
    """In-memory :class:`~geo_raster.cog.ByteRangeReader` that records reads."""

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

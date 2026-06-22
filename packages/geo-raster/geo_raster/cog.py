"""A self-contained Cloud-Optimized GeoTIFF (COG) windowed reader.

This module implements the byte-range read engine behind ``geo-raster``'s
``zonal_statistics`` raster reads (Requirements 8.7, 12.6). It parses just
enough of the TIFF / COG structure — the header, the first IFD, and the
tile-offset / tile-byte-count arrays — to locate the **individual tiles that
overlap a requested window**, and then fetches **only those tiles' byte
ranges** through an injected :class:`ByteRangeReader`. The full asset is never
read into memory or copied to local storage: metadata reads are a handful of
small ranges, and pixel reads are limited to the overlapping tiles.

Scope (kept deliberately focused on the COG common case):

* Classic TIFF (little- or big-endian); BigTIFF is reported as unsupported.
* **Tiled** organization (``TileWidth``/``TileLength``); striped TIFFs are
  reported as unsupported, since COGs are tiled.
* Uncompressed and DEFLATE/zlib-compressed tiles, with no predictor or the
  horizontal predictor (Predictor 2).
* ``PlanarConfiguration`` 1 (contiguous/chunky) and 2 (separate planes).
* Integer (8/16/32-bit, signed or unsigned) and floating-point (32/64-bit)
  sample formats.

Anything outside that scope raises an :class:`~geo_common.errors.UpstreamError`
that names the unsupported asset property, so the failure is attributable rather
than silent.

The reader is fully async and performs **no** real I/O itself — it calls
``await reader.read_range(offset, length)`` — which keeps it testable with an
in-memory recording reader and usable in production over an HTTP/S3 range
reader. This module is intentionally self-contained (depends only on the shared
``Error_Taxonomy``) so ``geo-raster`` can read rasters without pulling in any
other server package.
"""

from __future__ import annotations

import math
import struct
import zlib
from typing import Dict, List, Optional, Protocol, Tuple

from geo_common.errors import UpstreamError, ValidationError

__all__ = ["ByteRangeReader", "CogMetadata", "CogReader"]


class ByteRangeReader(Protocol):
    """Reads an exact byte range ``[offset, offset+length)`` from an asset.

    Implementations fetch the range from wherever the asset lives (S3 / HTTP
    via a ``Range`` request, or an in-memory buffer in tests) and return exactly
    those bytes. The reader must not transfer bytes outside the requested range.
    """

    async def read_range(self, offset: int, length: int) -> bytes:  # pragma: no cover - protocol
        ...


# TIFF field type -> (struct char, byte size). ASCII/RATIONAL handled specially.
_TYPE_STRUCT: Dict[int, Tuple[str, int]] = {
    1: ("B", 1),   # BYTE
    2: ("s", 1),   # ASCII (handled specially)
    3: ("H", 2),   # SHORT
    4: ("I", 4),   # LONG
    5: ("I", 8),   # RATIONAL (two LONGs)
    6: ("b", 1),   # SBYTE
    7: ("B", 1),   # UNDEFINED
    8: ("h", 2),   # SSHORT
    9: ("i", 4),   # SLONG
    10: ("i", 8),  # SRATIONAL (two SLONGs)
    11: ("f", 4),  # FLOAT
    12: ("d", 8),  # DOUBLE
}

# TIFF tag numbers used by this reader.
_TAG_IMAGE_WIDTH = 256
_TAG_IMAGE_LENGTH = 257
_TAG_BITS_PER_SAMPLE = 258
_TAG_COMPRESSION = 259
_TAG_SAMPLES_PER_PIXEL = 277
_TAG_PLANAR_CONFIG = 284
_TAG_PREDICTOR = 317
_TAG_TILE_WIDTH = 322
_TAG_TILE_LENGTH = 323
_TAG_TILE_OFFSETS = 324
_TAG_TILE_BYTE_COUNTS = 325
_TAG_SAMPLE_FORMAT = 339
_TAG_MODEL_PIXEL_SCALE = 33550
_TAG_MODEL_TIEPOINT = 33922
_TAG_MODEL_TRANSFORMATION = 34264
_TAG_GDAL_NODATA = 42113

_COMPRESSION_NONE = 1
_COMPRESSION_DEFLATE = (8, 32946)  # Adobe Deflate / Deflate

_SAMPLE_FORMAT_UINT = 1
_SAMPLE_FORMAT_INT = 2
_SAMPLE_FORMAT_FLOAT = 3


class CogMetadata:
    """Parsed COG metadata for the first image (IFD).

    Carries the grid geometry (image + tile dimensions), sample description
    (count, bit depth, sample format), tile index (offsets + byte counts),
    optional nodata value, and optional georeferencing (a 6-element GDAL-style
    geotransform) used to resolve a georeferenced window.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        tile_width: int,
        tile_height: int,
        samples_per_pixel: int,
        bits_per_sample: int,
        sample_format: int,
        compression: int,
        predictor: int,
        planar_config: int,
        tile_offsets: List[int],
        tile_byte_counts: List[int],
        nodata: Optional[float],
        geotransform: Optional[Tuple[float, float, float, float, float, float]],
    ) -> None:
        self.width = width
        self.height = height
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.samples_per_pixel = samples_per_pixel
        self.bits_per_sample = bits_per_sample
        self.sample_format = sample_format
        self.compression = compression
        self.predictor = predictor
        self.planar_config = planar_config
        self.tile_offsets = tile_offsets
        self.tile_byte_counts = tile_byte_counts
        self.nodata = nodata
        self.geotransform = geotransform

    @property
    def tiles_across(self) -> int:
        return math.ceil(self.width / self.tile_width)

    @property
    def tiles_down(self) -> int:
        return math.ceil(self.height / self.tile_height)

    @property
    def tiles_per_plane(self) -> int:
        return self.tiles_across * self.tiles_down

    @property
    def dtype(self) -> str:
        """A short dtype name, e.g. ``"uint16"`` / ``"float32"``."""
        if self.sample_format == _SAMPLE_FORMAT_FLOAT:
            return f"float{self.bits_per_sample}"
        if self.sample_format == _SAMPLE_FORMAT_INT:
            return f"int{self.bits_per_sample}"
        return f"uint{self.bits_per_sample}"


class CogReader:
    """Reads windows from a COG via an injected :class:`ByteRangeReader`.

    Usage::

        reader = CogReader(byte_range_reader, source_id="s3://amzn-s3-demo-bucket/key.tif")
        meta = await reader.open()
        block = await reader.read_window_pixels(
            col_off=10, row_off=20, width=64, height=64, bands=[1]
        )
    """

    def __init__(self, source: ByteRangeReader, *, source_id: str) -> None:
        self._source = source
        self._source_id = source_id
        self._endian: str = "<"
        self._meta: Optional[CogMetadata] = None

    @property
    def metadata(self) -> CogMetadata:
        if self._meta is None:
            raise RuntimeError("CogReader.open() must be awaited before use")
        return self._meta

    # ------------------------------------------------------------------
    # Header + IFD parsing (small metadata byte-range reads only)
    # ------------------------------------------------------------------

    async def open(self) -> CogMetadata:
        """Parse the TIFF header and first IFD; return :class:`CogMetadata`."""
        head = await self._source.read_range(0, 8)
        if len(head) < 8:
            raise UpstreamError(
                "asset is too small to be a valid TIFF", source=self._source_id
            )
        order = head[:2]
        if order == b"II":
            self._endian = "<"
        elif order == b"MM":
            self._endian = ">"
        else:
            raise UpstreamError(
                "asset is not a TIFF (bad byte-order marker)", source=self._source_id
            )
        magic = struct.unpack(self._endian + "H", head[2:4])[0]
        if magic == 43:
            raise UpstreamError(
                "BigTIFF assets are not supported by geo-raster",
                source=self._source_id,
            )
        if magic != 42:
            raise UpstreamError(
                "asset is not a classic TIFF", source=self._source_id
            )
        ifd_offset = struct.unpack(self._endian + "I", head[4:8])[0]
        tags = await self._read_ifd(ifd_offset)
        self._meta = self._build_metadata(tags)
        return self._meta

    async def _read_ifd(self, ifd_offset: int) -> Dict[int, List[object]]:
        """Read one IFD into a ``{tag: [values]}`` mapping."""
        count_raw = await self._source.read_range(ifd_offset, 2)
        n_entries = struct.unpack(self._endian + "H", count_raw)[0]
        entries = await self._source.read_range(ifd_offset + 2, n_entries * 12)
        tags: Dict[int, List[object]] = {}
        for i in range(n_entries):
            entry = entries[i * 12 : (i + 1) * 12]
            tag = struct.unpack(self._endian + "H", entry[0:2])[0]
            typ = struct.unpack(self._endian + "H", entry[2:4])[0]
            count = struct.unpack(self._endian + "I", entry[4:8])[0]
            value_field = entry[8:12]
            tags[tag] = await self._read_values(typ, count, value_field)
        return tags

    async def _read_values(
        self, typ: int, count: int, value_field: bytes
    ) -> List[object]:
        """Resolve a tag's values, fetching external data when it overflows 4 bytes."""
        if typ not in _TYPE_STRUCT:
            # Unknown type: keep the raw 4 bytes so parsing never crashes.
            return [value_field]
        _, size = _TYPE_STRUCT[typ]
        total = size * count
        if total <= 4:
            data = value_field[:total]
        else:
            offset = struct.unpack(self._endian + "I", value_field)[0]
            data = await self._source.read_range(offset, total)
        return self._decode_values(typ, count, data)

    def _decode_values(self, typ: int, count: int, data: bytes) -> List[object]:
        if typ == 2:  # ASCII
            text = data.split(b"\x00", 1)[0].decode("ascii", errors="replace")
            return [text]
        if typ in (5, 10):  # RATIONAL / SRATIONAL -> floats
            char = "I" if typ == 5 else "i"
            nums = struct.unpack(self._endian + char * (count * 2), data)
            out: List[object] = []
            for k in range(count):
                num = nums[2 * k]
                den = nums[2 * k + 1]
                out.append(float(num) / den if den else 0.0)
            return out
        char, _ = _TYPE_STRUCT[typ]
        return list(struct.unpack(self._endian + char * count, data))

    def _build_metadata(self, tags: Dict[int, List[object]]) -> CogMetadata:
        def first_int(tag: int, default: Optional[int] = None) -> Optional[int]:
            if tag in tags and tags[tag]:
                return int(tags[tag][0])  # type: ignore[arg-type]
            return default

        width = first_int(_TAG_IMAGE_WIDTH)
        height = first_int(_TAG_IMAGE_LENGTH)
        if width is None or height is None:
            raise UpstreamError(
                "asset is missing image dimensions", source=self._source_id
            )

        tile_width = first_int(_TAG_TILE_WIDTH)
        tile_height = first_int(_TAG_TILE_LENGTH)
        if tile_width is None or tile_height is None:
            raise UpstreamError(
                "asset is not tiled (striped TIFFs are unsupported by geo-raster)",
                source=self._source_id,
            )

        samples = first_int(_TAG_SAMPLES_PER_PIXEL, 1) or 1
        bits_values = [int(v) for v in tags.get(_TAG_BITS_PER_SAMPLE, [8])]  # type: ignore[arg-type]
        if len(set(bits_values)) > 1:
            raise UpstreamError(
                "assets with mixed bit depths per sample are unsupported",
                source=self._source_id,
            )
        bits = bits_values[0]
        fmt_values = [int(v) for v in tags.get(_TAG_SAMPLE_FORMAT, [1])]  # type: ignore[arg-type]
        sample_format = fmt_values[0]

        compression = first_int(_TAG_COMPRESSION, _COMPRESSION_NONE) or _COMPRESSION_NONE
        predictor = first_int(_TAG_PREDICTOR, 1) or 1
        planar_config = first_int(_TAG_PLANAR_CONFIG, 1) or 1

        if compression != _COMPRESSION_NONE and compression not in _COMPRESSION_DEFLATE:
            raise UpstreamError(
                f"unsupported tile compression {compression} (geo-raster supports "
                "uncompressed and DEFLATE only)",
                source=self._source_id,
            )
        if predictor not in (1, 2):
            raise UpstreamError(
                f"unsupported predictor {predictor} (geo-raster supports none or "
                "horizontal differencing)",
                source=self._source_id,
            )
        if sample_format not in (
            _SAMPLE_FORMAT_UINT,
            _SAMPLE_FORMAT_INT,
            _SAMPLE_FORMAT_FLOAT,
        ):
            raise UpstreamError(
                f"unsupported sample format {sample_format}", source=self._source_id
            )
        self._struct_char(bits, sample_format)  # validate dtype is supported

        tile_offsets = [int(v) for v in tags.get(_TAG_TILE_OFFSETS, [])]  # type: ignore[arg-type]
        tile_byte_counts = [int(v) for v in tags.get(_TAG_TILE_BYTE_COUNTS, [])]  # type: ignore[arg-type]
        if not tile_offsets or len(tile_offsets) != len(tile_byte_counts):
            raise UpstreamError(
                "asset has a malformed tile index", source=self._source_id
            )

        nodata = self._parse_nodata(tags.get(_TAG_GDAL_NODATA))
        geotransform = self._parse_geotransform(tags)

        return CogMetadata(
            width=width,
            height=height,
            tile_width=tile_width,
            tile_height=tile_height,
            samples_per_pixel=samples,
            bits_per_sample=bits,
            sample_format=sample_format,
            compression=compression,
            predictor=predictor,
            planar_config=planar_config,
            tile_offsets=tile_offsets,
            tile_byte_counts=tile_byte_counts,
            nodata=nodata,
            geotransform=geotransform,
        )

    @staticmethod
    def _parse_nodata(raw: Optional[List[object]]) -> Optional[float]:
        if not raw:
            return None
        try:
            return float(str(raw[0]).strip())
        except (TypeError, ValueError):
            return None

    def _parse_geotransform(
        self, tags: Dict[int, List[object]]
    ) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Build a GDAL-style 6-tuple geotransform from GeoTIFF model tags."""
        if _TAG_MODEL_TRANSFORMATION in tags:
            m = [float(v) for v in tags[_TAG_MODEL_TRANSFORMATION]]  # type: ignore[arg-type]
            if len(m) >= 16:
                # 4x4 row-major: [a b c d; e f g h; ...]; x = a*col + b*row + d.
                return (m[3], m[0], m[1], m[7], m[4], m[5])
        scale = tags.get(_TAG_MODEL_PIXEL_SCALE)
        tiepoint = tags.get(_TAG_MODEL_TIEPOINT)
        if scale and tiepoint:
            s = [float(v) for v in scale]  # type: ignore[arg-type]
            t = [float(v) for v in tiepoint]  # type: ignore[arg-type]
            if len(s) >= 2 and len(t) >= 6:
                sx, sy = s[0], s[1]
                i, j, _k, x, y, _z = t[0], t[1], t[2], t[3], t[4], t[5]
                origin_x = x - i * sx
                origin_y = y + j * sy
                return (origin_x, sx, 0.0, origin_y, 0.0, -sy)
        return None

    # ------------------------------------------------------------------
    # Windowed pixel reads (only overlapping tiles are fetched)
    # ------------------------------------------------------------------

    async def read_window_pixels(
        self,
        *,
        col_off: int,
        row_off: int,
        width: int,
        height: int,
        bands: List[int],
    ) -> List[List[float]]:
        """Return per-band row-major values for the requested pixel window.

        Fetches **only** the tiles overlapping the window. ``bands`` is a list
        of 1-based band indices, already validated by the caller.
        """
        meta = self.metadata
        if col_off + width > meta.width or row_off + height > meta.height:
            raise ValidationError(
                "requested window extends beyond the asset extent "
                f"({meta.width}x{meta.height})",
                source=self._source_id,
                detail={"parameter": "window"},
            )

        fill = meta.nodata if meta.nodata is not None else 0.0
        out: List[List[float]] = [[fill] * (width * height) for _ in bands]

        tc0 = col_off // meta.tile_width
        tc1 = (col_off + width - 1) // meta.tile_width
        tr0 = row_off // meta.tile_height
        tr1 = (row_off + height - 1) // meta.tile_height

        for tr in range(tr0, tr1 + 1):
            for tc in range(tc0, tc1 + 1):
                tile_values = await self._read_tile(tr, tc, bands)
                self._blit_tile(
                    tile_values,
                    out,
                    tr=tr,
                    tc=tc,
                    col_off=col_off,
                    row_off=row_off,
                    width=width,
                    height=height,
                    n_bands=len(bands),
                )
        return out

    def _blit_tile(
        self,
        tile_values: List[List[float]],
        out: List[List[float]],
        *,
        tr: int,
        tc: int,
        col_off: int,
        row_off: int,
        width: int,
        height: int,
        n_bands: int,
    ) -> None:
        """Copy the in-window portion of one tile into the output bands."""
        meta = self.metadata
        tile_origin_col = tc * meta.tile_width
        tile_origin_row = tr * meta.tile_height

        # Intersection of the tile with the requested window, in image coords.
        start_col = max(col_off, tile_origin_col)
        end_col = min(col_off + width, tile_origin_col + meta.tile_width)
        start_row = max(row_off, tile_origin_row)
        end_row = min(row_off + height, tile_origin_row + meta.tile_height)

        for img_row in range(start_row, end_row):
            in_tile_row = img_row - tile_origin_row
            out_row = img_row - row_off
            for img_col in range(start_col, end_col):
                in_tile_col = img_col - tile_origin_col
                out_col = img_col - col_off
                tile_idx = in_tile_row * meta.tile_width + in_tile_col
                out_idx = out_row * width + out_col
                for b in range(n_bands):
                    out[b][out_idx] = tile_values[b][tile_idx]

    async def _read_tile(
        self, tile_row: int, tile_col: int, bands: List[int]
    ) -> List[List[float]]:
        """Fetch + decode one tile, returning per-requested-band value grids.

        Each returned band grid is ``tile_height * tile_width`` values in
        row-major order. Only this tile's byte range(s) are read.
        """
        meta = self.metadata
        tile_in_plane = tile_row * meta.tiles_across + tile_col
        per_tile = meta.tile_width * meta.tile_height

        if meta.planar_config == 2:
            # Separate planes: one tile per band; fetch only requested bands.
            result: List[List[float]] = []
            for band in bands:
                plane = band - 1
                idx = plane * meta.tiles_per_plane + tile_in_plane
                raw = await self._fetch_tile_samples(idx, per_tile * 1, spp=1)
                result.append([float(v) for v in raw])
            return result

        # Contiguous: one tile holds all samples interleaved per pixel.
        spp = meta.samples_per_pixel
        raw = await self._fetch_tile_samples(tile_in_plane, per_tile * spp, spp=spp)
        result = []
        for band in bands:
            s = band - 1
            grid = [float(raw[p * spp + s]) for p in range(per_tile)]
            result.append(grid)
        return result

    async def _fetch_tile_samples(
        self, tile_index: int, n_values: int, *, spp: int
    ) -> List[float]:
        """Read one tile's bytes by index, decompress, de-predict, and unpack."""
        meta = self.metadata
        if not (0 <= tile_index < len(meta.tile_offsets)):
            raise UpstreamError(
                f"tile index {tile_index} is out of range", source=self._source_id
            )
        offset = meta.tile_offsets[tile_index]
        byte_count = meta.tile_byte_counts[tile_index]
        raw = await self._source.read_range(offset, byte_count)
        if meta.compression in _COMPRESSION_DEFLATE:
            raw = zlib.decompress(raw)

        char = self._struct_char(meta.bits_per_sample, meta.sample_format)
        values = list(
            struct.unpack(
                self._endian + char * n_values,
                raw[: n_values * (meta.bits_per_sample // 8)],
            )
        )

        if meta.predictor == 2:
            self._apply_horizontal_predictor(
                values,
                rows=meta.tile_height,
                cols=meta.tile_width,
                spp=spp,
                bits=meta.bits_per_sample,
                sample_format=meta.sample_format,
            )
        return values

    @staticmethod
    def _struct_char(bits: int, sample_format: int) -> str:
        """Return the struct format char for ``(bits, sample_format)``."""
        if sample_format == _SAMPLE_FORMAT_FLOAT:
            mapping = {32: "f", 64: "d"}
        elif sample_format == _SAMPLE_FORMAT_INT:
            mapping = {8: "b", 16: "h", 32: "i"}
        else:
            mapping = {8: "B", 16: "H", 32: "I"}
        if bits not in mapping:
            raise UpstreamError(
                f"unsupported {bits}-bit sample for sample format {sample_format}"
            )
        return mapping[bits]

    @staticmethod
    def _apply_horizontal_predictor(
        values: List[float],
        *,
        rows: int,
        cols: int,
        spp: int,
        bits: int,
        sample_format: int,
    ) -> None:
        """Undo TIFF Predictor 2 (horizontal differencing) in place."""
        mask = (1 << bits) - 1 if sample_format != _SAMPLE_FORMAT_FLOAT else None
        for r in range(rows):
            base = r * cols * spp
            for s in range(spp):
                for c in range(1, cols):
                    cur = base + c * spp + s
                    prev = base + (c - 1) * spp + s
                    acc = values[cur] + values[prev]
                    if mask is not None:
                        acc &= mask
                    values[cur] = acc

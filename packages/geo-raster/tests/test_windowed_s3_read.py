"""Windowed S3 read behavior over a mocked HTTP transport (task 13.7).

The other ``geo-raster`` windowed-read tests drive the COG engine through an in-memory
:class:`~tests._cogbuild.RecordingByteRangeReader`, which bypasses HTTP entirely.
This module closes the remaining gap for Requirements 7.2 and 12.1 by exercising
the *production* read path end to end:

    read_window / band_math
        -> geo_raster.reader.HttpRangeReader
            -> geo_common.http.HttpClient   (shared retry/backoff/timeout)
                -> httpx.MockTransport       (a mocked S3 origin)

The mocked transport stands in for an S3 object store. It serves HTTP ``Range``
requests against an in-memory COG and records every byte range it is asked for,
so each test can assert the windowed-read contract against the wire:

* an ``s3://amzn-s3-demo-bucket/key`` href is resolved to its HTTPS virtual-hosted endpoint
  (default and region-specific), and the asset is fetched from there;
* every fetch is a ranged ``GET`` answered with ``206 Partial Content`` — the
  full asset is never requested in one shot and never copied wholesale
  (Requirement 12.1);
* only the tiles overlapping the requested window are fetched, so the bytes
  transferred are a small fraction of the asset (Requirements 7.2, 12.1); and
* the pixels returned are exactly the in-window pixels (Requirement 7.2).
"""

from __future__ import annotations

import re
from typing import List, Tuple

import httpx
import pytest

from _cogbuild import build_cog

from geo_common.http import HttpClient
from geo_common.retry import RetryPolicy

from geo_raster import CogReader, GeoRasterServer, PixelWindow, band_math, read_window
from geo_raster.reader import HttpRangeReader

_RANGE_RE = re.compile(r"bytes=(\d+)-(\d+)")


def _ramp(width: int, height: int, *, offset: int = 0) -> List[float]:
    """A deterministic band: value = (row*1000 + col + offset) mod 65536."""
    return [((r * 1000 + c + offset) % 65536) for r in range(height) for c in range(width)]


class MockS3Origin:
    """An ``httpx.MockTransport`` handler that serves ranged reads of a COG.

    Backed by a ``bytes`` buffer, it answers ``Range: bytes=a-b`` requests with
    ``206 Partial Content`` carrying exactly those bytes, and records the URL and
    ``(offset, length)`` of every request. A request without a ``Range`` header
    would return the whole object with ``200`` — but the reader always sends a
    ``Range`` header, so receiving a ``200`` here would itself be a contract
    violation the tests can detect.
    """

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.requests: List[httpx.Request] = []
        self.ranges: List[Tuple[int, int]] = []
        self.unranged = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        raw_range = request.headers.get("Range")
        if raw_range is None:
            self.unranged += 1
            return httpx.Response(200, content=self._data)
        match = _RANGE_RE.fullmatch(raw_range.strip())
        assert match is not None, f"unparseable Range header: {raw_range!r}"
        start = int(match.group(1))
        end = int(match.group(2))
        chunk = self._data[start : end + 1]
        self.ranges.append((start, len(chunk)))
        total = len(self._data)
        return httpx.Response(
            206,
            content=chunk,
            headers={
                "Content-Range": f"bytes {start}-{start + len(chunk) - 1}/{total}",
                "Accept-Ranges": "bytes",
            },
        )

    @property
    def total_bytes_served(self) -> int:
        return sum(length for _, length in self.ranges)

    def fetched_offsets(self) -> set:
        return {offset for offset, _ in self.ranges}


async def _no_sleep(_seconds: float) -> None:
    return None


def _client(origin: MockS3Origin) -> HttpClient:
    """An :class:`HttpClient` whose transport is the mocked S3 origin."""
    return HttpClient(
        RetryPolicy(max_attempts=1),
        transport=httpx.MockTransport(origin),
        sleep=_no_sleep,
    )


async def _tile_offsets(data: bytes) -> List[int]:
    """Parse the asset's tile offsets via an in-memory reader (no HTTP)."""
    from _cogbuild import RecordingByteRangeReader

    cog = CogReader(RecordingByteRangeReader(data), source_id="mem")
    meta = await cog.open()
    return list(meta.tile_offsets)


# --- s3:// resolution to the HTTPS virtual-hosted endpoint ------------------


async def test_s3_href_resolves_to_virtual_hosted_https_endpoint() -> None:
    """An ``s3://amzn-s3-demo-bucket/key`` href is fetched from ``bucket.s3.amazonaws.com``."""
    width = height = 128
    data = build_cog(
        width=width, height=height, tile_width=64, tile_height=64, bands=[_ramp(width, height)]
    )
    origin = MockS3Origin(data)
    client = _client(origin)
    try:
        await read_window(
            asset_href="s3://amzn-s3-demo-imagery/scenes/B04.tif",
            window=PixelWindow(col_off=0, row_off=0, width=4, height=4),
            http=client,
        )
    finally:
        await client.aclose()

    assert origin.requests, "the asset was fetched"
    host = origin.requests[0].url.host
    path = origin.requests[0].url.path
    assert host == "amzn-s3-demo-imagery.s3.amazonaws.com"
    assert path == "/scenes/B04.tif"


async def test_s3_href_resolves_to_region_specific_endpoint() -> None:
    """A non-default region resolves to the regional virtual-hosted endpoint."""
    width = height = 64
    data = build_cog(
        width=width, height=height, tile_width=32, tile_height=32, bands=[_ramp(width, height)]
    )
    origin = MockS3Origin(data)
    client = _client(origin)
    try:
        await read_window(
            asset_href="s3://amzn-s3-demo-eu/key.tif",
            window=PixelWindow(col_off=0, row_off=0, width=2, height=2),
            http=client,
            region="eu-central-1",
        )
    finally:
        await client.aclose()

    assert origin.requests[0].url.host == "amzn-s3-demo-eu.s3.eu-central-1.amazonaws.com"


# --- only in-window bytes fetched; no full-asset copy -----------------------


async def test_windowed_read_fetches_only_in_window_tile_over_http() -> None:
    """Req 7.2/12.1: over real HTTP ranges, only the overlapping tile is fetched."""
    width = height = 256
    band = _ramp(width, height)
    data = build_cog(
        width=width, height=height, tile_width=128, tile_height=128, bands=[band]
    )
    tile_offsets = await _tile_offsets(data)

    origin = MockS3Origin(data)
    client = _client(origin)
    try:
        result = await read_window(
            asset_href="s3://amzn-s3-demo-bucket/asset.tif",
            window=PixelWindow(col_off=10, row_off=20, width=8, height=4),
            http=client,
        )
    finally:
        await client.aclose()

    # Correct in-window pixels (Req 7.2).
    assert (result.width, result.height) == (8, 4)
    for r in range(4):
        for c in range(8):
            assert result.value(1, r, c) == band[(20 + r) * width + (10 + c)]

    # Only tile (0,0) was fetched; the other three tiles' byte ranges never were.
    fetched = origin.fetched_offsets()
    assert tile_offsets[0] in fetched
    assert not ({tile_offsets[1], tile_offsets[2], tile_offsets[3]} & fetched)

    # No full-asset copy (Req 12.1): every fetch was a small ranged read, and the
    # total bytes served is a small fraction of the whole asset.
    assert origin.unranged == 0
    assert origin.total_bytes_served < len(data)
    assert max(length for _, length in origin.ranges) < len(data)


async def test_every_fetch_is_a_partial_content_range_request() -> None:
    """Req 12.1: the reader only ever issues ranged GETs (answered 206)."""
    width = height = 128
    data = build_cog(
        width=width, height=height, tile_width=64, tile_height=64, bands=[_ramp(width, height)]
    )
    origin = MockS3Origin(data)
    client = _client(origin)
    try:
        await read_window(
            asset_href="s3://amzn-s3-demo-bucket/asset.tif",
            window=PixelWindow(col_off=1, row_off=1, width=3, height=3),
            http=client,
        )
    finally:
        await client.aclose()

    # Every request carried a Range header (so every response was a 206 slice);
    # none asked for the whole object.
    assert origin.unranged == 0
    assert all("Range" in req.headers for req in origin.requests)


async def test_window_spanning_tiles_fetches_each_overlapping_tile_over_http() -> None:
    """A window straddling the tile grid fetches every overlapping tile, no more."""
    width = height = 256
    band = _ramp(width, height)
    data = build_cog(
        width=width, height=height, tile_width=128, tile_height=128, bands=[band]
    )
    tile_offsets = await _tile_offsets(data)

    origin = MockS3Origin(data)
    client = _client(origin)
    try:
        result = await read_window(
            asset_href="s3://amzn-s3-demo-bucket/asset.tif",
            window=PixelWindow(col_off=120, row_off=120, width=16, height=16),
            http=client,
        )
    finally:
        await client.aclose()

    assert (result.width, result.height) == (16, 16)
    for r in range(16):
        for c in range(16):
            assert result.value(1, r, c) == band[(120 + r) * width + (120 + c)]
    # The window touches all four tiles; each overlapping tile's range is fetched.
    assert set(tile_offsets) <= origin.fetched_offsets()


async def test_https_href_is_fetched_directly_with_ranges() -> None:
    """An ``http(s)://`` href is read via ranges without any s3 rewriting."""
    width = height = 64
    band = _ramp(width, height)
    data = build_cog(
        width=width, height=height, tile_width=32, tile_height=32, bands=[band]
    )
    origin = MockS3Origin(data)
    client = _client(origin)
    try:
        result = await read_window(
            asset_href="https://example.com/cogs/asset.tif",
            window=PixelWindow(col_off=0, row_off=0, width=2, height=2),
            http=client,
        )
    finally:
        await client.aclose()

    assert origin.requests[0].url.host == "example.com"
    assert origin.requests[0].url.path == "/cogs/asset.tif"
    assert origin.unranged == 0
    assert result.value(1, 0, 0) == band[0]


# --- band_math over the mocked S3 transport ---------------------------------


async def test_band_math_over_http_reads_only_referenced_band() -> None:
    """Req 7.2/12.1: band_math over HTTP fetches only the referenced band's tiles."""
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

    meta = await CogReader(_RecordingFor(data), source_id="m").open()
    tiles_per_plane = meta.tiles_per_plane
    band1 = {meta.tile_offsets[t] for t in range(tiles_per_plane)}
    band2 = {meta.tile_offsets[tiles_per_plane + t] for t in range(tiles_per_plane)}
    band3 = {meta.tile_offsets[2 * tiles_per_plane + t] for t in range(tiles_per_plane)}

    origin = MockS3Origin(data)
    client = _client(origin)
    try:
        result = await band_math(
            asset_href="s3://amzn-s3-demo-bucket/multi.tif",
            expression="B2 * 2",
            window={"col_off": 0, "row_off": 0, "width": 4, "height": 4},
            http=client,
        )
    finally:
        await client.aclose()

    fetched = origin.fetched_offsets()
    assert band2 & fetched
    assert not (band1 & fetched)
    assert not (band3 & fetched)
    assert origin.unranged == 0
    assert result.value(0, 0, 1) == pytest.approx(b2[1] * 2)


# --- through the assembled server -------------------------------------------


async def test_server_read_window_over_mock_s3_transport() -> None:
    """The assembled server reads in-window pixels over its shared HttpClient."""
    width = height = 64
    band = _ramp(width, height)
    data = build_cog(
        width=width, height=height, tile_width=32, tile_height=32, bands=[band]
    )
    origin = MockS3Origin(data)
    server = GeoRasterServer(http=_client(origin))
    try:
        result = await server.read_window(
            asset_href="s3://amzn-s3-demo-bucket/scene.tif",
            window=PixelWindow(col_off=2, row_off=3, width=5, height=5),
        )
    finally:
        await server.aclose()

    assert (result.width, result.height) == (5, 5)
    for r in range(5):
        for c in range(5):
            assert result.value(1, r, c) == band[(3 + r) * width + (2 + c)]
    assert origin.unranged == 0
    assert origin.total_bytes_served < len(data)


# --- HttpRangeReader unit behavior ------------------------------------------


async def test_http_range_reader_returns_exactly_the_requested_bytes() -> None:
    """A direct :class:`HttpRangeReader` read returns exactly the slice asked for."""
    data = bytes(range(256)) * 4  # 1024 deterministic bytes
    origin = MockS3Origin(data)
    client = _client(origin)
    reader = HttpRangeReader("s3://amzn-s3-demo-bucket/blob.bin", client)
    try:
        chunk = await reader.read_range(100, 16)
    finally:
        await client.aclose()

    assert chunk == data[100:116]
    assert origin.ranges == [(100, 16)]
    # A zero-length read short-circuits without touching the transport.
    assert await reader.read_range(0, 0) == b""
    assert len(origin.requests) == 1


class _RecordingFor:
    """Tiny in-memory byte-range reader used to parse tile offsets in one test."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read_range(self, offset: int, length: int) -> bytes:
        return self._data[offset : offset + length]

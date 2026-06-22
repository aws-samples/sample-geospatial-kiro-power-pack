"""Tests for ``geo-raster`` raster reads and S3 read-failure handling.

These exercise the full ``zonal_statistics`` path through the
:class:`~geo_raster.server.GeoRasterServer`:

* reading a georeferenced window from an (in-memory) Cloud-Optimized GeoTIFF via
  byte ranges and computing correct per-zone statistics (Requirements 8.7, 8.8),
  asserting the byte-range discipline (only the tiles overlapping the zones'
  window are fetched);
* aborting with local storage unchanged and a ``network`` Error_Taxonomy error
  indicating the S3 read failure after the read fails (Requirement 12.6);
* issuing no read at all when there are no zones to summarize.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import pytest

from _rasterbuild import RecordingByteRangeReader, build_cog

from geo_common.errors import ErrorCategory, GeoError, NetworkError

from geo_raster import (
    CogReader,
    Feature,
    FeatureCollection,
    GeoJSONGeometry,
    GeoRasterServer,
)

# North-up 4x4 grid: origin (0, 4), 1 unit/cell, px_h = -1; value = row*4 + col.
_GT = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)


def _values() -> List[float]:
    return [float(r * 4 + c) for r in range(4) for c in range(4)]


def _polygon(min_x: float, min_y: float, max_x: float, max_y: float, zid: str) -> Feature:
    ring = [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
        [min_x, min_y],
    ]
    geom = GeoJSONGeometry(type="Polygon", coordinates=[ring])
    return Feature(geometry=geom, properties={"zone_id": zid})


async def _tile_offsets(data: bytes) -> List[int]:
    reader = RecordingByteRangeReader(data)
    cog = CogReader(reader, source_id="mem")
    meta = await cog.open()
    return list(meta.tile_offsets)


async def test_zonal_statistics_reads_window_via_byte_ranges() -> None:
    """Req 8.7/12.1: stats are correct and only the overlapping tile is fetched."""
    data = build_cog(
        width=4,
        height=4,
        tile_width=2,
        tile_height=2,
        bands=[_values()],
        dtype="float64",
        geotransform=_GT,
    )
    tile_offsets = await _tile_offsets(data)

    server = GeoRasterServer()
    try:
        reader = RecordingByteRangeReader(data)
        # Zone over cells {0,1,4,5} (top-left 2x2 == tile (0,0)).
        zones = FeatureCollection(features=[_polygon(0.0, 2.0, 2.0, 4.0, "A")])
        result = await server.zonal_statistics(
            raster_href="s3://amzn-s3-demo-bucket/asset.tif", zones=zones, reader=reader
        )
    finally:
        await server.aclose()

    assert len(result) == 1
    assert result[0].statistics == {
        "min": 0.0,
        "max": 5.0,
        "mean": 2.5,
        "sum": 10.0,
        "count": 4.0,
    }

    # Byte-range discipline: only tile (0,0) was fetched; the other three tiles'
    # byte ranges were never requested, and far less than the whole asset moved.
    fetched = set(reader.fetched_offsets())
    assert tile_offsets[0] in fetched
    assert not ({tile_offsets[1], tile_offsets[2], tile_offsets[3]} & fetched)
    assert reader.total_bytes_read < len(data)


async def test_empty_zone_within_extent_is_no_data_via_read() -> None:
    """A zone inside the extent but covering no cell centers yields no-data."""
    data = build_cog(
        width=4,
        height=4,
        tile_width=2,
        tile_height=2,
        bands=[_values()],
        dtype="float64",
        geotransform=_GT,
    )
    server = GeoRasterServer()
    try:
        reader = RecordingByteRangeReader(data)
        # A tiny box strictly between cell centers (x in (1.5,2.5), y in (2.5,3.5)).
        empty = _polygon(1.6, 2.6, 1.9, 2.9, "gap")
        covered = _polygon(0.0, 2.0, 2.0, 4.0, "A")
        zones = FeatureCollection(features=[empty, covered])
        result = await server.zonal_statistics(
            raster_href="s3://amzn-s3-demo-bucket/asset.tif", zones=zones, reader=reader
        )
    finally:
        await server.aclose()

    by_id = {z.zone_id: z for z in result}
    assert by_id["gap"].no_data is True
    assert set(by_id["gap"].statistics.values()) == {None}
    assert by_id["A"].no_data is False
    assert by_id["A"].statistics["count"] == 4.0


class _FailingReader:
    """A byte-range reader that simulates an exhausted-retry S3 read failure."""

    def __init__(self) -> None:
        self.calls = 0

    async def read_range(self, offset: int, length: int) -> bytes:
        self.calls += 1
        # Mimic the shared HttpClient surfacing a network error after it has
        # retried the failing S3 read within the 30-second window (Req 5.4/5.9).
        raise NetworkError(
            "request timed out after 3 retry attempts",
            source="s3://amzn-s3-demo-bucket/asset.tif",
        )


async def test_s3_read_failure_aborts_with_local_storage_unchanged(tmp_path) -> None:
    """Req 12.6: abort, leave local storage unchanged, return an S3-read error."""
    before = sorted(os.listdir(tmp_path))

    server = GeoRasterServer()
    try:
        zones = FeatureCollection(features=[_polygon(0.0, 2.0, 2.0, 4.0, "A")])
        with pytest.raises(GeoError) as excinfo:
            await server.zonal_statistics(
                raster_href="s3://amzn-s3-demo-bucket/asset.tif",
                zones=zones,
                reader=_FailingReader(),
            )
    finally:
        await server.aclose()

    err = excinfo.value
    # Categorized as a network failure indicating the S3 read failure.
    assert err.category is ErrorCategory.NETWORK
    assert err.detail is not None and err.detail.get("reason") == "s3_read_failure"
    # Local storage is unchanged: nothing was written during the aborted read.
    assert sorted(os.listdir(tmp_path)) == before


async def test_no_zones_returns_empty_and_issues_no_read() -> None:
    """With no zones there is nothing to summarize and no read is issued."""
    reader = RecordingByteRangeReader(b"not-a-real-tiff")
    server = GeoRasterServer()
    try:
        result = await server.zonal_statistics(
            raster_href="s3://amzn-s3-demo-bucket/asset.tif",
            zones=FeatureCollection(features=[]),
            reader=reader,
        )
    finally:
        await server.aclose()

    assert result == []
    assert reader.reads == []


async def test_server_registers_zonal_statistics_tool_and_catalog_entry() -> None:
    """The assembled server exposes the tool and an Open catalog entry."""
    server = GeoRasterServer()
    try:
        assert "zonal_statistics" in server.tool_names()
        entries = server.catalog_entries()
        names = [e.name for e in entries]
        assert "zonal_statistics" in names
        assert {e.provider_server for e in entries} == {"geo-raster"}
        # Optional AWS credentials never block startup (Req 16.5).
        assert server.missing_required_credentials(configured_keys=[]) == []
        server.start(configured_keys=[])
        assert server.started is True
    finally:
        await server.aclose()

"""Focused S3 read abort-and-cleanup unit tests for ``geo-raster`` (task 14.12).

Task 14.5 covers the S3 read-failure abort through the full
``GeoRasterServer.zonal_statistics`` path; this module pins the same contract on
the lower-level :func:`geo_raster.reader.read_raster_grid` engine so the
abort-and-leave-local-storage-unchanged behaviour is verified at its source
(Requirement 12.6):

* a byte-range read that fails after the shared client's retries surfaces as a
  taxonomy ``network`` error explicitly tagged ``s3_read_failure``, and a read
  *was* attempted;
* the read path writes nothing to local storage, so the working directory is
  unchanged across the aborted read;
* a successful windowed read still works and only fetches the overlapping
  tile's bytes (byte-range discipline / Req 12.1), so the abort test is not
  passing vacuously.

Everything runs against an in-memory asset and a mocked failing reader - no S3
or network access. The repo's ``asyncio_mode = "auto"`` runs the async tests.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import pytest

from _rasterbuild import RecordingByteRangeReader, build_cog

from geo_common.errors import ErrorCategory, NetworkError
from geo_raster.reader import read_raster_grid

# North-up 4x4 grid: origin (0, 4), 1 unit/cell, px_h = -1; value = row*4 + col.
_GT = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)


def _values() -> List[float]:
    return [float(r * 4 + c) for r in range(4) for c in range(4)]


def _asset() -> bytes:
    return build_cog(
        width=4,
        height=4,
        tile_width=2,
        tile_height=2,
        bands=[_values()],
        dtype="float64",
        geotransform=_GT,
    )


class _FailingReader:
    """A byte-range reader simulating an exhausted-retry S3 read failure."""

    def __init__(self) -> None:
        self.calls = 0

    async def read_range(self, offset: int, length: int) -> bytes:
        self.calls += 1
        raise NetworkError(
            "request timed out after 3 retry attempts",
            source="s3://amzn-s3-demo-bucket/asset.tif",
            original="read timeout",
        )


async def test_read_raster_grid_aborts_on_s3_read_failure(tmp_path, monkeypatch) -> None:
    """Req 12.6: a failed read aborts, leaves local storage unchanged, raises."""
    monkeypatch.chdir(tmp_path)
    before = sorted(os.listdir(tmp_path))

    reader = _FailingReader()
    with pytest.raises(NetworkError) as excinfo:
        await read_raster_grid(
            raster_href="s3://amzn-s3-demo-bucket/asset.tif",
            window_bbox=(0.0, 2.0, 2.0, 4.0),
            reader=reader,
        )

    err = excinfo.value
    assert err.category is ErrorCategory.NETWORK
    # The error explicitly indicates the S3 read failure (Req 12.6).
    assert err.detail is not None and err.detail.get("reason") == "s3_read_failure"
    assert err.detail.get("href") == "s3://amzn-s3-demo-bucket/asset.tif"
    # The original cause is retained for diagnostics.
    assert err.original is not None
    # A read was actually attempted (not short-circuited before the read).
    assert reader.calls >= 1
    # Local storage is unchanged: nothing was written during the aborted read.
    assert sorted(os.listdir(tmp_path)) == before


async def test_read_raster_grid_success_reads_only_overlapping_tile() -> None:
    """Control: a valid windowed read succeeds and only fetches the window's tile."""
    data = _asset()
    reader = RecordingByteRangeReader(data)
    grid = await read_raster_grid(
        raster_href="s3://amzn-s3-demo-bucket/asset.tif",
        window_bbox=(0.0, 2.0, 2.0, 4.0),  # top-left 2x2 == tile (0,0)
        reader=reader,
    )
    # The top-left 2x2 window holds cells {0,1,4,5} (flat row-major values).
    assert grid.width == 2 and grid.height == 2
    assert grid.values == [0.0, 1.0, 4.0, 5.0]
    # Byte-range discipline: far less than the whole asset was transferred.
    assert reader.total_bytes_read < len(data)


async def test_read_raster_grid_validates_href() -> None:
    """A malformed href is rejected before any read is attempted."""
    from geo_common.errors import ValidationError

    reader = RecordingByteRangeReader(_asset())
    with pytest.raises(ValidationError):
        await read_raster_grid(raster_href="ftp://bucket/asset.tif", reader=reader)
    assert reader.reads == []

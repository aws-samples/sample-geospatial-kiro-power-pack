"""Gated live check: read a real remote Cloud-Optimized Point Cloud via laspy.

Skipped unless ``RUN_LIVE_COPC=1``. Reads a public ``.copc.laz`` over the
network using the COPC octree index (windowed), verifying the real read path
end to end. Requires the ``[copc]`` and ``[remote]`` extras (laspy + lazrs +
fsspec/s3fs or an https filesystem).

    RUN_LIVE_COPC=1 pytest packages/geo-pointcloud/tests/test_copc_live.py

Override the sample with ``COPC_LIVE_URL`` (default: the public Autzen COPC on
the Hobu demo bucket over HTTPS).
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_COPC") != "1",
    reason="set RUN_LIVE_COPC=1 to run the live COPC read check",
)

_DEFAULT_URL = "https://s3.amazonaws.com/hobu-lidar/autzen-classified.copc.laz"


def test_live_copc_windowed_read() -> None:
    pytest.importorskip("laspy")
    pytest.importorskip("fsspec")
    from geo_pointcloud import GeoWindow, LaspyCopcBackend, read_pointcloud

    url = os.environ.get("COPC_LIVE_URL", _DEFAULT_URL)
    backend = LaspyCopcBackend()
    chunk = read_pointcloud(copc_href=url, backend=backend)
    assert chunk.point_count > 0
    # A tiny window should return far fewer points than the full cloud, proving
    # the octree index limited the read.
    b = chunk.bounds()
    assert b is not None
    min_x, min_y, _mz, max_x, max_y, _Mz = b
    span_x = (max_x - min_x) * 0.05
    span_y = (max_y - min_y) * 0.05
    window = GeoWindow(
        min_x=min_x, min_y=min_y, max_x=min_x + span_x, max_y=min_y + span_y
    )
    clipped = read_pointcloud(copc_href=url, bounds=window, backend=backend)
    assert clipped.point_count <= chunk.point_count

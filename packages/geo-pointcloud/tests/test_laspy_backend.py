"""Tests for the real laspy COPC/LAZ backend (``[copc]`` extra).

Skipped unless ``laspy`` is importable. These write a genuine ``.laz`` with
``laspy`` and read it back through :class:`LaspyCopcBackend` (the plain LAS/LAZ
path) - exercising real LAZ I/O, not a stand-in - and check that the default
:class:`SmartCopcBackend` routes a non-container file to the laspy reader.

The COPC *octree* read path needs a real ``.copc.laz`` (which only PDAL can
produce), so it is covered by the gated live test ``test_copc_live.py``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("laspy")
pytest.importorskip("lazrs")

from geo_pointcloud import (
    LAZ_FORMAT,
    GeoWindow,
    LaspyCopcBackend,
    PointCloudChunk,
    PointRecord,
    read_pointcloud,
    write_pointcloud,
)

_TOL = 1e-5


def _chunk() -> PointCloudChunk:
    return PointCloudChunk(
        crs="EPSG:32631",
        points=[
            PointRecord(x=1.0, y=2.0, z=3.0, intensity=10, classification=2),
            PointRecord(x=4.5, y=5.25, z=6.125, intensity=20, classification=5),
            PointRecord(x=-7.0, y=8.0, z=-9.0, red=100, green=150, blue=200),
        ],
    )


def _positions(chunk: PointCloudChunk):
    return sorted((p.x, p.y, p.z) for p in chunk.points)


def _close(a, b) -> bool:
    return all(
        abs(ax - bx) <= _TOL and abs(ay - by) <= _TOL and abs(az - bz) <= _TOL
        for (ax, ay, az), (bx, by, bz) in zip(a, b)
    )


def test_laz_write_read_round_trip(tmp_path) -> None:
    backend = LaspyCopcBackend()
    dst = str(tmp_path / "cloud.laz")
    result = write_pointcloud(points=_chunk(), dst_href=dst, backend=backend)
    assert result.format == LAZ_FORMAT
    assert result.point_count == 3

    back = read_pointcloud(copc_href=dst, backend=backend)
    assert back.point_count == 3
    assert _close(_positions(back), _positions(_chunk()))
    # Standard LAS attributes round-trip exactly.
    by_x = {round(p.x, 3): p for p in back.points}
    assert by_x[1.0].intensity == 10 and by_x[1.0].classification == 2
    assert by_x[-7.0].red == 100 and by_x[-7.0].green == 150 and by_x[-7.0].blue == 200


def test_windowed_read_filters(tmp_path) -> None:
    backend = LaspyCopcBackend()
    dst = str(tmp_path / "win.laz")
    write_pointcloud(points=_chunk(), dst_href=dst, backend=backend)
    window = GeoWindow(min_x=0.0, min_y=0.0, max_x=6.0, max_y=6.0)
    back = read_pointcloud(copc_href=dst, bounds=window, backend=backend)
    xs = sorted(round(p.x, 3) for p in back.points)
    assert xs == [1.0, 4.5]  # (-7, 8) excluded


def test_smart_default_routes_laz_to_laspy(tmp_path) -> None:
    # Writing via laspy produces a real .laz (not our gzip container); the
    # default SmartCopcBackend must route the read to the laspy reader, not the
    # local-container reader.
    dst = str(tmp_path / "routed.laz")
    write_pointcloud(points=_chunk(), dst_href=dst, backend=LaspyCopcBackend())
    back = read_pointcloud(copc_href=dst)  # default backend
    assert back.point_count == 3
    assert _close(_positions(back), _positions(_chunk()))

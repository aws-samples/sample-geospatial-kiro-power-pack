"""Unit tests for ``geo-pointcloud`` COPC read/write (Requirement 8.11).

Exercises the read/write core and the server wiring with concrete examples and
edge cases: a lossless round-trip preserves the point set and per-point
attributes, windowed reads return only points inside the window, and malformed
or missing inputs surface on the shared ``Error_Taxonomy``.
"""

from __future__ import annotations

import os

import pytest

from geo_common.errors import ErrorCategory, GeoError
from geo_common.models import CredentialClassification, OpennessTier

from geo_pointcloud import (
    COPC_FORMAT,
    GeoWindow,
    GeoPointcloudServer,
    LocalCopcBackend,
    PointCloudChunk,
    PointRecord,
    read_pointcloud,
    write_pointcloud,
)


def _sample_chunk() -> PointCloudChunk:
    return PointCloudChunk(
        crs="EPSG:32631",
        points=[
            PointRecord(x=1.0, y=2.0, z=3.0, intensity=10, classification=2),
            PointRecord(x=4.5, y=5.25, z=6.125, intensity=20, classification=5),
            PointRecord(x=-7.0, y=8.0, z=-9.0, red=100, green=150, blue=200),
            PointRecord(x=0.0, y=0.0, z=0.0, gps_time=123456.789),
        ],
    )


def _as_set(chunk: PointCloudChunk):
    return {tuple(sorted(p.model_dump().items())) for p in chunk.points}


def test_write_then_read_preserves_point_set(tmp_path) -> None:
    """A chunk written to COPC and read back yields the same point set."""
    chunk = _sample_chunk()
    dst = str(tmp_path / "cloud.copc")

    result = write_pointcloud(points=chunk, dst_href=dst)
    assert result.format == COPC_FORMAT
    assert result.valid is True
    assert result.point_count == 4
    assert os.path.exists(dst)

    back = read_pointcloud(copc_href=dst)
    assert back.point_count == 4
    # Point set (order-independent) and every per-point attribute are preserved.
    assert _as_set(back) == _as_set(chunk)


def test_roundtrip_preserves_crs(tmp_path) -> None:
    chunk = _sample_chunk()
    dst = str(tmp_path / "crs.copc")
    write_pointcloud(points=chunk, dst_href=dst)
    back = read_pointcloud(copc_href=dst)
    assert back.crs == "EPSG:32631"


def test_empty_chunk_roundtrips(tmp_path) -> None:
    dst = str(tmp_path / "empty.copc")
    result = write_pointcloud(points=PointCloudChunk(points=[]), dst_href=dst)
    assert result.point_count == 0
    back = read_pointcloud(copc_href=dst)
    assert back.point_count == 0
    assert back.points == []


def test_windowed_read_returns_only_points_inside(tmp_path) -> None:
    chunk = PointCloudChunk(
        points=[
            PointRecord(x=0.0, y=0.0, z=0.0),
            PointRecord(x=5.0, y=5.0, z=1.0),
            PointRecord(x=10.0, y=10.0, z=2.0),
            PointRecord(x=100.0, y=100.0, z=3.0),
        ]
    )
    dst = str(tmp_path / "windowed.copc")
    write_pointcloud(points=chunk, dst_href=dst)

    window = GeoWindow(min_x=-1.0, min_y=-1.0, max_x=6.0, max_y=6.0)
    back = read_pointcloud(copc_href=dst, bounds=window)
    positions = sorted(p.position() for p in back.points)
    assert positions == [(0.0, 0.0, 0.0), (5.0, 5.0, 1.0)]


def test_windowed_read_respects_z_bounds(tmp_path) -> None:
    chunk = PointCloudChunk(
        points=[
            PointRecord(x=1.0, y=1.0, z=0.0),
            PointRecord(x=1.0, y=1.0, z=50.0),
            PointRecord(x=1.0, y=1.0, z=100.0),
        ]
    )
    dst = str(tmp_path / "zbounds.copc")
    write_pointcloud(points=chunk, dst_href=dst)

    window = GeoWindow(min_x=0.0, min_y=0.0, max_x=2.0, max_y=2.0, min_z=10.0, max_z=60.0)
    back = read_pointcloud(copc_href=dst, bounds=window)
    assert [p.z for p in back.points] == [50.0]


def test_read_missing_file_raises_not_found(tmp_path) -> None:
    missing = str(tmp_path / "does-not-exist.copc")
    with pytest.raises(GeoError) as excinfo:
        read_pointcloud(copc_href=missing)
    assert excinfo.value.category is ErrorCategory.NOT_FOUND


def test_write_empty_href_raises_validation() -> None:
    with pytest.raises(GeoError) as excinfo:
        write_pointcloud(points=_sample_chunk(), dst_href="  ")
    assert excinfo.value.category is ErrorCategory.VALIDATION


def test_read_empty_href_raises_validation() -> None:
    with pytest.raises(GeoError) as excinfo:
        read_pointcloud(copc_href="")
    assert excinfo.value.category is ErrorCategory.VALIDATION


def test_local_backend_morton_orders_but_preserves_set(tmp_path) -> None:
    # Points written in an arbitrary order come back as the same set even though
    # the local backend Morton-orders them on write (COPC-style organization).
    chunk = PointCloudChunk(
        points=[PointRecord(x=float(i % 7), y=float((i * 3) % 5), z=float(i)) for i in range(20)]
    )
    dst = str(tmp_path / "morton.copc")
    LocalCopcBackend().write(chunk, dst)
    back = LocalCopcBackend().read(dst, None)
    assert _as_set(back) == _as_set(chunk)
    assert back.point_count == chunk.point_count


@pytest.mark.asyncio
async def test_server_tools_roundtrip(tmp_path) -> None:
    server = GeoPointcloudServer()
    assert set(server.tool_names()) == {"read_pointcloud", "write_pointcloud"}

    chunk = _sample_chunk()
    dst = str(tmp_path / "server.copc")
    result = await server.write_pointcloud(points=chunk, dst_href=dst)
    assert result.point_count == 4

    back = await server.read_pointcloud(copc_href=dst)
    assert _as_set(back) == _as_set(chunk)


def test_server_catalog_and_credentials() -> None:
    server = GeoPointcloudServer()
    entries = server.catalog_entries()
    names = {e.name for e in entries}
    assert names == {"read_pointcloud", "write_pointcloud"}
    for entry in entries:
        assert entry.provider_server == "geo-pointcloud"
        assert entry.openness_tier is OpennessTier.OPEN
        assert entry.pillar == "B"
        assert entry.install_command == "uvx geo-pointcloud"

    # Optional AWS credentials only: starts with no configured credentials (Req 16.5).
    specs = server.required_credentials()
    assert {c.mcp_json_key for c in specs} == {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    }
    assert all(
        c.classification is CredentialClassification.OPTIONAL for c in specs
    )
    server.start(configured_keys=[])
    assert server.started is True

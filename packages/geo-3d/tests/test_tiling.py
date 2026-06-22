"""Tests for ``geo-3d`` ``points_to_3d_tiles`` (point cloud -> 3D Tiles).

Verifies the written tileset round-trips through ``inspect_tileset`` (valid),
the ``.pnts`` binary header parses and reports the right point count, color
embedding works, and the validation guards fire before anything is written.
"""

from __future__ import annotations

import json
import struct

import pytest

from geo_common.errors import ErrorCategory, ValidationError

from geo_3d.server import Geo3DServer


def _server() -> Geo3DServer:
    return Geo3DServer()


_POINTS = [
    [-122.42, 37.77, 10.0],
    [-122.41, 37.78, 12.5],
    [-122.40, 37.79, 15.0],
]


def _parse_pnts_header(raw: bytes):
    assert raw[:4] == b"pnts"
    version, byte_length, ftj, ftb, btj, btb = struct.unpack_from("<IIIIII", raw, 4)
    ft_json = json.loads(raw[28 : 28 + ftj].decode("utf-8").rstrip())
    return version, byte_length, ftj, ftb, ft_json


# --- Happy path + round trip ------------------------------------------------


async def test_writes_valid_tileset_and_pnts(tmp_path):
    server = _server()
    try:
        result = await server.points_to_3d_tiles(
            points=_POINTS, output_dir=str(tmp_path)
        )
    finally:
        await server.aclose()

    assert result.point_count == 3
    assert result.has_colors is False
    assert "box" in result.bounding_volume
    assert result.geometric_error >= 0.0

    # tileset.json + .pnts exist on disk.
    tileset_path = tmp_path / "tileset.json"
    pnts_path = tmp_path / "points.pnts"
    assert tileset_path.is_file() and pnts_path.is_file()
    assert result.byte_length == pnts_path.stat().st_size

    # The written tileset round-trips through our own validator.
    report = await server.inspect_tileset(source=str(tileset_path))
    assert report.valid is True
    assert report.asset_version == "1.0"
    assert report.content_uris == ["points.pnts"]


async def test_pnts_header_and_point_count(tmp_path):
    server = _server()
    try:
        result = await server.points_to_3d_tiles(
            points=_POINTS, output_dir=str(tmp_path)
        )
    finally:
        await server.aclose()
    raw = (tmp_path / "points.pnts").read_bytes()
    version, byte_length, ftj, ftb, ft_json = _parse_pnts_header(raw)
    assert version == 1
    assert byte_length == len(raw) == result.byte_length
    assert ft_json["POINTS_LENGTH"] == 3
    assert "RTC_CENTER" in ft_json and ft_json["POSITION"]["byteOffset"] == 0
    # Feature-table binary starts 8-byte aligned (header 28 + ftJSON length).
    assert (28 + ftj) % 8 == 0
    # 3 points x 3 float32 = 36 bytes of positions (before any padding).
    assert ftb >= 36


async def test_colors_are_embedded(tmp_path):
    server = _server()
    try:
        result = await server.points_to_3d_tiles(
            points=_POINTS,
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            output_dir=str(tmp_path),
        )
    finally:
        await server.aclose()
    assert result.has_colors is True
    _, _, _, _, ft_json = _parse_pnts_header((tmp_path / "points.pnts").read_bytes())
    assert "RGB" in ft_json
    assert ft_json["RGB"]["byteOffset"] == 3 * 3 * 4  # after 3 float32 positions


async def test_mapping_points_and_geometric_error_override(tmp_path):
    server = _server()
    try:
        result = await server.points_to_3d_tiles(
            points=[{"x": 0, "y": 0, "z": 0}, {"x": 2, "y": 0, "z": 0}],
            output_dir=str(tmp_path),
            geometric_error=42.0,
        )
    finally:
        await server.aclose()
    assert result.point_count == 2
    assert result.geometric_error == 42.0


# --- Validation guards ------------------------------------------------------


async def test_empty_points_is_validation_error(tmp_path):
    server = _server()
    try:
        with pytest.raises(ValidationError) as exc:
            await server.points_to_3d_tiles(points=[], output_dir=str(tmp_path))
    finally:
        await server.aclose()
    assert exc.value.category is ErrorCategory.VALIDATION
    assert exc.value.detail.get("parameter") == "points"


async def test_non_numeric_point_is_validation_error(tmp_path):
    server = _server()
    try:
        with pytest.raises(ValidationError):
            await server.points_to_3d_tiles(
                points=[[0, 0, "z"]], output_dir=str(tmp_path)
            )
    finally:
        await server.aclose()


async def test_colors_length_mismatch_is_validation_error(tmp_path):
    server = _server()
    try:
        with pytest.raises(ValidationError) as exc:
            await server.points_to_3d_tiles(
                points=_POINTS, colors=[[255, 0, 0]], output_dir=str(tmp_path)
            )
    finally:
        await server.aclose()
    assert exc.value.detail.get("parameter") == "colors"


async def test_output_dir_that_is_a_file_is_validation_error(tmp_path):
    a_file = tmp_path / "not_a_dir"
    a_file.write_text("x")
    server = _server()
    try:
        with pytest.raises(ValidationError) as exc:
            await server.points_to_3d_tiles(points=_POINTS, output_dir=str(a_file))
    finally:
        await server.aclose()
    assert exc.value.detail.get("parameter") == "output_dir"


def test_tiler_tool_registered():
    server = Geo3DServer()
    assert "points_to_3d_tiles" in server.tool_names()
    assert len(server.catalog_entries()) == 4


# --- Octree subdivision -----------------------------------------------------

_MANY = [[float(i % 7), float((i // 7) % 7), float(i % 5)] for i in range(50)]


async def test_octree_partitions_points_across_tiles(tmp_path):
    """With max_points_per_tile set, the cloud is split into an octree of
    multiple tiles whose point counts sum to the total (additive refinement)."""
    server = _server()
    try:
        result = await server.points_to_3d_tiles(
            points=_MANY, output_dir=str(tmp_path), max_points_per_tile=8
        )
    finally:
        await server.aclose()

    assert result.point_count == 50
    assert result.tile_count > 1   # actually subdivided
    assert result.max_depth >= 1

    # Sum of POINTS_LENGTH across all written .pnts == total (no loss/dup).
    total = 0
    pnts_files = sorted(tmp_path.glob("content_*.pnts"))
    assert len(pnts_files) == result.tile_count
    for f in pnts_files:
        _, _, _, _, ft = _parse_pnts_header(f.read_bytes())
        total += ft["POINTS_LENGTH"]
    assert total == 50

    # The hierarchical tileset round-trips through inspect_tileset.
    report = await server.inspect_tileset(source=result.tileset_path)
    assert report.valid is True
    assert report.tile_count == result.tile_count
    assert report.max_depth == result.max_depth


async def test_octree_geometric_error_decreases_with_depth(tmp_path):
    """Root geometricError exceeds its children's (LOD refines downward)."""
    import json as _json

    server = _server()
    try:
        await server.points_to_3d_tiles(
            points=_MANY, output_dir=str(tmp_path), max_points_per_tile=8
        )
    finally:
        await server.aclose()
    tileset = _json.loads((tmp_path / "tileset.json").read_text())
    root = tileset["root"]
    assert root.get("children")
    for child in root["children"]:
        assert child["geometricError"] <= root["geometricError"]


async def test_small_cloud_stays_single_tile_even_with_max_set(tmp_path):
    server = _server()
    try:
        result = await server.points_to_3d_tiles(
            points=_POINTS, output_dir=str(tmp_path), max_points_per_tile=100
        )
    finally:
        await server.aclose()
    assert result.tile_count == 1 and result.max_depth == 0


async def test_bad_max_points_per_tile_is_validation_error(tmp_path):
    server = _server()
    try:
        with pytest.raises(ValidationError) as exc:
            await server.points_to_3d_tiles(
                points=_MANY, output_dir=str(tmp_path), max_points_per_tile=0
            )
    finally:
        await server.aclose()
    assert exc.value.detail.get("parameter") == "max_points_per_tile"

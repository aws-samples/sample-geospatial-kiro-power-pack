"""Tests for ``geo-3d`` ``dem_to_mesh`` (DEM grid -> glTF/GLB terrain mesh).

Verifies the written mesh round-trips through ``inspect_gltf`` (valid glTF 2.0),
the vertex/triangle counts are right, nodata cells create holes (dropped
triangles + re-indexed vertices), both .glb and .gltf outputs work, and the
validation guards fire before anything is written.
"""

from __future__ import annotations

import pytest

from geo_common.errors import ErrorCategory, ValidationError

from geo_3d.server import Geo3DServer

# A small 3x3 DEM and a bbox over a ~tiny extent.
_DEM = [
    [10.0, 11.0, 12.0],
    [10.5, 12.0, 13.0],
    [11.0, 12.5, 14.0],
]
_BBOX = (-122.45, 37.74, -122.44, 37.75)


def _server() -> Geo3DServer:
    return Geo3DServer()


async def test_dem_to_glb_round_trips_through_inspect_gltf(tmp_path):
    server = _server()
    out = tmp_path / "terrain.glb"
    try:
        result = await server.dem_to_mesh(
            elevations=_DEM, bbox=_BBOX, output_path=str(out)
        )
        report = await server.inspect_gltf(source=str(out))
    finally:
        await server.aclose()

    assert result.format == "glb"
    # 3x3 grid -> 9 vertices, 2x2 cells x 2 triangles = 8 triangles.
    assert result.vertex_count == 9
    assert result.triangle_count == 8
    assert result.has_holes is False
    assert out.stat().st_size == result.byte_length
    assert len(result.min_xyz) == 3 and len(result.max_xyz) == 3

    # The written GLB validates as glTF 2.0 with one mesh and three accessors
    # (POSITION, NORMAL, indices).
    assert report.valid is True and report.binary is True
    assert report.gltf_version == "2.0"
    assert report.counts.get("meshes") == 1
    assert report.counts.get("accessors") == 3


async def test_dem_to_gltf_json_output(tmp_path):
    server = _server()
    out = tmp_path / "terrain.gltf"
    try:
        result = await server.dem_to_mesh(
            elevations=_DEM, bbox=_BBOX, output_path=str(out)
        )
        report = await server.inspect_gltf(source=str(out))
    finally:
        await server.aclose()
    assert result.format == "gltf"
    assert report.valid is True and report.binary is False
    assert report.gltf_version == "2.0"


async def test_mesh_has_vertex_normals(tmp_path):
    """The mesh includes a NORMAL attribute (3 accessors: POSITION, NORMAL,
    indices), so viewers can shade it properly."""
    import json as _json
    import struct as _struct

    server = _server()
    out = tmp_path / "n.glb"
    try:
        await server.dem_to_mesh(elevations=_DEM, bbox=_BBOX, output_path=str(out))
    finally:
        await server.aclose()
    raw = out.read_bytes()
    # Parse the GLB JSON chunk and confirm the primitive declares a NORMAL.
    json_len = _struct.unpack_from("<I", raw, 12)[0]
    doc = _json.loads(raw[20 : 20 + json_len].decode("utf-8").rstrip())
    attrs = doc["meshes"][0]["primitives"][0]["attributes"]
    assert "POSITION" in attrs and "NORMAL" in attrs
    assert len(doc["accessors"]) == 3


async def test_vertical_exaggeration_scales_height(tmp_path):
    server = _server()
    try:
        flat = await server.dem_to_mesh(
            elevations=_DEM, bbox=_BBOX, output_path=str(tmp_path / "a.glb")
        )
        exag = await server.dem_to_mesh(
            elevations=_DEM, bbox=_BBOX, output_path=str(tmp_path / "b.glb"),
            vertical_exaggeration=10.0,
        )
    finally:
        await server.aclose()
    # Y is elevation; exaggeration multiplies the height span ~10x.
    flat_h = flat.max_xyz[1] - flat.min_xyz[1]
    exag_h = exag.max_xyz[1] - exag.min_xyz[1]
    assert exag_h == pytest.approx(flat_h * 10.0, rel=1e-5)


async def test_nodata_cells_create_holes(tmp_path):
    server = _server()
    dem = [
        [10.0, 11.0, 12.0],
        [10.5, None, 13.0],   # nodata in the centre
        [11.0, 12.5, 14.0],
    ]
    try:
        result = await server.dem_to_mesh(
            elevations=dem, bbox=_BBOX, output_path=str(tmp_path / "holes.glb")
        )
    finally:
        await server.aclose()
    # The centre nodata vertex touches all 8 triangles; every triangle that
    # references it is dropped, so fewer than 8 remain and holes are flagged.
    assert result.has_holes is True
    assert result.triangle_count < 8
    # The nodata vertex is not emitted (re-indexed away): < 9 vertices.
    assert result.vertex_count < 9


# --- Validation guards ------------------------------------------------------


async def test_too_small_grid_is_validation_error(tmp_path):
    server = _server()
    try:
        with pytest.raises(ValidationError) as exc:
            await server.dem_to_mesh(
                elevations=[[1.0, 2.0]], bbox=_BBOX,
                output_path=str(tmp_path / "x.glb"),
            )
    finally:
        await server.aclose()
    assert exc.value.detail.get("parameter") == "elevations"


async def test_non_rectangular_grid_is_validation_error(tmp_path):
    server = _server()
    try:
        with pytest.raises(ValidationError):
            await server.dem_to_mesh(
                elevations=[[1.0, 2.0, 3.0], [1.0, 2.0]], bbox=_BBOX,
                output_path=str(tmp_path / "x.glb"),
            )
    finally:
        await server.aclose()


async def test_inverted_bbox_is_validation_error(tmp_path):
    server = _server()
    try:
        with pytest.raises(ValidationError) as exc:
            await server.dem_to_mesh(
                elevations=_DEM, bbox=(-122.44, 37.75, -122.45, 37.74),
                output_path=str(tmp_path / "x.glb"),
            )
    finally:
        await server.aclose()
    assert exc.value.detail.get("parameter") == "bbox"


async def test_bad_extension_is_validation_error(tmp_path):
    server = _server()
    try:
        with pytest.raises(ValidationError) as exc:
            await server.dem_to_mesh(
                elevations=_DEM, bbox=_BBOX, output_path=str(tmp_path / "x.obj")
            )
    finally:
        await server.aclose()
    assert exc.value.detail.get("parameter") == "output_path"


async def test_non_positive_exaggeration_is_validation_error(tmp_path):
    server = _server()
    try:
        with pytest.raises(ValidationError) as exc:
            await server.dem_to_mesh(
                elevations=_DEM, bbox=_BBOX, output_path=str(tmp_path / "x.glb"),
                vertical_exaggeration=0,
            )
    finally:
        await server.aclose()
    assert exc.value.detail.get("parameter") == "vertical_exaggeration"


async def test_all_nodata_produces_no_triangles_validation_error(tmp_path):
    server = _server()
    dem = [[None, None], [None, None]]
    try:
        with pytest.raises(ValidationError) as exc:
            await server.dem_to_mesh(
                elevations=dem, bbox=_BBOX, output_path=str(tmp_path / "x.glb")
            )
    finally:
        await server.aclose()
    assert exc.value.detail.get("parameter") == "elevations"

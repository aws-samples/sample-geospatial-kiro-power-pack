"""Tests for ``geo-3d`` 3D-format inspection (tileset + glTF/GLB).

Pure-local inspection, so most tests pass documents inline or via a temp file;
the URL path is exercised through an ``httpx.MockTransport`` wired into the
shared :class:`HttpClient`. The repo's ``asyncio_mode = "auto"`` lets the
``async def`` tests run directly.
"""

from __future__ import annotations

import json
import struct

import httpx
import pytest

from geo_common.errors import ErrorCategory, NotFoundError, ValidationError
from geo_common.http import HttpClient
from geo_common.models import OpennessTier
from geo_common.retry import RetryPolicy

from geo_3d.models import GltfReport, TilesetReport
from geo_3d.server import INSTALL_COMMAND, Geo3DServer


def _server(handler=None) -> Geo3DServer:
    if handler is None:
        return Geo3DServer()
    client = HttpClient(RetryPolicy(max_attempts=1), transport=httpx.MockTransport(handler))
    return Geo3DServer(http=client)


_VALID_TILESET = {
    "asset": {"version": "1.1"},
    "geometricError": 500.0,
    "root": {
        "geometricError": 500.0,
        "refine": "ADD",
        "boundingVolume": {"region": [-1.3, 0.6, -1.2, 0.7, 0, 200]},
        "content": {"uri": "root.b3dm"},
        "children": [
            {"geometricError": 0.0, "boundingVolume": {"box": [0] * 12},
             "content": {"uri": "a.b3dm"}},
            {"geometricError": 0.0, "boundingVolume": {"box": [0] * 12},
             "content": {"uri": "b.b3dm"},
             "children": [
                 {"geometricError": 0.0, "boundingVolume": {"box": [0] * 12},
                  "content": {"uri": "c.b3dm"}},
             ]},
        ],
    },
}


def _glb(json_obj: dict, *, version: int = 2, good_magic: bool = True) -> bytes:
    """Build a minimal GLB (header + one JSON chunk) for tests."""
    chunk = json.dumps(json_obj).encode("utf-8")
    pad = (4 - len(chunk) % 4) % 4
    chunk += b" " * pad
    total = 12 + 8 + len(chunk)
    magic = 0x46546C67 if good_magic else 0x00000000
    header = struct.pack("<III", magic, version, total)
    chunk_hdr = struct.pack("<II", len(chunk), 0x4E4F534A)  # JSON chunk
    return header + chunk_hdr + chunk


# --- inspect_tileset --------------------------------------------------------


async def test_valid_tileset_inline():
    server = _server()
    try:
        report = await server.inspect_tileset(tileset=_VALID_TILESET)
    finally:
        await server.aclose()
    assert isinstance(report, TilesetReport)
    assert report.valid is True
    assert report.asset_version == "1.1"
    assert report.geometric_error == 500.0
    assert report.refine == "ADD"
    assert report.tile_count == 4  # root + 3 descendants
    assert report.max_depth == 2
    assert report.content_count == 4
    assert "root.b3dm" in report.content_uris
    assert report.issues == []


async def test_tileset_missing_asset_and_geometric_error_is_invalid():
    server = _server()
    try:
        report = await server.inspect_tileset(tileset={"root": {"boundingVolume": {}}})
    finally:
        await server.aclose()
    assert report.valid is False
    assert any("asset" in i for i in report.issues)
    assert any("geometricError" in i for i in report.issues)


async def test_tileset_bad_refine_is_flagged_but_walked():
    bad = json.loads(json.dumps(_VALID_TILESET))
    bad["root"]["refine"] = "SOMETIMES"
    server = _server()
    try:
        report = await server.inspect_tileset(tileset=bad)
    finally:
        await server.aclose()
    assert any("refine" in i for i in report.issues)
    assert report.tile_count == 4  # still walked the tree


async def test_tileset_inline_json_string():
    """A JSON string is auto-parsed, so chat clients that struggle with nested
    objects can pass the tileset as a stringified JSON."""
    server = _server()
    try:
        report = await server.inspect_tileset(
            tileset=json.dumps(_VALID_TILESET)
        )
    finally:
        await server.aclose()
    assert report.valid is True and report.tile_count == 4


async def test_tileset_bad_json_string_is_validation_error():
    server = _server()
    try:
        with pytest.raises(ValidationError) as exc:
            await server.inspect_tileset(tileset="not { valid json")
    finally:
        await server.aclose()
    assert exc.value.detail.get("parameter") == "tileset"


async def test_tileset_inline_and_source_both_supplied_is_validation_error():
    server = _server()
    try:
        with pytest.raises(ValidationError) as exc:
            await server.inspect_tileset(tileset=_VALID_TILESET, source="x.json")
    finally:
        await server.aclose()
    assert exc.value.category is ErrorCategory.VALIDATION


async def test_tileset_neither_supplied_is_validation_error():
    server = _server()
    try:
        with pytest.raises(ValidationError):
            await server.inspect_tileset()
    finally:
        await server.aclose()


async def test_tileset_from_file(tmp_path):
    path = tmp_path / "tileset.json"
    path.write_text(json.dumps(_VALID_TILESET))
    server = _server()
    try:
        report = await server.inspect_tileset(source=str(path))
    finally:
        await server.aclose()
    assert report.valid is True and report.tile_count == 4


async def test_tileset_from_url():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_VALID_TILESET)

    server = _server(handler)
    try:
        report = await server.inspect_tileset(source="https://example.com/tileset.json")
    finally:
        await server.aclose()
    assert report.valid is True


async def test_tileset_missing_file_is_not_found():
    server = _server()
    try:
        with pytest.raises(NotFoundError):
            await server.inspect_tileset(source="/tmp/geo3d_does_not_exist_42.json")
    finally:
        await server.aclose()


# --- inspect_gltf -----------------------------------------------------------

_GLTF_DOC = {
    "asset": {"version": "2.0", "generator": "geo-3d-test"},
    "scenes": [{"nodes": [0]}],
    "nodes": [{"mesh": 0}],
    "meshes": [{"primitives": [{}]}],
    "materials": [{}, {}],
    "accessors": [{}, {}, {}],
    "buffers": [{"byteLength": 1024}],
}


async def test_inspect_glb_binary(tmp_path):
    path = tmp_path / "model.glb"
    path.write_bytes(_glb(_GLTF_DOC))
    server = _server()
    try:
        report = await server.inspect_gltf(source=str(path))
    finally:
        await server.aclose()
    assert isinstance(report, GltfReport)
    assert report.valid is True
    assert report.binary is True
    assert report.gltf_version == "2.0"
    assert report.generator == "geo-3d-test"
    assert report.counts["meshes"] == 1
    assert report.counts["materials"] == 2
    assert report.counts["accessors"] == 3


async def test_inspect_gltf_json(tmp_path):
    path = tmp_path / "model.gltf"
    path.write_text(json.dumps(_GLTF_DOC))
    server = _server()
    try:
        report = await server.inspect_gltf(source=str(path))
    finally:
        await server.aclose()
    assert report.valid is True
    assert report.binary is False
    assert report.gltf_version == "2.0"
    assert report.counts["nodes"] == 1


async def test_inspect_glb_bad_magic_is_invalid(tmp_path):
    path = tmp_path / "bad.glb"
    # Starts with "glTF"? No - use a non-glTF binary that still has a header.
    path.write_bytes(struct.pack("<III", 0x00000000, 2, 20) + b"\x00\x00\x00\x00")
    server = _server()
    try:
        report = await server.inspect_gltf(source=str(path))
    finally:
        await server.aclose()
    # No "glTF" prefix -> treated as JSON glTF, which fails to parse -> invalid.
    assert report.valid is False
    assert report.issues


async def test_inspect_gltf_blank_source_is_validation_error():
    server = _server()
    try:
        with pytest.raises(ValidationError):
            await server.inspect_gltf(source="   ")
    finally:
        await server.aclose()


# --- server scaffold + catalog + credentials -------------------------------


def test_server_registers_both_tools():
    server = Geo3DServer()
    assert server.server_name == "geo-3d" and server.pillar == "B"
    assert set(server.tool_names()) == {
        "inspect_tileset", "inspect_gltf", "points_to_3d_tiles", "dem_to_mesh"
    }


def test_catalog_entries_open_provider():
    server = Geo3DServer()
    entries = {e.name: e for e in server.catalog_entries()}
    assert set(entries) == {
        "inspect_tileset", "inspect_gltf", "points_to_3d_tiles", "dem_to_mesh"
    }
    for entry in entries.values():
        assert entry.provider_server == "geo-3d"
        assert entry.openness_tier is OpennessTier.OPEN
        assert entry.install_command == INSTALL_COMMAND
        assert 1 <= len(entry.capability_description) <= 500


def test_open_server_no_credentials_and_starts():
    server = Geo3DServer()
    assert server.required_credentials() == []
    server.start(configured_keys=[])
    assert server.started is True

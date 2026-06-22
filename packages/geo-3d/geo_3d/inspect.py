"""3D format inspection/validation for ``geo-3d`` (Pillar B, direction 3).

Pure-Python, dependency-free inspectors for the two most common 3D geospatial
formats:

* :func:`inspect_tileset` - structurally validate an OGC **3D Tiles**
  ``tileset.json`` and summarize its tile tree; and
* :func:`inspect_gltf` - parse a **glTF 2.0 / GLB** asset header and summarize
  its contents.

Each accepts the document inline, from a local file path, or from an ``https``
URL (fetched through the shared :class:`~geo_common.http.HttpClient`). Inputs
are validated before any I/O (a malformed request raises a
:class:`~geo_common.errors.ValidationError`); a *parseable-but-invalid* document
returns a report with ``valid=False`` and populated ``issues`` rather than
raising, so callers can distinguish "bad request" from "this file is not a
valid tileset/glTF".
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from geo_common.errors import GeoError, NotFoundError, UpstreamError, ValidationError
from geo_common.http import HttpClient

from geo_3d.models import GltfReport, TilesetReport

__all__ = ["inspect_tileset", "inspect_gltf", "MAX_CONTENT_URIS", "MAX_TILES"]

_SERVER_NAME = "geo-3d"

#: Cap on URIs/tiles walked so a pathological tileset can't produce unbounded
#: output or run unbounded work.
MAX_CONTENT_URIS = 1000
MAX_TILES = 100_000

#: glTF/GLB binary magic and chunk types (little-endian uint32 values).
_GLB_MAGIC = 0x46546C67  # "glTF"
_GLB_CHUNK_JSON = 0x4E4F534A  # "JSON"

_GLTF_COUNT_KEYS = (
    "scenes",
    "nodes",
    "meshes",
    "materials",
    "accessors",
    "bufferViews",
    "buffers",
    "images",
    "textures",
    "samplers",
    "animations",
    "skins",
    "cameras",
)


# ---------------------------------------------------------------------------
# 3D Tiles tileset inspection
# ---------------------------------------------------------------------------


async def inspect_tileset(
    *,
    http: HttpClient,
    tileset: Optional[Any] = None,
    source: Optional[str] = None,
) -> TilesetReport:
    """Validate and summarize an OGC 3D Tiles ``tileset.json``.

    Provide the tileset **inline** (``tileset`` - a parsed JSON object **or** a
    JSON string that will be auto-parsed) or via ``source`` (a local file path
    or ``https`` URL); exactly one is required. A missing/blank/both-supplied
    input, or a ``source`` that does not parse as a JSON object, raises a
    :class:`~geo_common.errors.ValidationError`. A document that parses but is
    not a structurally valid tileset returns a :class:`TilesetReport` with
    ``valid=False`` and explanatory ``issues``.
    """
    # Accept a JSON string as the inline tileset (robust path for chat clients
    # that struggle to transmit nested objects as structured arguments).
    if isinstance(tileset, str):
        try:
            tileset = json.loads(tileset)
        except (ValueError, TypeError) as exc:
            raise ValidationError(
                "tileset string is not valid JSON: %s" % exc,
                source=_SERVER_NAME,
                detail={"parameter": "tileset"},
            )
    doc = await _resolve_json(http, inline=tileset, source=source, kind="tileset")
    return _report_tileset(doc)


def _report_tileset(doc: Any) -> TilesetReport:
    issues: List[str] = []
    if not isinstance(doc, dict):
        return TilesetReport(valid=False, issues=["tileset is not a JSON object"])

    asset = doc.get("asset")
    asset_version: Optional[str] = None
    if not isinstance(asset, dict):
        issues.append("missing 'asset' object")
    else:
        version = asset.get("version")
        if version is None:
            issues.append("missing 'asset.version'")
        else:
            asset_version = str(version)

    root = doc.get("root")
    geometric_error: Optional[float] = None
    refine: Optional[str] = None
    root_bv: Optional[Dict[str, object]] = None
    tile_count = 0
    max_depth = 0
    content_uris: List[str] = []

    if not isinstance(root, dict):
        issues.append("missing 'root' tile")
    else:
        ge = root.get("geometricError")
        if isinstance(ge, (int, float)) and not isinstance(ge, bool):
            geometric_error = float(ge)
        else:
            issues.append("root tile missing numeric 'geometricError'")
        bv = root.get("boundingVolume")
        if isinstance(bv, dict):
            root_bv = bv
        else:
            issues.append("root tile missing 'boundingVolume'")
        refine_raw = root.get("refine")
        if isinstance(refine_raw, str):
            refine = refine_raw.upper()
            if refine not in ("ADD", "REPLACE"):
                issues.append("root 'refine' must be ADD or REPLACE, got %r" % refine_raw)
        tile_count, max_depth, content_uris, walk_issues = _walk_tiles(root)
        issues.extend(walk_issues)

    valid = (
        asset_version is not None
        and isinstance(root, dict)
        and geometric_error is not None
    )
    return TilesetReport(
        valid=valid,
        asset_version=asset_version,
        geometric_error=geometric_error,
        refine=refine,
        root_bounding_volume=root_bv,
        tile_count=tile_count,
        max_depth=max_depth,
        content_count=len(content_uris),
        content_uris=content_uris[:MAX_CONTENT_URIS],
        issues=issues,
    )


def _walk_tiles(root: Dict[str, Any]) -> Tuple[int, int, List[str], List[str]]:
    """Walk the tile tree breadth-first, collecting counts, depth, URIs, issues."""
    issues: List[str] = []
    uris: List[str] = []
    tile_count = 0
    max_depth = 0
    # (tile, depth) queue; guard against runaway trees with MAX_TILES.
    queue: List[Tuple[Any, int]] = [(root, 0)]
    while queue:
        tile, depth = queue.pop(0)
        if tile_count >= MAX_TILES:
            issues.append("tile tree exceeds %d tiles; truncated" % MAX_TILES)
            break
        if not isinstance(tile, dict):
            issues.append("encountered a non-object tile at depth %d" % depth)
            continue
        tile_count += 1
        max_depth = max(max_depth, depth)
        for uri in _tile_content_uris(tile):
            if len(uris) < MAX_CONTENT_URIS:
                uris.append(uri)
        children = tile.get("children")
        if isinstance(children, list):
            for child in children:
                queue.append((child, depth + 1))
    return tile_count, max_depth, uris, issues


def _tile_content_uris(tile: Dict[str, Any]) -> List[str]:
    """Extract content URI(s) from a tile (``content.uri``/``url`` or ``contents``)."""
    out: List[str] = []
    content = tile.get("content")
    if isinstance(content, dict):
        uri = content.get("uri") or content.get("url")
        if isinstance(uri, str) and uri:
            out.append(uri)
    contents = tile.get("contents")
    if isinstance(contents, list):
        for entry in contents:
            if isinstance(entry, dict):
                uri = entry.get("uri") or entry.get("url")
                if isinstance(uri, str) and uri:
                    out.append(uri)
    return out


# ---------------------------------------------------------------------------
# glTF / GLB inspection
# ---------------------------------------------------------------------------


async def inspect_gltf(*, http: HttpClient, source: str) -> GltfReport:
    """Parse a glTF 2.0 / GLB asset and summarize it.

    ``source`` is a local file path or ``https`` URL to a ``.gltf`` (JSON) or
    ``.glb`` (binary) asset. A blank source raises a
    :class:`~geo_common.errors.ValidationError`; a fetch/read failure surfaces
    as a taxonomy-classified error. A file that is read but is not valid
    glTF/GLB returns a :class:`GltfReport` with ``valid=False`` and ``issues``.
    """
    raw = await _resolve_bytes(http, source=source)
    return _report_gltf(raw)


def _report_gltf(raw: bytes) -> GltfReport:
    byte_length = len(raw)
    if raw[:4] == b"glTF":
        return _report_glb(raw, byte_length)
    # Otherwise treat as JSON glTF.
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        return GltfReport(
            valid=False,
            binary=False,
            byte_length=byte_length,
            issues=["not GLB and not valid JSON glTF: %s" % exc],
        )
    return _report_gltf_json(doc, binary=False, byte_length=byte_length)


def _report_glb(raw: bytes, byte_length: int) -> GltfReport:
    issues: List[str] = []
    if len(raw) < 12:
        return GltfReport(
            valid=False, binary=True, byte_length=byte_length,
            issues=["GLB shorter than the 12-byte header"],
        )
    magic, version, declared_length = struct.unpack_from("<III", raw, 0)
    if magic != _GLB_MAGIC:
        issues.append("bad GLB magic")
    if declared_length != byte_length:
        issues.append(
            "GLB header length %d != actual %d" % (declared_length, byte_length)
        )
    # First chunk should be JSON.
    doc: Optional[Dict[str, Any]] = None
    if len(raw) >= 20:
        chunk_len, chunk_type = struct.unpack_from("<II", raw, 12)
        start = 20
        end = start + chunk_len
        if chunk_type != _GLB_CHUNK_JSON:
            issues.append("first GLB chunk is not a JSON chunk")
        elif end > len(raw):
            issues.append("GLB JSON chunk length exceeds file size")
        else:
            try:
                doc = json.loads(raw[start:end].decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                issues.append("GLB JSON chunk did not parse: %s" % exc)
    else:
        issues.append("GLB has no JSON chunk")

    report = _report_gltf_json(
        doc if isinstance(doc, dict) else {}, binary=True, byte_length=byte_length
    )
    # GLB-specific issues precede any JSON-content issues; keep both.
    report.issues = issues + report.issues
    # The container is GLB; its glTF version comes from the JSON chunk's asset.
    report.valid = report.valid and magic == _GLB_MAGIC and not any(
        "magic" in i or "chunk" in i for i in issues
    )
    if version != 2:
        report.issues.append("GLB container version is %d (expected 2)" % version)
    return report


def _report_gltf_json(doc: Any, *, binary: bool, byte_length: int) -> GltfReport:
    issues: List[str] = []
    if not isinstance(doc, dict):
        return GltfReport(
            valid=False, binary=binary, byte_length=byte_length,
            issues=["glTF is not a JSON object"],
        )
    asset = doc.get("asset")
    gltf_version: Optional[str] = None
    generator: Optional[str] = None
    if not isinstance(asset, dict):
        issues.append("missing 'asset' object")
    else:
        version = asset.get("version")
        if version is None:
            issues.append("missing 'asset.version'")
        else:
            gltf_version = str(version)
        if isinstance(asset.get("generator"), str):
            generator = asset["generator"]

    counts: Dict[str, int] = {}
    for key in _GLTF_COUNT_KEYS:
        value = doc.get(key)
        if isinstance(value, list):
            counts[key] = len(value)

    valid = gltf_version is not None
    return GltfReport(
        valid=valid,
        binary=binary,
        gltf_version=gltf_version,
        generator=generator,
        byte_length=byte_length,
        counts=counts,
        issues=issues,
    )


# ---------------------------------------------------------------------------
# Input resolution (inline / path / https URL)
# ---------------------------------------------------------------------------


async def _resolve_json(
    http: HttpClient,
    *,
    inline: Optional[Dict[str, Any]],
    source: Optional[str],
    kind: str,
) -> Any:
    """Resolve a JSON document from exactly one of inline content or a source."""
    if inline is not None and source is not None:
        raise ValidationError(
            "provide either inline %s or a source, not both" % kind,
            source=_SERVER_NAME,
            detail={"parameter": kind},
        )
    if inline is not None:
        if not isinstance(inline, dict):
            raise ValidationError(
                "%s must be a JSON object" % kind,
                source=_SERVER_NAME,
                detail={"parameter": kind},
            )
        return inline
    if not source:
        raise ValidationError(
            "provide either inline %s or a source (path or https URL)" % kind,
            source=_SERVER_NAME,
            detail={"parameter": "source"},
        )
    raw = await _resolve_bytes(http, source=source)
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise UpstreamError(
            "%s source did not contain valid JSON" % kind,
            source=_SERVER_NAME,
            original=str(exc),
        )


async def _resolve_bytes(http: HttpClient, *, source: str) -> bytes:
    """Read bytes from a local path or fetch them from an ``https`` URL."""
    if not isinstance(source, str) or not source.strip():
        raise ValidationError(
            "source must be a non-empty file path or https URL",
            source=_SERVER_NAME,
            detail={"parameter": "source"},
        )
    text = source.strip()
    if text.startswith("http://") or text.startswith("https://"):
        try:
            response = await http.get(text)
        except GeoError:
            raise
        if response.status_code == 404:
            raise NotFoundError(
                "3D source not found: %s (HTTP 404)" % text,
                source=_SERVER_NAME,
                detail={"status_code": 404},
            )
        if response.status_code >= 400:
            raise UpstreamError(
                "fetching the 3D source returned HTTP %d" % response.status_code,
                source=_SERVER_NAME,
                detail={"status_code": response.status_code},
            )
        return response.content
    path = Path(text)
    if not path.is_file():
        raise NotFoundError(
            "3D source file not found: %s" % text,
            source=_SERVER_NAME,
            detail={"parameter": "source"},
        )
    return path.read_bytes()

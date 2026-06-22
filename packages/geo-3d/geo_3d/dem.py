"""DEM grid -> glTF/GLB terrain mesh for ``geo-3d`` (Pillar B, direction 2).

:func:`dem_to_mesh` turns a digital-elevation-model grid (the kind
``geo-terrain``'s ``elevation`` tool returns for an extent) into a glTF 2.0
terrain mesh: each grid cell becomes two triangles, with elevations as vertex
heights. It is pure-Python (stdlib ``struct``/``json``/``array``/``base64``; no
native libraries) and round-trippable - the written asset validates with
:func:`~geo_3d.inspect.inspect_gltf`.

The mesh is built in a **local metric frame**: X = easting (m), Y = up
(elevation × vertical exaggeration), Z = southing (m), with the extent's
metres-per-degree derived at the bbox centre latitude. Nodata cells (``None``)
are honoured by dropping any triangle that touches them, leaving holes, and the
remaining vertices are re-indexed so the asset has no orphan vertices.

Output format follows ``output_path``'s extension: ``.glb`` (binary, default)
or ``.gltf`` (JSON with a self-contained base64 ``data:`` buffer). Inputs are
validated before anything is written.
"""

from __future__ import annotations

import array
import base64
import json
import math
import struct
import sys
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from geo_common.errors import UpstreamError, ValidationError

from geo_3d.models import MeshResult

__all__ = ["dem_to_mesh", "MAX_DEM_CELLS"]

_SERVER_NAME = "geo-3d"

#: Upper bound on DEM grid cells (vertices), bounding memory/output size.
MAX_DEM_CELLS = 4_000_000

_METRES_PER_DEG_LAT = 111_320.0

# glTF constants.
_GLB_MAGIC = b"glTF"
_GLB_VERSION = 2
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_FLOAT = 5126
_USHORT = 5123
_UINT = 5125
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_TRIANGLES = 4


def dem_to_mesh(
    *,
    elevations: Sequence[Sequence[Any]],
    bbox: Sequence[float],
    output_path: str,
    vertical_exaggeration: float = 1.0,
) -> MeshResult:
    """Mesh a DEM ``elevations`` grid over ``bbox`` into a glTF/GLB at ``output_path``.

    ``elevations`` is a rectangular 2D grid (rows north→south, columns
    west→east) of numbers, with ``None`` for nodata; it must be at least 2×2.
    ``bbox`` is ``(west, south, east, north)`` in EPSG:4326. ``output_path``
    ends in ``.glb`` (binary) or ``.gltf`` (self-contained JSON).
    ``vertical_exaggeration`` (>0) scales elevation.

    Returns a :class:`~geo_3d.models.MeshResult`. Raises a
    :class:`~geo_common.errors.ValidationError` for a malformed grid, an
    out-of-range/inverted bbox, a bad exaggeration, too many cells, or a
    non-writable/!.glb/.gltf ``output_path`` - before anything is written.
    """
    grid, height, width = _validate_grid(elevations)
    west, south, east, north = _validate_bbox(bbox)
    exag = _validate_exaggeration(vertical_exaggeration)
    out_path, fmt = _validate_output_path(output_path)

    # Metres per degree at the bbox centre latitude (simple equirectangular).
    mid_lat = math.radians((south + north) / 2.0)
    m_per_deg_lon = _METRES_PER_DEG_LAT * max(math.cos(mid_lat), 1e-6)
    width_m = (east - west) * m_per_deg_lon
    height_m = (north - south) * _METRES_PER_DEG_LAT

    positions, indices, used, has_holes = _build_mesh(
        grid, height, width, width_m, height_m, exag
    )
    if not indices:
        raise ValidationError(
            "the DEM produced no triangles (too many nodata cells or a "
            "degenerate grid); nothing to mesh",
            source=_SERVER_NAME,
            detail={"parameter": "elevations"},
        )

    min_xyz, max_xyz = _aabb(positions)
    gltf, binary = _build_gltf(positions, indices, len(used), min_xyz, max_xyz)

    try:
        if fmt == "glb":
            out_path.write_bytes(_to_glb(gltf, binary))
        else:
            data_uri = "data:application/octet-stream;base64," + base64.b64encode(
                binary
            ).decode("ascii")
            gltf["buffers"][0]["uri"] = data_uri
            out_path.write_text(json.dumps(gltf), encoding="utf-8")
    except OSError as exc:
        raise UpstreamError(
            "could not write the mesh: %s" % exc,
            source=_SERVER_NAME,
            original=str(exc),
        )

    return MeshResult(
        output_path=str(out_path),
        format=fmt,
        vertex_count=len(used),
        triangle_count=len(indices) // 3,
        byte_length=out_path.stat().st_size,
        min_xyz=list(min_xyz),
        max_xyz=list(max_xyz),
        has_holes=has_holes,
    )


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------


def _build_mesh(
    grid: List[List[Optional[float]]],
    height: int,
    width: int,
    width_m: float,
    height_m: float,
    exag: float,
) -> Tuple[List[Tuple[float, float, float]], List[int], List[int], bool]:
    """Build re-indexed vertex positions + triangle indices from the DEM grid.

    Returns ``(positions, indices, used_flat_ids, has_holes)``. A triangle is
    emitted only when its three corners are all valid; vertices are re-indexed
    so only referenced ones are kept (no orphans), giving a correct AABB.
    """
    remap: Dict[int, int] = {}
    positions: List[Tuple[float, float, float]] = []
    used: List[int] = []

    def _vertex(r: int, c: int) -> int:
        flat = r * width + c
        existing = remap.get(flat)
        if existing is not None:
            return existing
        elev = grid[r][c]
        x = (c / (width - 1)) * width_m
        y = float(elev) * exag
        z = (r / (height - 1)) * height_m  # north (r=0) -> 0, increasing south
        new_id = len(positions)
        positions.append((x, y, z))
        used.append(flat)
        remap[flat] = new_id
        return new_id

    indices: List[int] = []
    has_holes = False
    for r in range(height - 1):
        for c in range(width - 1):
            tl = grid[r][c]
            tr = grid[r][c + 1]
            bl = grid[r + 1][c]
            br = grid[r + 1][c + 1]
            # Triangle 1: tl, bl, tr ; Triangle 2: tr, bl, br.
            if tl is not None and bl is not None and tr is not None:
                indices.extend((_vertex(r, c), _vertex(r + 1, c), _vertex(r, c + 1)))
            else:
                has_holes = True
            if tr is not None and bl is not None and br is not None:
                indices.extend(
                    (_vertex(r, c + 1), _vertex(r + 1, c), _vertex(r + 1, c + 1))
                )
            else:
                has_holes = True
    return positions, indices, used, has_holes


def _aabb(
    positions: List[Tuple[float, float, float]]
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


# ---------------------------------------------------------------------------
# glTF / GLB assembly
# ---------------------------------------------------------------------------


def _le_float32(values: List[float]) -> bytes:
    arr = array.array("f", values)
    if sys.byteorder == "big":  # pragma: no cover
        arr.byteswap()
    return arr.tobytes()


def _le_uint(values: List[int], *, wide: bool) -> bytes:
    arr = array.array("I" if wide else "H", values)
    if sys.byteorder == "big":  # pragma: no cover
        arr.byteswap()
    return arr.tobytes()


def _compute_normals(
    positions: List[Tuple[float, float, float]], indices: List[int]
) -> List[Tuple[float, float, float]]:
    """Per-vertex normals: area-weighted average of incident face normals.

    Accumulates each triangle's (un-normalized) face normal into its three
    vertices, then normalizes. A vertex with no usable normal (degenerate
    neighbourhood) defaults to +Y (up), matching the mesh's up axis.
    """
    sums = [[0.0, 0.0, 0.0] for _ in positions]
    for t in range(0, len(indices), 3):
        i0, i1, i2 = indices[t], indices[t + 1], indices[t + 2]
        ax, ay, az = positions[i0]
        bx, by, bz = positions[i1]
        cx, cy, cz = positions[i2]
        e1 = (bx - ax, by - ay, bz - az)
        e2 = (cx - ax, cy - ay, cz - az)
        # Cross product e1 x e2 (magnitude == 2*triangle area -> area weighting).
        nx = e1[1] * e2[2] - e1[2] * e2[1]
        ny = e1[2] * e2[0] - e1[0] * e2[2]
        nz = e1[0] * e2[1] - e1[1] * e2[0]
        for idx in (i0, i1, i2):
            sums[idx][0] += nx
            sums[idx][1] += ny
            sums[idx][2] += nz
    normals: List[Tuple[float, float, float]] = []
    for nx, ny, nz in sums:
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length > 0.0:
            normals.append((nx / length, ny / length, nz / length))
        else:
            normals.append((0.0, 1.0, 0.0))
    return normals


def _build_gltf(
    positions: List[Tuple[float, float, float]],
    indices: List[int],
    vertex_count: int,
    min_xyz: Tuple[float, float, float],
    max_xyz: Tuple[float, float, float],
) -> Tuple[Dict[str, Any], bytes]:
    """Assemble the glTF JSON + a single binary buffer (positions, normals, indices)."""
    flat_pos: List[float] = [v for p in positions for v in p]
    pos_bin = _pad(_le_float32(flat_pos), 4, b"\x00")

    normals = _compute_normals(positions, indices)
    flat_norm: List[float] = [v for n in normals for v in n]
    norm_bin = _pad(_le_float32(flat_norm), 4, b"\x00")

    wide = vertex_count > 65535
    index_bin = _pad(_le_uint(indices, wide=wide), 4, b"\x00")

    pos_offset = 0
    norm_offset = len(pos_bin)
    index_offset = norm_offset + len(norm_bin)
    buffer = pos_bin + norm_bin + index_bin

    gltf: Dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "geo-3d"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1},
                        "indices": 2,
                        "mode": _TRIANGLES,
                        "material": 0,
                    }
                ]
            }
        ],
        "materials": [
            {
                "name": "terrain",
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.55, 0.55, 0.55, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
            }
        ],
        "buffers": [{"byteLength": len(buffer)}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": pos_offset,
                "byteLength": len(pos_bin),
                "target": _ARRAY_BUFFER,
            },
            {
                "buffer": 0,
                "byteOffset": norm_offset,
                "byteLength": len(norm_bin),
                "target": _ARRAY_BUFFER,
            },
            {
                "buffer": 0,
                "byteOffset": index_offset,
                "byteLength": len(index_bin),
                "target": _ELEMENT_ARRAY_BUFFER,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": _FLOAT,
                "count": vertex_count,
                "type": "VEC3",
                "min": [min_xyz[0], min_xyz[1], min_xyz[2]],
                "max": [max_xyz[0], max_xyz[1], max_xyz[2]],
            },
            {
                "bufferView": 1,
                "componentType": _FLOAT,
                "count": vertex_count,
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": _UINT if wide else _USHORT,
                "count": len(indices),
                "type": "SCALAR",
            },
        ],
    }
    return gltf, buffer


def _to_glb(gltf: Dict[str, Any], binary: bytes) -> bytes:
    """Pack glTF JSON + binary buffer into a binary GLB container."""
    json_bytes = _pad(json.dumps(gltf).encode("utf-8"), 4, b" ")
    bin_bytes = _pad(binary, 4, b"\x00")
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    out = bytearray()
    out += _GLB_MAGIC
    out += struct.pack("<II", _GLB_VERSION, total)
    out += struct.pack("<II", len(json_bytes), _CHUNK_JSON)
    out += json_bytes
    out += struct.pack("<II", len(bin_bytes), _CHUNK_BIN)
    out += bin_bytes
    return bytes(out)


def _pad(data: bytes, boundary: int, fill: bytes) -> bytes:
    remainder = len(data) % boundary
    if remainder == 0:
        return data
    return data + fill * (boundary - remainder)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_grid(
    elevations: Any,
) -> Tuple[List[List[Optional[float]]], int, int]:
    if isinstance(elevations, (str, bytes)) or not isinstance(elevations, Sequence):
        raise ValidationError(
            "elevations must be a 2D grid (list of rows of numbers)",
            source=_SERVER_NAME,
            detail={"parameter": "elevations"},
        )
    rows = list(elevations)
    if len(rows) < 2:
        raise ValidationError(
            "elevations must have at least 2 rows",
            source=_SERVER_NAME,
            detail={"parameter": "elevations"},
        )
    width: Optional[int] = None
    grid: List[List[Optional[float]]] = []
    total = 0
    for r, row in enumerate(rows):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ValidationError(
                "elevations row %d must be a list of numbers" % r,
                source=_SERVER_NAME,
                detail={"parameter": "elevations", "row": r},
            )
        cells = list(row)
        if width is None:
            width = len(cells)
            if width < 2:
                raise ValidationError(
                    "elevations must have at least 2 columns",
                    source=_SERVER_NAME,
                    detail={"parameter": "elevations"},
                )
        elif len(cells) != width:
            raise ValidationError(
                "elevations is not rectangular: row %d has %d cells, expected %d"
                % (r, len(cells), width),
                source=_SERVER_NAME,
                detail={"parameter": "elevations", "row": r},
            )
        out_row: List[Optional[float]] = []
        for c, value in enumerate(cells):
            if value is None:
                out_row.append(None)
            elif isinstance(value, bool) or not isinstance(value, Real):
                raise ValidationError(
                    "elevations[%d][%d] must be a number or null, got %r"
                    % (r, c, value),
                    source=_SERVER_NAME,
                    detail={"parameter": "elevations", "row": r, "col": c},
                )
            else:
                out_row.append(float(value))
        grid.append(out_row)
        total += len(out_row)
    if total > MAX_DEM_CELLS:
        raise ValidationError(
            "elevations has %d cells, exceeding the maximum of %d"
            % (total, MAX_DEM_CELLS),
            source=_SERVER_NAME,
            detail={"parameter": "elevations", "max": MAX_DEM_CELLS},
        )
    assert width is not None
    return grid, len(grid), width


def _validate_bbox(bbox: Any) -> Tuple[float, float, float, float]:
    if isinstance(bbox, (str, bytes)) or not isinstance(bbox, Sequence):
        raise ValidationError(
            "bbox must be (west, south, east, north)",
            source=_SERVER_NAME,
            detail={"parameter": "bbox"},
        )
    values = list(bbox)
    if len(values) != 4:
        raise ValidationError(
            "bbox must contain exactly four numbers (west, south, east, north)",
            source=_SERVER_NAME,
            detail={"parameter": "bbox"},
        )
    nums: List[float] = []
    for label, value in zip(("west", "south", "east", "north"), values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValidationError(
                "bbox %s must be a number" % label,
                source=_SERVER_NAME,
                detail={"parameter": "bbox"},
            )
        nums.append(float(value))
    west, south, east, north = nums
    if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
        raise ValidationError(
            "bbox longitudes must be within [-180, 180]",
            source=_SERVER_NAME,
            detail={"parameter": "bbox"},
        )
    if not (-90.0 <= south <= 90.0 and -90.0 <= north <= 90.0):
        raise ValidationError(
            "bbox latitudes must be within [-90, 90]",
            source=_SERVER_NAME,
            detail={"parameter": "bbox"},
        )
    if west >= east or south >= north:
        raise ValidationError(
            "bbox must have west < east and south < north",
            source=_SERVER_NAME,
            detail={"parameter": "bbox"},
        )
    return west, south, east, north


def _validate_exaggeration(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or value <= 0:
        raise ValidationError(
            "vertical_exaggeration must be a positive number",
            source=_SERVER_NAME,
            detail={"parameter": "vertical_exaggeration"},
        )
    return float(value)


def _validate_output_path(output_path: Any) -> Tuple[Path, str]:
    if not isinstance(output_path, str) or not output_path.strip():
        raise ValidationError(
            "output_path must be a non-empty .glb or .gltf file path",
            source=_SERVER_NAME,
            detail={"parameter": "output_path"},
        )
    path = Path(output_path.strip())
    suffix = path.suffix.lower()
    if suffix == ".glb":
        fmt = "glb"
    elif suffix == ".gltf":
        fmt = "gltf"
    else:
        raise ValidationError(
            "output_path must end in .glb or .gltf, got %r" % path.suffix,
            source=_SERVER_NAME,
            detail={"parameter": "output_path"},
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValidationError(
            "output_path directory is not creatable: %s" % exc,
            source=_SERVER_NAME,
            detail={"parameter": "output_path"},
        )
    return path, fmt

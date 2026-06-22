"""Point-cloud -> OGC 3D Tiles tiling for ``geo-3d`` (Pillar B, direction 1).

:func:`points_to_3d_tiles` writes a point cloud out as a valid OGC 3D Tiles
tileset: a ``tileset.json`` plus a binary ``.pnts`` (Point Cloud) tile. It pairs
with ``geo-pointcloud`` (which reads COPC/LAS into points) - the points are
passed in directly, so ``geo-3d`` stays dependency-free (pure ``struct``/``json``
with the stdlib ``array`` module for the binary body; no native libraries).

This first version emits a **single root tile** holding all points, which is a
fully conformant 3D Tiles 1.0 tileset (octree subdivision is a future
enhancement). The output is round-trippable: the written ``tileset.json``
validates with :func:`~geo_3d.inspect.inspect_tileset`.

Inputs are validated before anything is written (a malformed/empty point list,
a mismatched colors list, or a non-writable output directory raises a
:class:`~geo_common.errors.ValidationError`).
"""

from __future__ import annotations

import array
import json
import struct
import sys
from numbers import Real
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from geo_common.errors import UpstreamError, ValidationError

from geo_3d.models import PointTilesetResult

__all__ = ["points_to_3d_tiles", "MAX_POINTS"]

_SERVER_NAME = "geo-3d"

#: Upper bound on the number of points tiled in one call, to bound memory and
#: keep a single ``.pnts`` tile a reasonable size.
MAX_POINTS = 5_000_000

#: Octree guards: maximum recursion depth and total tiles written, to bound work
#: and output for pathological inputs (e.g. many coincident points).
MAX_OCTREE_DEPTH = 16
MAX_OCTREE_TILES = 200_000

#: ``.pnts`` 3D Tiles 1.0 binary constants.
_PNTS_MAGIC = b"pnts"
_PNTS_VERSION = 1


def points_to_3d_tiles(
    *,
    points: Sequence[Any],
    output_dir: str,
    colors: Optional[Sequence[Any]] = None,
    geometric_error: Optional[float] = None,
    content_name: str = "points.pnts",
    max_points_per_tile: Optional[int] = None,
) -> PointTilesetResult:
    """Tile ``points`` into a 3D Tiles tileset written under ``output_dir``.

    ``points`` is a sequence of ``[x, y, z]`` triples or ``{"x","y","z"}``
    mappings; ``colors`` (optional) is a matching sequence of ``[r, g, b]``
    (0-255) per point. The root bounding volume and a default ``geometric_error``
    (the bounding-box diagonal) are derived from the points; pass
    ``geometric_error`` to override the top-level value.

    By default a **single root tile** holds all points. Pass
    ``max_points_per_tile`` to instead build an **octree**: the cloud is
    recursively split into 8 octants until each node holds at most that many
    points, producing a level-of-detail hierarchy (coarse tiles load first and
    refine on zoom) of multiple ``.pnts`` files plus a nested ``tileset.json``.
    Every point lands in exactly one tile (additive refinement), so no point is
    lost or duplicated.

    Writes ``output_dir/tileset.json`` and the ``.pnts`` content file(s), and
    returns a :class:`PointTilesetResult`. Raises a
    :class:`~geo_common.errors.ValidationError` for an empty/malformed point
    list, a ``colors`` length mismatch, more than :data:`MAX_POINTS` points, a
    bad ``max_points_per_tile``/``geometric_error``, or a non-writable
    ``output_dir`` - before writing anything.
    """
    coords = _validate_points(points)
    rgb = _validate_colors(colors, len(coords))
    out_dir = _validate_output_dir(output_dir)
    ge_override = _validate_geometric_error(geometric_error)
    max_per_tile = _validate_max_per_tile(max_points_per_tile)

    (min_x, min_y, min_z), (max_x, max_y, max_z) = _bounds(coords)
    cx, cy, cz = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0)
    hx, hy, hz = ((max_x - min_x) / 2.0, (max_y - min_y) / 2.0, (max_z - min_z) / 2.0)
    box = [cx, cy, cz, hx, 0.0, 0.0, 0.0, hy, 0.0, 0.0, 0.0, hz]
    diagonal = (hx * hx + hy * hy + hz * hz) ** 0.5 * 2.0
    top_ge = ge_override if ge_override is not None else max(diagonal, 0.0)

    if max_per_tile is None or len(coords) <= max_per_tile:
        # Single root tile holding all points.
        pnts = _build_pnts(coords, (cx, cy, cz), rgb)
        content_path = out_dir / content_name
        _write_bytes(content_path, pnts)
        root_tile = {
            "boundingVolume": {"box": box},
            "geometricError": 0.0,
            "refine": "ADD",
            "content": {"uri": content_name},
        }
        tileset = {"asset": {"version": "1.0"}, "geometricError": top_ge, "root": root_tile}
        tileset_path = out_dir / "tileset.json"
        _write_text(tileset_path, json.dumps(tileset, indent=2))
        return PointTilesetResult(
            tileset_path=str(tileset_path),
            content_path=str(content_path),
            point_count=len(coords),
            bounding_volume={"box": box},
            geometric_error=top_ge,
            has_colors=rgb is not None,
            byte_length=len(pnts),
            tile_count=1,
            max_depth=0,
        )

    # Octree: recursively subdivide into a level-of-detail hierarchy.
    root_tile, state = _build_octree(
        coords,
        rgb,
        out_dir,
        max_per_tile,
        (min_x, min_y, min_z),
        (max_x, max_y, max_z),
    )
    tileset = {"asset": {"version": "1.0"}, "geometricError": top_ge, "root": root_tile}
    tileset_path = out_dir / "tileset.json"
    _write_text(tileset_path, json.dumps(tileset))
    return PointTilesetResult(
        tileset_path=str(tileset_path),
        content_path=str(out_dir / state["root_content"]),
        point_count=len(coords),
        bounding_volume={"box": box},
        geometric_error=top_ge,
        has_colors=rgb is not None,
        byte_length=state["bytes"],
        tile_count=state["count"],
        max_depth=state["max_depth"],
    )


def _write_bytes(path: Path, data: bytes) -> None:
    try:
        path.write_bytes(data)
    except OSError as exc:
        raise UpstreamError(
            "could not write %s: %s" % (path.name, exc),
            source=_SERVER_NAME,
            original=str(exc),
        )


def _write_text(path: Path, text: str) -> None:
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise UpstreamError(
            "could not write %s: %s" % (path.name, exc),
            source=_SERVER_NAME,
            original=str(exc),
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_points(points: Any) -> List[Tuple[float, float, float]]:
    if isinstance(points, (str, bytes)) or not isinstance(points, Sequence):
        raise ValidationError(
            "points must be a non-empty list of [x, y, z] coordinates",
            source=_SERVER_NAME,
            detail={"parameter": "points"},
        )
    items = list(points)
    if not items:
        raise ValidationError(
            "points must contain at least one [x, y, z] coordinate",
            source=_SERVER_NAME,
            detail={"parameter": "points"},
        )
    if len(items) > MAX_POINTS:
        raise ValidationError(
            "points exceeds the maximum of %d per call (%d given)"
            % (MAX_POINTS, len(items)),
            source=_SERVER_NAME,
            detail={"parameter": "points", "max": MAX_POINTS},
        )
    out: List[Tuple[float, float, float]] = []
    for index, point in enumerate(items):
        out.append(_one_point(point, index))
    return out


def _one_point(point: Any, index: int) -> Tuple[float, float, float]:
    if isinstance(point, dict):
        try:
            x, y, z = point["x"], point["y"], point["z"]
        except KeyError:
            raise ValidationError(
                "point %d mapping must have 'x', 'y', and 'z'" % index,
                source=_SERVER_NAME,
                detail={"parameter": "points", "index": index},
            )
    elif isinstance(point, (str, bytes)) or not isinstance(point, Sequence):
        raise ValidationError(
            "point %d must be an [x, y, z] triple or {x, y, z} mapping" % index,
            source=_SERVER_NAME,
            detail={"parameter": "points", "index": index},
        )
    else:
        values = list(point)
        if len(values) < 3:
            raise ValidationError(
                "point %d must have at least 3 coordinates (x, y, z)" % index,
                source=_SERVER_NAME,
                detail={"parameter": "points", "index": index},
            )
        x, y, z = values[0], values[1], values[2]
    return (_num(x, index), _num(y, index), _num(z, index))


def _num(value: Any, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(
            "point %d has a non-numeric coordinate: %r" % (index, value),
            source=_SERVER_NAME,
            detail={"parameter": "points", "index": index},
        )
    return float(value)


def _validate_colors(
    colors: Optional[Sequence[Any]], n_points: int
) -> Optional[List[Tuple[int, int, int]]]:
    if colors is None:
        return None
    if isinstance(colors, (str, bytes)) or not isinstance(colors, Sequence):
        raise ValidationError(
            "colors must be a list of [r, g, b] triples (0-255)",
            source=_SERVER_NAME,
            detail={"parameter": "colors"},
        )
    items = list(colors)
    if len(items) != n_points:
        raise ValidationError(
            "colors length (%d) must match points length (%d)"
            % (len(items), n_points),
            source=_SERVER_NAME,
            detail={"parameter": "colors"},
        )
    out: List[Tuple[int, int, int]] = []
    for index, color in enumerate(items):
        if isinstance(color, (str, bytes)) or not isinstance(color, Sequence):
            raise ValidationError(
                "color %d must be an [r, g, b] triple" % index,
                source=_SERVER_NAME,
                detail={"parameter": "colors", "index": index},
            )
        vals = list(color)
        if len(vals) < 3:
            raise ValidationError(
                "color %d must have 3 channels (r, g, b)" % index,
                source=_SERVER_NAME,
                detail={"parameter": "colors", "index": index},
            )
        rgb = []
        for channel in vals[:3]:
            if isinstance(channel, bool) or not isinstance(channel, Real):
                raise ValidationError(
                    "color %d has a non-numeric channel" % index,
                    source=_SERVER_NAME,
                    detail={"parameter": "colors", "index": index},
                )
            rgb.append(max(0, min(255, int(channel))))
        out.append((rgb[0], rgb[1], rgb[2]))
    return out


def _validate_geometric_error(value: Any) -> Optional[float]:
    """Validate an optional non-negative ``geometric_error`` override."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or value < 0:
        raise ValidationError(
            "geometric_error must be a non-negative number",
            source=_SERVER_NAME,
            detail={"parameter": "geometric_error"},
        )
    return float(value)


def _validate_max_per_tile(value: Any) -> Optional[int]:
    """Validate an optional positive-integer ``max_points_per_tile`` (octree)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(
            "max_points_per_tile must be a positive integer",
            source=_SERVER_NAME,
            detail={"parameter": "max_points_per_tile"},
        )
    return value


def _validate_output_dir(output_dir: Any) -> Path:
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValidationError(
            "output_dir must be a non-empty directory path",
            source=_SERVER_NAME,
            detail={"parameter": "output_dir"},
        )
    path = Path(output_dir.strip())
    if path.exists() and not path.is_dir():
        raise ValidationError(
            "output_dir exists but is not a directory: %s" % path,
            source=_SERVER_NAME,
            detail={"parameter": "output_dir"},
        )
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValidationError(
            "output_dir is not creatable/writable: %s" % exc,
            source=_SERVER_NAME,
            detail={"parameter": "output_dir"},
        )
    return path


# ---------------------------------------------------------------------------
# Octree subdivision (additive-refinement level-of-detail)
# ---------------------------------------------------------------------------


def _build_octree(
    coords: List[Tuple[float, float, float]],
    rgb: Optional[List[Tuple[int, int, int]]],
    out_dir: Path,
    max_per_tile: int,
    bmin: Tuple[float, float, float],
    bmax: Tuple[float, float, float],
):
    """Build an additive-refinement octree tileset, writing one ``.pnts`` per node.

    Each node keeps an evenly-spaced sample of up to ``max_per_tile`` points and
    pushes the remainder down into 8 octant children, so every point lands in
    exactly one tile (the node + descendants partition the cloud). Returns
    ``(root_tile_dict, state)`` where ``state`` carries the tile count, total
    bytes, max depth, and the root content filename.
    """
    state = {"count": 0, "bytes": 0, "max_depth": 0, "root_content": None}

    def _write(sample_idx: List[int], center: Tuple[float, float, float]) -> str:
        if state["count"] >= MAX_OCTREE_TILES:
            raise ValidationError(
                "octree exceeded the maximum of %d tiles; raise "
                "max_points_per_tile" % MAX_OCTREE_TILES,
                source=_SERVER_NAME,
                detail={"parameter": "max_points_per_tile"},
            )
        sub_coords = [coords[i] for i in sample_idx]
        sub_rgb = [rgb[i] for i in sample_idx] if rgb is not None else None
        pnts = _build_pnts(sub_coords, center, sub_rgb)
        name = "content_%d.pnts" % state["count"]
        _write_bytes(out_dir / name, pnts)
        if state["root_content"] is None:
            state["root_content"] = name
        state["count"] += 1
        state["bytes"] += len(pnts)
        return name

    def _recurse(idx: List[int], lo, hi, depth: int) -> dict:
        state["max_depth"] = max(state["max_depth"], depth)
        cx, cy, cz = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2)
        n = len(idx)
        if n <= max_per_tile or depth >= MAX_OCTREE_DEPTH:
            sample, remaining = idx, []
        else:
            sampled_positions = {(j * n) // max_per_tile for j in range(max_per_tile)}
            sample = [idx[p] for p in sampled_positions]
            remaining = [idx[p] for p in range(n) if p not in sampled_positions]

        content = _write(sample, (cx, cy, cz))
        dx, dy, dz = (hi[0] - lo[0]), (hi[1] - lo[1]), (hi[2] - lo[2])
        diag = (dx * dx + dy * dy + dz * dz) ** 0.5
        tile: dict = {
            "boundingVolume": {"box": _box(lo, hi)},
            "geometricError": 0.0 if not remaining else diag / 2.0,
            "refine": "ADD",
            "content": {"uri": content},
        }
        if remaining:
            children = []
            for child_idx, child_lo, child_hi in _octants(
                remaining, coords, lo, hi, (cx, cy, cz)
            ):
                if child_idx:
                    children.append(_recurse(child_idx, child_lo, child_hi, depth + 1))
            if children:
                tile["children"] = children
        return tile

    root = _recurse(list(range(len(coords))), bmin, bmax, 0)
    return root, state


def _box(lo, hi) -> List[float]:
    """3D Tiles ``box`` [center xyz, x/y/z half-axes] for an AABB lo..hi."""
    cx, cy, cz = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2)
    hx, hy, hz = ((hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, (hi[2] - lo[2]) / 2)
    return [cx, cy, cz, hx, 0.0, 0.0, 0.0, hy, 0.0, 0.0, 0.0, hz]


def _octants(idx, coords, lo, hi, center):
    """Partition ``idx`` into up to 8 octant buckets with their child AABBs."""
    cx, cy, cz = center
    buckets: List[List[int]] = [[] for _ in range(8)]
    for i in idx:
        x, y, z = coords[i]
        bit = (1 if x >= cx else 0) | (2 if y >= cy else 0) | (4 if z >= cz else 0)
        buckets[bit].append(i)
    out = []
    for bit in range(8):
        if not buckets[bit]:
            continue
        clo = (
            cx if bit & 1 else lo[0],
            cy if bit & 2 else lo[1],
            cz if bit & 4 else lo[2],
        )
        chi = (
            hi[0] if bit & 1 else cx,
            hi[1] if bit & 2 else cy,
            hi[2] if bit & 4 else cz,
        )
        out.append((buckets[bit], clo, chi))
    return out


# ---------------------------------------------------------------------------
# .pnts binary writer (3D Tiles 1.0 Point Cloud)
# ---------------------------------------------------------------------------


def _bounds(
    coords: List[Tuple[float, float, float]]
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _le_float32(values: List[float]) -> bytes:
    """Pack floats as little-endian float32 (efficiently, endian-correct)."""
    arr = array.array("f", values)
    if sys.byteorder == "big":  # pragma: no cover - little-endian CI
        arr.byteswap()
    return arr.tobytes()


def _build_pnts(
    coords: List[Tuple[float, float, float]],
    center: Tuple[float, float, float],
    rgb: Optional[List[Tuple[int, int, int]]],
) -> bytes:
    """Build a 3D Tiles 1.0 ``.pnts`` byte string for ``coords`` (RTC-centred)."""
    cx, cy, cz = center
    n = len(coords)
    # Positions relative to RTC_CENTER keep float32 precision for large CRS coords.
    flat_pos: List[float] = []
    for x, y, z in coords:
        flat_pos.extend((x - cx, y - cy, z - cz))
    position_bin = _le_float32(flat_pos)

    feature_table: dict = {
        "POINTS_LENGTH": n,
        "RTC_CENTER": [cx, cy, cz],
        "POSITION": {"byteOffset": 0},
    }
    binary = position_bin
    if rgb is not None:
        feature_table["RGB"] = {"byteOffset": len(position_bin)}
        color_bin = array.array("B", [c for triple in rgb for c in triple]).tobytes()
        binary += color_bin

    ft_json = json.dumps(feature_table).encode("utf-8")
    # The feature-table binary must start 8-byte aligned (header is 28 bytes).
    ft_json = _pad(ft_json, boundary=8, offset=28, fill=b" ")
    binary = _pad(binary, boundary=8, offset=28 + len(ft_json), fill=b"\x00")

    byte_length = 28 + len(ft_json) + len(binary)
    header = (
        _PNTS_MAGIC
        + struct.pack(
            "<IIIIII",
            _PNTS_VERSION,
            byte_length,
            len(ft_json),
            len(binary),
            0,  # batch table JSON length
            0,  # batch table binary length
        )
    )
    return header + ft_json + binary


def _pad(data: bytes, *, boundary: int, offset: int, fill: bytes) -> bytes:
    """Right-pad ``data`` with ``fill`` so ``offset + len`` is ``boundary``-aligned."""
    total = offset + len(data)
    remainder = total % boundary
    if remainder == 0:
        return data
    return data + fill * (boundary - remainder)

"""Read and write Cloud-Optimized Point Cloud (COPC) data (Requirement 8.11).

This module holds the logic-bearing core of ``geo-pointcloud``, kept free of
any MCP plumbing so it is easy to test and reuse:

* :class:`PointCloudBackend` - the pluggable read/write engine. ``geo-pointcloud``
  reads and writes COPC through a backend so the production native stack and a
  portable default can be swapped without changing the tool contract.
* :class:`PdalCopcBackend` - the **production** backend. It lazily imports PDAL
  and reads/writes standard ``.copc.laz`` via ``readers.copc`` /
  ``writers.copc``, using COPC's octree index for windowed reads.
* :class:`LocalCopcBackend` - the **default** backend. It writes a
  self-contained, lossless COPC-style container (a spatially Morton-ordered
  point block, mirroring COPC's octree organization) and reads it back,
  preserving the point set exactly. It depends only on the Python standard
  library, so the round-trip is exercisable everywhere the native PDAL/COPC
  stack is not installed. This mirrors how the rest of the pack provides a
  genuine, dependency-light default while leaving production weights/engines
  substitutable.
* :func:`read_pointcloud` / :func:`write_pointcloud` - the two tools. Writing a
  chunk and reading it back preserves the set of points (design Property 20);
  reading with a :class:`~geo_pointcloud.models.GeoWindow` returns only the
  points inside the window.

All input validation surfaces on the shared ``Error_Taxonomy``: a malformed
request raises a :class:`~geo_common.errors.ValidationError`, a missing source
file raises a :class:`~geo_common.errors.NotFoundError`, and an unmapped
backend/library failure raises a :class:`~geo_common.errors.UpstreamError`
retaining the original detail (Requirement 11.5).

Python 3.9 compatibility: this module uses ``from __future__ import
annotations`` and ``typing`` generics so it imports cleanly on 3.9+.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Any, Dict, List, Optional

from geo_common.errors import NotFoundError, UpstreamError, ValidationError

from geo_pointcloud.models import (
    OPTIONAL_DIMENSIONS,
    FormatResult,
    GeoWindow,
    PointCloudChunk,
    PointRecord,
)

__all__ = [
    "COPC_FORMAT",
    "PointCloudBackend",
    "LocalCopcBackend",
    "PdalCopcBackend",
    "default_backend",
    "read_pointcloud",
    "write_pointcloud",
]

#: The format label this server produces/consumes.
COPC_FORMAT = "COPC"

#: Source identifier placed on errors raised by this server.
_SOURCE = "geo-pointcloud"

#: Magic header identifying a :class:`LocalCopcBackend` container.
_LOCAL_MAGIC = "GEO-POINTCLOUD/COPC-LOCAL/1"


class PointCloudBackend:
    """Pluggable engine that reads and writes a point cloud at an ``href``.

    A backend persists a :class:`PointCloudChunk` to ``href`` and reads it back,
    optionally filtered to a :class:`GeoWindow`. The read/write *contract*
    (validation, windowing semantics, the preserved point set) lives in
    :func:`read_pointcloud` / :func:`write_pointcloud`; concrete backends only
    move bytes, so a production engine (PDAL/COPC) can replace the default
    without changing that contract.
    """

    name: str = "point-cloud-backend"

    def write(self, chunk: PointCloudChunk, href: str) -> int:  # pragma: no cover - interface
        """Persist ``chunk`` to ``href``; return the number of points written."""
        raise NotImplementedError

    def read(self, href: str, window: Optional[GeoWindow]) -> PointCloudChunk:  # pragma: no cover - interface
        """Read the cloud at ``href`` (optionally filtered to ``window``)."""
        raise NotImplementedError


def _morton_order(points: List[PointRecord]) -> List[PointRecord]:
    """Return ``points`` in a deterministic space-filling (Morton) order.

    COPC organizes points into a spatially-coherent octree rather than storing
    them in input order. We mirror that by sorting on a 3D Morton code computed
    from each point's position normalized into the cloud's bounding box. This
    makes the default container spatially coherent (and demonstrates that the
    round-trip preserves the point *set*, not the input sequence) while staying
    a pure, deterministic reordering.
    """
    if len(points) <= 1:
        return list(points)

    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    bits = 21  # 21 bits per axis fits a 63-bit interleaved Morton code.
    scale = (1 << bits) - 1

    def _norm(v: float, lo: float, hi: float) -> int:
        if hi <= lo:
            return 0
        q = int(round((v - lo) / (hi - lo) * scale))
        return max(0, min(scale, q))

    def _spread(value: int) -> int:
        # Interleave a ``bits``-bit integer with two zero bits between each bit.
        result = 0
        for i in range(bits):
            result |= ((value >> i) & 1) << (3 * i)
        return result

    def _key(point: PointRecord) -> int:
        cx = _spread(_norm(point.x, min_x, max_x))
        cy = _spread(_norm(point.y, min_y, max_y))
        cz = _spread(_norm(point.z, min_z, max_z))
        return (cx << 2) | (cy << 1) | cz

    # Sort by Morton code; ties keep input order (Python sort is stable).
    return [p for _, p in sorted(((_key(p), p) for p in points), key=lambda kp: kp[0])]


class LocalCopcBackend(PointCloudBackend):
    """Portable, lossless default backend (no native PDAL/COPC dependency).

    Writes a single self-contained container at ``href``: a magic line, a JSON
    header (CRS, dimension names, point count), and the point records as JSON,
    gzip-compressed. Points are Morton-ordered on write so the stored block is
    spatially coherent like a COPC octree. Reading parses the container back
    into a :class:`PointCloudChunk`; because every field is serialized exactly
    (Python's JSON round-trips ``float`` and ``int`` values losslessly), the
    point set read back equals the point set written (design Property 20).
    """

    name = "local-copc"

    def write(self, chunk: PointCloudChunk, href: str) -> int:
        ordered = _morton_order(chunk.points)
        header = {
            "magic": _LOCAL_MAGIC,
            "format": COPC_FORMAT,
            "crs": chunk.crs,
            "dimensions": list(OPTIONAL_DIMENSIONS),
            "point_count": len(ordered),
        }
        payload = {
            "header": header,
            "points": [_record_to_row(p) for p in ordered],
        }
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        directory = os.path.dirname(os.path.abspath(href))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with gzip.open(href, "wb") as fh:
            fh.write(raw)
        return len(ordered)

    def read(self, href: str, window: Optional[GeoWindow]) -> PointCloudChunk:
        if not os.path.exists(href):
            raise NotFoundError(
                "point-cloud source not found: %s" % href,
                source=_SOURCE,
                detail={"href": href},
            )
        try:
            with gzip.open(href, "rb") as fh:
                payload = json.loads(fh.read().decode("utf-8"))
        except (OSError, ValueError) as exc:
            raise UpstreamError(
                "could not read point-cloud container at %s" % href,
                source=_SOURCE,
                detail={"href": href},
                original=str(exc) or type(exc).__name__,
            ) from exc

        header = payload.get("header") if isinstance(payload, dict) else None
        if not isinstance(header, dict) or header.get("magic") != _LOCAL_MAGIC:
            raise UpstreamError(
                "file at %s is not a geo-pointcloud COPC container" % href,
                source=_SOURCE,
                detail={"href": href},
            )

        rows = payload.get("points") or []
        points = [_row_to_record(row) for row in rows]
        if window is not None:
            points = [p for p in points if window.contains(p)]
        crs = header.get("crs") or "EPSG:4326"
        return PointCloudChunk(points=points, crs=crs)


class PdalCopcBackend(PointCloudBackend):
    """Production backend: standard ``.copc.laz`` via PDAL (Requirement 8.11).

    Reads and writes Cloud-Optimized Point Cloud files with PDAL's
    ``readers.copc`` / ``writers.copc`` stages, using the COPC octree index to
    satisfy windowed reads server-side. PDAL is imported lazily so installing
    ``geo-pointcloud`` does not require the native PDAL stack; constructing this
    backend without PDAL available raises a taxonomy ``ValidationError``.
    """

    name = "pdal-copc"

    def __init__(self) -> None:
        self._pdal = _import_pdal()

    def write(self, chunk: PointCloudChunk, href: str) -> int:
        np = _import_numpy()
        array = _chunk_to_structured_array(chunk, np)
        pipeline = self._pdal.Pipeline(
            json.dumps(
                {
                    "pipeline": [
                        {"type": "writers.copc", "filename": href},
                    ]
                }
            ),
            arrays=[array],
        )
        try:
            pipeline.execute()
        except Exception as exc:  # noqa: BLE001 - mapped onto the taxonomy below
            raise UpstreamError(
                "PDAL failed to write COPC at %s" % href,
                source=_SOURCE,
                detail={"href": href},
                original=str(exc) or type(exc).__name__,
            ) from exc
        return chunk.point_count

    def read(self, href: str, window: Optional[GeoWindow]) -> PointCloudChunk:
        stages: List[Dict[str, Any]] = [{"type": "readers.copc", "filename": href}]
        if window is not None:
            stages[0]["bounds"] = _window_to_pdal_bounds(window)
        pipeline = self._pdal.Pipeline(json.dumps({"pipeline": stages}))
        try:
            pipeline.execute()
        except Exception as exc:  # noqa: BLE001 - mapped onto the taxonomy below
            raise UpstreamError(
                "PDAL failed to read COPC at %s" % href,
                source=_SOURCE,
                detail={"href": href},
                original=str(exc) or type(exc).__name__,
            ) from exc
        arrays = pipeline.arrays
        points: List[PointRecord] = []
        for array in arrays:
            points.extend(_structured_array_to_records(array))
        return PointCloudChunk(points=points)


def default_backend() -> PointCloudBackend:
    """The default read/write backend (portable, lossless local COPC)."""
    return LocalCopcBackend()


def write_pointcloud(
    *,
    points: PointCloudChunk,
    dst_href: str,
    backend: Optional[PointCloudBackend] = None,
) -> FormatResult:
    """Write a point-cloud chunk to COPC at ``dst_href`` (Requirement 8.11).

    Validates the destination and the chunk, then persists every point through
    the configured ``backend`` (default :class:`LocalCopcBackend`). Returns a
    :class:`FormatResult` naming the output, the produced format, the point
    count, and validity. A point cloud written here and read back with
    :func:`read_pointcloud` yields the same set of points (design Property 20).

    Raises :class:`~geo_common.errors.ValidationError` for an empty
    destination, and surfaces backend failures as taxonomy-classified errors.
    """
    if not dst_href or not dst_href.strip():
        raise ValidationError(
            "dst_href must be a non-empty path",
            source=_SOURCE,
            detail={"parameter": "dst_href"},
        )
    if not isinstance(points, PointCloudChunk):
        raise ValidationError(
            "points must be a PointCloudChunk",
            source=_SOURCE,
            detail={"parameter": "points", "type": type(points).__name__},
        )

    eng = backend if backend is not None else default_backend()
    written = eng.write(points, dst_href)
    return FormatResult(
        href=dst_href,
        format=COPC_FORMAT,
        valid=True,
        point_count=written,
        message="wrote %d point(s) via %s backend" % (written, eng.name),
    )


def read_pointcloud(
    *,
    copc_href: str,
    bounds: Optional[GeoWindow] = None,
    backend: Optional[PointCloudBackend] = None,
) -> PointCloudChunk:
    """Read a COPC point cloud, optionally clipped to ``bounds`` (Req 8.11).

    Reads the cloud at ``copc_href`` through the configured ``backend`` (default
    :class:`LocalCopcBackend`). When ``bounds`` is given, only the points inside
    that :class:`GeoWindow` are returned, exploiting COPC's spatial index.

    Raises :class:`~geo_common.errors.ValidationError` for an empty href and
    :class:`~geo_common.errors.NotFoundError` when the source does not exist;
    backend/library failures surface as taxonomy-classified errors.
    """
    if not copc_href or not copc_href.strip():
        raise ValidationError(
            "copc_href must be a non-empty path",
            source=_SOURCE,
            detail={"parameter": "copc_href"},
        )

    eng = backend if backend is not None else default_backend()
    return eng.read(copc_href, bounds)


# ---------------------------------------------------------------------------
# Serialization helpers (local backend)
# ---------------------------------------------------------------------------


def _record_to_row(point: PointRecord) -> List[Any]:
    """Encode a point as a compact positional row ``[x, y, z, *optional]``."""
    row: List[Any] = [point.x, point.y, point.z]
    for dim in OPTIONAL_DIMENSIONS:
        row.append(getattr(point, dim))
    return row


def _row_to_record(row: List[Any]) -> PointRecord:
    """Decode a positional row produced by :func:`_record_to_row`."""
    values: Dict[str, Any] = {"x": row[0], "y": row[1], "z": row[2]}
    for index, dim in enumerate(OPTIONAL_DIMENSIONS, start=3):
        if index < len(row):
            values[dim] = row[index]
    return PointRecord(**values)


# ---------------------------------------------------------------------------
# PDAL helpers (production backend)
# ---------------------------------------------------------------------------


def _import_pdal():
    """Import the ``pdal`` Python bindings or raise a taxonomy error."""
    try:
        import pdal  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without PDAL
        raise ValidationError(
            "the PDAL backend requires the 'pdal' package and the native PDAL "
            "library; install them or use the default local backend",
            source=_SOURCE,
            detail={"missing_dependency": "pdal"},
            original=str(exc) or type(exc).__name__,
        ) from exc
    return pdal


def _import_numpy():
    """Import ``numpy`` (a PDAL dependency) or raise a taxonomy error."""
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without numpy
        raise ValidationError(
            "the PDAL backend requires 'numpy'",
            source=_SOURCE,
            detail={"missing_dependency": "numpy"},
            original=str(exc) or type(exc).__name__,
        ) from exc
    return np


#: Mapping from a :class:`PointRecord` field to the PDAL/LAS dimension name.
_PDAL_DIMENSION_NAMES = {
    "x": "X",
    "y": "Y",
    "z": "Z",
    "intensity": "Intensity",
    "classification": "Classification",
    "return_number": "ReturnNumber",
    "number_of_returns": "NumberOfReturns",
    "red": "Red",
    "green": "Green",
    "blue": "Blue",
    "gps_time": "GpsTime",
}


def _chunk_to_structured_array(chunk: PointCloudChunk, np):
    """Build a PDAL-compatible structured numpy array from a chunk."""
    fields = ["x", "y", "z"]
    for dim in OPTIONAL_DIMENSIONS:
        if any(getattr(p, dim) is not None for p in chunk.points):
            fields.append(dim)
    dtype = []
    for field in fields:
        np_kind = "f8" if field in ("x", "y", "z", "gps_time") else "i4"
        dtype.append((_PDAL_DIMENSION_NAMES[field], np_kind))
    array = np.empty(len(chunk.points), dtype=dtype)
    for i, point in enumerate(chunk.points):
        for field in fields:
            value = getattr(point, field)
            array[_PDAL_DIMENSION_NAMES[field]][i] = 0 if value is None else value
    return array


def _structured_array_to_records(array) -> List[PointRecord]:
    """Convert a PDAL structured numpy array back into point records."""
    name_to_field = {v: k for k, v in _PDAL_DIMENSION_NAMES.items()}
    present = [name for name in array.dtype.names if name in name_to_field]
    records: List[PointRecord] = []
    for row in array:
        values: Dict[str, Any] = {}
        for name in present:
            field = name_to_field[name]
            raw = row[name]
            values[field] = float(raw) if field in ("x", "y", "z", "gps_time") else int(raw)
        records.append(PointRecord(**values))
    return records


def _window_to_pdal_bounds(window: GeoWindow) -> str:
    """Render a :class:`GeoWindow` as a PDAL ``bounds`` string."""
    if window.min_z is not None and window.max_z is not None:
        return "([%r, %r], [%r, %r], [%r, %r])" % (
            window.min_x,
            window.max_x,
            window.min_y,
            window.max_y,
            window.min_z,
            window.max_z,
        )
    return "([%r, %r], [%r, %r])" % (
        window.min_x,
        window.max_x,
        window.min_y,
        window.max_y,
    )

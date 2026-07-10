"""Read and write Cloud-Optimized Point Cloud (COPC) data (Requirement 8.11).

This module holds the logic-bearing core of ``geo-pointcloud``, kept free of
any MCP plumbing so it is easy to test and reuse:

* :class:`PointCloudBackend` - the pluggable read/write engine. ``geo-pointcloud``
  reads and writes through a backend so the real COPC/LAZ stacks and a portable
  fallback can be swapped without changing the tool contract.
* :class:`LaspyCopcBackend` - real **Cloud-Optimized Point Cloud / LAZ read**
  via ``laspy`` + ``lazrs`` (the light ``[copc]`` extra, pure-Python + a Rust
  wheel, no native GDAL/PDAL). Windowed reads of a real ``.copc.laz`` use COPC's
  octree index (fetching only the nodes overlapping the window); it also reads
  plain LAS/LAZ. Writes standards-compliant ``.laz`` via ``laspy`` (interoperable
  LAZ, not the COPC octree - use the PDAL backend for that).
* :class:`PdalCopcBackend` - real COPC **read and write** via PDAL's
  ``readers.copc`` / ``writers.copc`` (the heavier ``[pdal]`` extra + native
  PDAL library). This is the path that writes standards-compliant ``.copc.laz``.
* :class:`LocalContainerBackend` - the zero-dependency fallback. It writes a
  self-contained, lossless *local container* (gzip-JSON, Morton-ordered) and
  reads it back, preserving the point set exactly. It is **not** interoperable
  COPC (it is not readable by PDAL/QGIS) - it exists so the round-trip works
  everywhere with no native or third-party dependency, and its format is
  labelled :data:`LOCAL_CONTAINER_FORMAT`, never ``COPC``.
* :class:`SmartCopcBackend` - the default backend, which routes per operation:
  it reads the local container when handed one, else reads real COPC/LAZ via
  ``laspy``/PDAL when available; and it writes real COPC via PDAL when the
  ``[pdal]`` extra is present, else the portable local container.
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
import importlib.util
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
    "LAZ_FORMAT",
    "LOCAL_CONTAINER_FORMAT",
    "PointCloudBackend",
    "LocalContainerBackend",
    "LocalCopcBackend",
    "LaspyCopcBackend",
    "PdalCopcBackend",
    "SmartCopcBackend",
    "default_backend",
    "read_pointcloud",
    "write_pointcloud",
]

#: Format label for standards-compliant Cloud-Optimized Point Cloud output
#: (produced by the real PDAL backend; also the source format the readers read).
COPC_FORMAT = "COPC"

#: Format label for standards-compliant LAZ output (the ``laspy`` writer emits
#: interoperable LAZ, which is not the COPC octree variant).
LAZ_FORMAT = "LAZ"

#: Format label for the portable, lossless local container - explicitly NOT
#: interoperable COPC (not readable by PDAL/QGIS). Honest labelling so nothing
#: the fallback produces is called ``COPC``.
LOCAL_CONTAINER_FORMAT = "GEO-POINTCLOUD-LOCAL"

#: Source identifier placed on errors raised by this server.
_SOURCE = "geo-pointcloud"

#: Magic header identifying a :class:`LocalContainerBackend` container.
_LOCAL_MAGIC = "GEO-POINTCLOUD/LOCAL-CONTAINER/1"


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

    #: The format label :func:`write_pointcloud` reports for output this backend
    #: produces. Overridden per backend so the reported format is honest
    #: (real ``COPC`` / ``LAZ`` vs the portable local container).
    output_format: str = COPC_FORMAT

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


class LocalContainerBackend(PointCloudBackend):
    """Portable, lossless fallback backend (no native or third-party dependency).

    Writes a single self-contained *local container* at ``href``: a magic line, a
    JSON header (CRS, dimension names, point count), and the point records as
    JSON, gzip-compressed. Points are Morton-ordered on write so the stored block
    is spatially coherent. Reading parses the container back into a
    :class:`PointCloudChunk`; because every field is serialized exactly (Python's
    JSON round-trips ``float`` and ``int`` losslessly), the point set read back
    equals the point set written (design Property 20).

    This container is **not** interoperable Cloud-Optimized Point Cloud: it is
    not readable by PDAL, QGIS, or other COPC tooling, so it reports
    :data:`LOCAL_CONTAINER_FORMAT` (never ``COPC``). It exists purely as a
    dependency-free fallback for environments without the ``[copc]`` (laspy) or
    ``[pdal]`` extra; install one of those to read/write real COPC/LAZ.
    """

    name = "local-container"
    output_format = LOCAL_CONTAINER_FORMAT

    def write(self, chunk: PointCloudChunk, href: str) -> int:
        ordered = _morton_order(chunk.points)
        header = {
            "magic": _LOCAL_MAGIC,
            "format": LOCAL_CONTAINER_FORMAT,
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
                "file at %s is not a geo-pointcloud local container; to read a "
                "real Cloud-Optimized Point Cloud/LAZ install the [copc] "
                "(laspy) or [pdal] extra" % href,
                source=_SOURCE,
                detail={"href": href},
            )

        rows = payload.get("points") or []
        points = [_row_to_record(row) for row in rows]
        if window is not None:
            points = [p for p in points if window.contains(p)]
        crs = header.get("crs") or "EPSG:4326"
        return PointCloudChunk(points=points, crs=crs)


#: Backward-compatible alias for the pre-rename name. The class no longer claims
#: to produce COPC (it writes the local container); the alias is kept so
#: existing imports keep working.
LocalCopcBackend = LocalContainerBackend


class LaspyCopcBackend(PointCloudBackend):
    """Real COPC/LAZ **read** (and LAZ write) via ``laspy`` (the ``[copc]`` extra).

    Reads a standards-compliant Cloud-Optimized Point Cloud (``.copc.laz``) using
    COPC's octree index for windowed reads - only the octree nodes overlapping
    the :class:`GeoWindow` are decoded, the point-cloud analogue of the pack's
    byte-range COG reads - and also reads plain LAS/LAZ (decoded in full, then
    filtered to the window). Writing emits interoperable ``.laz`` via ``laspy``;
    that is standard LAZ, **not** the COPC octree variant (use
    :class:`PdalCopcBackend` for standards-compliant COPC output), so it reports
    :data:`LAZ_FORMAT`.

    ``laspy`` + ``lazrs`` are pure-Python plus a Rust wheel (no native GDAL/PDAL)
    and imported lazily; a missing dependency raises a clear error naming the
    extra. Remote (``s3://`` / ``https://``) sources are read through ``fsspec``
    when it is installed.
    """

    name = "laspy-copc"
    output_format = LAZ_FORMAT

    #: LAS point format 3 carries RGB + GPS time, covering our optional dims.
    _WRITE_POINT_FORMAT = 3
    #: Fine coordinate scale so positions round-trip tightly (LAS stores scaled
    #: 32-bit integers; 1e-6 keeps sub-millimetre precision for typical ranges).
    _WRITE_SCALE = 1e-6

    def __init__(self) -> None:
        self._laspy = _import_laspy()

    # -- read ------------------------------------------------------------

    def read(self, href: str, window: Optional[GeoWindow]) -> PointCloudChunk:
        laspy = self._laspy
        remote = _is_remote(href)
        if not remote and not os.path.exists(href):
            raise NotFoundError(
                "point-cloud source not found: %s" % href,
                source=_SOURCE,
                detail={"href": href},
            )
        try:
            return self._read_copc(href, window)
        except (NotFoundError, ValidationError, UpstreamError):
            raise
        except Exception:
            # Not a COPC file (no COPC VLR) - fall back to plain LAS/LAZ.
            try:
                return self._read_plain(href, window)
            except Exception as exc:  # noqa: BLE001 - map to taxonomy
                raise UpstreamError(
                    "laspy failed to read point cloud at %s" % href,
                    source=_SOURCE,
                    detail={"href": href},
                    original=str(exc) or type(exc).__name__,
                ) from exc

    def _read_copc(self, href: str, window: Optional[GeoWindow]) -> PointCloudChunk:
        laspy = self._laspy
        opener = self._open_stream(href)
        with opener as stream:
            reader = laspy.CopcReader.open(stream)
            if window is not None:
                bounds = self._to_bounds(window)
                record = reader.query(bounds=bounds)
            else:
                record = reader.query()
            crs = _crs_from_header(reader.header)
            points = _laspy_points_to_records(record)
        if window is not None:
            # COPC query is node-granular; refine to the exact window.
            points = [p for p in points if window.contains(p)]
        return PointCloudChunk(points=points, crs=crs)

    def _read_plain(self, href: str, window: Optional[GeoWindow]) -> PointCloudChunk:
        laspy = self._laspy
        opener = self._open_stream(href)
        with opener as stream:
            las = laspy.read(stream)
        crs = _crs_from_header(las.header)
        points = _laspy_points_to_records(las)
        if window is not None:
            points = [p for p in points if window.contains(p)]
        return PointCloudChunk(points=points, crs=crs)

    def _open_stream(self, href: str):
        """Open ``href`` as a binary, seekable stream (local, or remote via fsspec)."""
        if _is_remote(href):
            try:
                import fsspec  # type: ignore
            except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
                raise UpstreamError(
                    "reading a remote point cloud (%s) requires 'fsspec' (and a "
                    "filesystem backend such as s3fs); install it or use a local "
                    "path" % href,
                    source=_SOURCE,
                    detail={"href": href},
                    original=str(exc),
                ) from exc
            return fsspec.open(href, "rb")
        return open(href, "rb")

    def _to_bounds(self, window: GeoWindow):
        import numpy as np

        big = 1e30
        min_z = window.min_z if window.min_z is not None else -big
        max_z = window.max_z if window.max_z is not None else big
        from laspy.copc import Bounds

        return Bounds(
            mins=np.array([window.min_x, window.min_y, min_z], dtype="float64"),
            maxs=np.array([window.max_x, window.max_y, max_z], dtype="float64"),
        )

    # -- write -----------------------------------------------------------

    def write(self, chunk: PointCloudChunk, href: str) -> int:
        laspy = self._laspy
        import numpy as np

        header = laspy.LasHeader(point_format=self._WRITE_POINT_FORMAT)
        header.scales = [self._WRITE_SCALE] * 3
        header.offsets = [0.0, 0.0, 0.0]
        las = laspy.LasData(header)
        n = len(chunk.points)
        if n:
            las.x = np.array([p.x for p in chunk.points], dtype="float64")
            las.y = np.array([p.y for p in chunk.points], dtype="float64")
            las.z = np.array([p.z for p in chunk.points], dtype="float64")
            _assign_optional_dims(las, chunk.points, np)
        directory = os.path.dirname(os.path.abspath(href))
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            las.write(href)
        except Exception as exc:  # noqa: BLE001 - map to taxonomy
            raise UpstreamError(
                "laspy failed to write LAZ at %s" % href,
                source=_SOURCE,
                detail={"href": href},
                original=str(exc) or type(exc).__name__,
            ) from exc
        return n


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


class SmartCopcBackend(PointCloudBackend):
    """Default backend that routes each operation to the best available engine.

    * **read** - a local file that is our gzip local container is read by
      :class:`LocalContainerBackend`; anything else (a real ``.copc.laz`` /
      ``.laz``, or any remote source) is read as real COPC/LAZ via
      :class:`LaspyCopcBackend` (``[copc]`` extra) or :class:`PdalCopcBackend`
      (``[pdal]`` extra) when available. With neither extra installed, only the
      local container is readable (a clear error is raised otherwise).
    * **write** - standards-compliant COPC via :class:`PdalCopcBackend` when the
      ``[pdal]`` extra is present; otherwise the portable, lossless local
      container (labelled :data:`LOCAL_CONTAINER_FORMAT`, not ``COPC``).

    This keeps the zero-dependency default fully functional and honest while
    making real COPC/LAZ I/O "just work" as soon as an extra is installed.
    """

    name = "smart-copc"

    def __init__(self) -> None:
        self._local = LocalContainerBackend()

    @property  # type: ignore[override]
    def output_format(self) -> str:  # noqa: D401 - reports the writer's format
        return COPC_FORMAT if _pdal_available() else LOCAL_CONTAINER_FORMAT

    def _writer(self) -> PointCloudBackend:
        if _pdal_available():
            return PdalCopcBackend()
        return self._local

    def _reader_for(self, href: str) -> PointCloudBackend:
        if not _is_remote(href) and os.path.exists(href) and _is_gzip_file(href):
            return self._local
        if _laspy_available():
            return LaspyCopcBackend()
        if _pdal_available():
            return PdalCopcBackend()
        # No real-COPC reader available; the local backend raises a clear error
        # (or NotFoundError for a missing file).
        return self._local

    def write(self, chunk: PointCloudChunk, href: str) -> int:
        return self._writer().write(chunk, href)

    def read(self, href: str, window: Optional[GeoWindow]) -> PointCloudChunk:
        return self._reader_for(href).read(href, window)


def default_backend() -> PointCloudBackend:
    """The default backend: :class:`SmartCopcBackend` (routes per operation).

    Reads real COPC/LAZ via ``laspy``/PDAL when installed (and the local
    container when handed one); writes real COPC via PDAL when installed, else
    the portable local container.
    """
    return SmartCopcBackend()


def _pdal_available() -> bool:
    """Whether the PDAL bindings are importable (checked without importing)."""
    return importlib.util.find_spec("pdal") is not None


def _laspy_available() -> bool:
    """Whether ``laspy`` is importable (checked without importing)."""
    return importlib.util.find_spec("laspy") is not None


def _is_remote(href: str) -> bool:
    """Whether ``href`` is a remote URL (``s3://`` / ``https://`` / ...)."""
    return "://" in href and not href.startswith("file://")


def _is_gzip_file(href: str) -> bool:
    """Whether the local file at ``href`` begins with the gzip magic bytes."""
    try:
        with open(href, "rb") as fh:
            return fh.read(2) == b"\x1f\x8b"
    except OSError:  # pragma: no cover - defensive
        return False


def write_pointcloud(
    *,
    points: PointCloudChunk,
    dst_href: str,
    backend: Optional[PointCloudBackend] = None,
) -> FormatResult:
    """Write a point-cloud chunk to COPC at ``dst_href`` (Requirement 8.11).

    Validates the destination and the chunk, then persists every point through
    the configured ``backend`` (default :class:`SmartCopcBackend`: real COPC via
    PDAL when the ``[pdal]`` extra is installed, otherwise the portable local
    container). The returned :class:`FormatResult` names the output, the format
    actually produced (``COPC`` / ``LAZ`` / the local-container label), the
    point count, and validity. A point cloud written here and read back with
    :func:`read_pointcloud` yields the same set of points (design Property 20;
    real COPC/LAZ round-trips within LAS storage precision).

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
    produced = getattr(eng, "output_format", COPC_FORMAT)
    return FormatResult(
        href=dst_href,
        format=produced,
        valid=True,
        point_count=written,
        message="wrote %d point(s) as %s via %s backend"
        % (written, produced, eng.name),
    )


def read_pointcloud(
    *,
    copc_href: str,
    bounds: Optional[GeoWindow] = None,
    backend: Optional[PointCloudBackend] = None,
) -> PointCloudChunk:
    """Read a COPC point cloud, optionally clipped to ``bounds`` (Req 8.11).

    Reads the cloud at ``copc_href`` through the configured ``backend`` (default
    :class:`SmartCopcBackend`: a real ``.copc.laz`` / ``.laz`` - local or remote
    - via ``laspy``/PDAL when installed, or the pack's local container). When
    ``bounds`` is given, only the points inside that :class:`GeoWindow` are
    returned, exploiting COPC's octree index for a real COPC source.

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


def _import_laspy():
    """Import ``laspy`` or raise a taxonomy error naming the ``[copc]`` extra."""
    try:
        import laspy  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without laspy
        raise UpstreamError(
            "reading/writing real COPC/LAZ requires the 'laspy' + 'lazrs' "
            "packages: pip install 'geo-pointcloud[copc]'",
            source=_SOURCE,
            detail={"missing_dependency": "laspy"},
            original=str(exc) or type(exc).__name__,
        ) from exc
    return laspy


#: Maps a LAS/laspy dimension name to the :class:`PointRecord` field it fills.
_LASPY_DIMENSION_FIELDS = {
    "intensity": "intensity",
    "classification": "classification",
    "return_number": "return_number",
    "number_of_returns": "number_of_returns",
    "red": "red",
    "green": "green",
    "blue": "blue",
    "gps_time": "gps_time",
}

#: Optional dims stored as floats (the rest are integers).
_FLOAT_OPTIONAL_DIMS = frozenset({"gps_time"})


def _crs_from_header(header: Any) -> str:
    """Best-effort CRS string from a laspy header, defaulting to EPSG:4326."""
    try:
        crs = header.parse_crs()
    except Exception:  # pragma: no cover - header may carry no CRS
        crs = None
    if crs is None:
        return "EPSG:4326"
    try:
        epsg = crs.to_epsg()
        if epsg:
            return "EPSG:%d" % epsg
        return crs.to_wkt()
    except Exception:  # pragma: no cover - pyproj variations
        return "EPSG:4326"


def _laspy_points_to_records(record: Any) -> "List[PointRecord]":
    """Convert a laspy point record / LasData into :class:`PointRecord` values."""
    xs = [float(v) for v in record.x]
    ys = [float(v) for v in record.y]
    zs = [float(v) for v in record.z]
    present = set()
    try:
        present = {d.name for d in record.point_format.dimensions}
    except Exception:  # pragma: no cover - defensive
        present = set()

    optional: "Dict[str, list]" = {}
    for las_name, field in _LASPY_DIMENSION_FIELDS.items():
        if las_name in present:
            try:
                optional[field] = list(getattr(record, las_name))
            except Exception:  # pragma: no cover - dimension access variance
                pass

    records: "List[PointRecord]" = []
    for i in range(len(xs)):
        values: Dict[str, Any] = {"x": xs[i], "y": ys[i], "z": zs[i]}
        for field, series in optional.items():
            if i < len(series):
                raw = series[i]
                values[field] = float(raw) if field in _FLOAT_OPTIONAL_DIMS else int(raw)
        records.append(PointRecord(**values))
    return records


def _assign_optional_dims(las: Any, points: "List[PointRecord]", np) -> None:
    """Assign the optional LAS dimensions present in ``points`` onto ``las``."""
    for field in OPTIONAL_DIMENSIONS:
        if not any(getattr(p, field) is not None for p in points):
            continue
        is_float = field in _FLOAT_OPTIONAL_DIMS
        default = 0.0 if is_float else 0
        values = [
            (getattr(p, field) if getattr(p, field) is not None else default)
            for p in points
        ]
        dtype = "float64" if is_float else "int32"
        try:
            setattr(las, field, np.array(values, dtype=dtype))
        except Exception:  # pragma: no cover - dimension not in point format
            pass


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

"""PyProj-backed CRS reprojection for ``geo-ops`` (Requirement 8.1).

This module holds the pure, local CRS-transform logic the ``geo-ops`` server
exposes as the ``transform_crs`` MCP tool (task 4.1):

* :func:`transform_crs` - reproject a :class:`~geo_ops.models.GeoJSONGeometry`
  from a source CRS to a destination CRS (Requirement 8.1). The function is
  synchronous and side-effect free; the server (see :mod:`geo_ops.server`)
  wraps it as an ``async`` MCP tool.

The transform walks the GeoJSON coordinate structure recursively, reprojecting
every position with a single :class:`pyproj.Transformer` built with
``always_xy=True`` (so coordinates are always interpreted/returned as
``(x/lon, y/lat)`` regardless of the CRS's declared axis order). ``Multi*`` and
``GeometryCollection`` geometries are handled by recursion.

Round-trip fidelity (Requirement 15.2; design Property 1): reprojecting a
geometry to a target CRS and back to the source CRS reproduces the original
geometry within a small, documented tolerance expressed in source-CRS units -
see :data:`CRS_ROUNDTRIP_TOLERANCE` and ``tests/test_crs_roundtrip_property.py``.

Malformed input - an unknown/unparseable CRS, or coordinates that are not a
valid GeoJSON position structure - raises a ``geo_common``
:class:`~geo_common.errors.ValidationError` so failures surface on the shared
``Error_Taxonomy`` (Req 7.12 / 8.10 / 15.5) rather than as raw PyProj
exceptions, and no partial output is produced.
"""

from __future__ import annotations

from typing import Any, Callable, List

from geo_common.errors import ValidationError

from geo_ops.models import GeoJSONGeometry

__all__ = [
    "CRS_ROUNDTRIP_TOLERANCE",
    "transform_crs",
]

#: Documented round-trip tolerance for :func:`transform_crs`, expressed in
#: *source-CRS units* (design Property 1 / Requirement 15.2).
#:
#: Reprojecting a geometry to a target CRS and back to its source CRS is not
#: bit-exact: each forward/inverse projection step introduces floating-point
#: rounding (and, for ellipsoidal projections, iterative-inverse) error. Across
#: well-behaved CRS pairs and within each projection's area of validity the
#: accumulated per-vertex deviation stays far below this bound; ``1e-6`` units
#: (~0.1 mm in a metric CRS, ~0.1 m at the equator for degrees) is a
#: conservative, meaningful ceiling.
CRS_ROUNDTRIP_TOLERANCE: float = 1e-6


def _is_position(coords: Any) -> bool:
    """True when ``coords`` is a single GeoJSON position ``[x, y, ...]``.

    A position is a sequence whose first element is a real number (and not a
    ``bool``, which is an ``int`` subclass), e.g. ``[lon, lat]`` or
    ``[x, y, z]``. Nested coordinate arrays (rings, multi-geometries) have a
    list/tuple as their first element instead.
    """
    return (
        isinstance(coords, (list, tuple))
        and len(coords) >= 2
        and isinstance(coords[0], (int, float))
        and not isinstance(coords[0], bool)
        and isinstance(coords[1], (int, float))
        and not isinstance(coords[1], bool)
    )


def _transform_coords(
    coords: Any, fn: Callable[[float, float], "tuple[float, float]"]
) -> Any:
    """Recursively reproject a GeoJSON coordinate structure.

    Applies ``fn`` (an ``(x, y) -> (x, y)`` reprojection) to every position,
    preserving the nesting and any extra ordinates (e.g. ``z``) beyond the
    first two. Raises :class:`ValidationError` if ``coords`` is neither a
    position nor a list of coordinate structures.
    """
    if _is_position(coords):
        new_x, new_y = fn(coords[0], coords[1])
        return [new_x, new_y, *list(coords[2:])]
    if isinstance(coords, (list, tuple)):
        return [_transform_coords(c, fn) for c in coords]
    raise ValidationError(
        "geometry coordinates are not a valid GeoJSON position structure",
        source="geo-ops",
        detail={"offending_value_type": type(coords).__name__},
    )


def _make_transformer(src_crs: str, dst_crs: str, *, source: str):
    """Build a PyProj ``Transformer`` from ``src_crs`` to ``dst_crs``.

    Uses ``always_xy=True`` so both directions interpret/emit coordinates as
    ``(x/lon, y/lat)`` regardless of the CRS's declared axis order. An
    unknown/unparseable CRS raises a taxonomy :class:`ValidationError`.
    """
    from pyproj import Transformer
    from pyproj.exceptions import CRSError, ProjError

    try:
        return Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    except (CRSError, ProjError, ValueError) as exc:  # unknown/unparseable CRS
        raise ValidationError(
            "could not build a CRS transform from %r to %r: %s"
            % (src_crs, dst_crs, str(exc) or type(exc).__name__),
            source=source,
            detail={"src_crs": src_crs, "dst_crs": dst_crs},
            original=str(exc) or type(exc).__name__,
        ) from exc


def transform_crs(
    geometry: GeoJSONGeometry,
    *,
    src_crs: str,
    dst_crs: str,
    source: str = "geo-ops",
) -> GeoJSONGeometry:
    """Reproject ``geometry`` from ``src_crs`` to ``dst_crs`` (Req 8.1).

    Returns a **new** :class:`GeoJSONGeometry` of the same type with every
    coordinate reprojected; the input geometry is not mutated (the function is
    pure). ``src_crs`` / ``dst_crs`` are any PyProj-acceptable CRS identifiers
    (e.g. ``"EPSG:4326"``, ``"EPSG:3857"``, a PROJ/WKT string).

    Round-tripping through a target CRS and back reproduces the original within
    :data:`CRS_ROUNDTRIP_TOLERANCE` source-CRS units (Req 15.2; Property 1).

    Raises :class:`~geo_common.errors.ValidationError` (taxonomy
    ``validation``) for an unknown/unparseable CRS or a malformed coordinate
    structure, producing no partial output.
    """
    transformer = _make_transformer(src_crs, dst_crs, source=source)
    return _transform_geometry(geometry, transformer.transform, source=source)


def _transform_geometry(
    geometry: GeoJSONGeometry,
    fn: Callable[[float, float], "tuple[float, float]"],
    *,
    source: str,
) -> GeoJSONGeometry:
    """Reproject a single geometry (recursing into ``GeometryCollection``)."""
    if geometry.type == "GeometryCollection":
        members: List[GeoJSONGeometry] = geometry.geometries or []
        return GeoJSONGeometry(
            type="GeometryCollection",
            geometries=[_transform_geometry(m, fn, source=source) for m in members],
        )

    if geometry.coordinates is None:
        raise ValidationError(
            "geometry of type %r has no coordinates to transform" % (geometry.type,),
            source=source,
            detail={"geometry_type": geometry.type},
        )

    new_coords = _transform_coords(geometry.coordinates, fn)
    return GeoJSONGeometry(type=geometry.type, coordinates=new_coords)

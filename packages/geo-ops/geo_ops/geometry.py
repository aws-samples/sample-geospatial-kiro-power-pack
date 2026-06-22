"""GEOS-backed geometry operations for ``geo-ops`` (Requirements 8.2, 8.3).

This module holds the pure, local geometry logic the ``geo-ops`` server exposes
as MCP tools (task 4.3):

* :func:`validate_geometry` - report whether a geometry is valid and, when it
  is not, *why* (Requirements 8.2, 8.3; design Property 16). Backed by Shapely
  / GEOS ``is_valid`` + ``explain_validity``.
* :func:`spatial_join` - attribute join of two feature collections by a spatial
  predicate (``intersects``, ``within``, ``contains`` ...), via GeoPandas
  ``sjoin``.
* :func:`overlay` - geometric set operation (``intersection``, ``union``,
  ``difference``, ``symmetric_difference``, ``identity``) between two feature
  collections, via GeoPandas ``overlay``.

The functions are synchronous and side-effect free; the server (see
:mod:`geo_ops.server`) wraps them as ``async`` MCP tools. Malformed input
(coordinates Shapely cannot interpret, unknown predicate/op) raises a
``geo_common`` :class:`~geo_common.errors.ValidationError` so failures surface
on the shared ``Error_Taxonomy`` (Req 7.12 / 8.10 / 15.5) rather than as raw
library exceptions.
"""

from __future__ import annotations

from typing import Any, Dict, List

from geo_common.errors import ValidationError

from geo_ops.models import (
    Feature,
    FeatureCollection,
    GeoJSONGeometry,
    GeometryValidity,
)

__all__ = [
    "SPATIAL_PREDICATES",
    "OVERLAY_OPS",
    "validate_geometry",
    "spatial_join",
    "overlay",
    "buffer",
    "convex_hull",
]

#: The spatial predicates :func:`spatial_join` accepts (the binary predicates
#: GeoPandas ``sjoin`` supports).
SPATIAL_PREDICATES = frozenset(
    {
        "intersects",
        "contains",
        "within",
        "touches",
        "crosses",
        "overlaps",
        "covers",
        "covered_by",
        "contains_properly",
    }
)

#: The set operations :func:`overlay` accepts (the GeoPandas ``overlay`` ops).
OVERLAY_OPS = frozenset(
    {"intersection", "union", "identity", "symmetric_difference", "difference"}
)


def _to_shape(geometry: GeoJSONGeometry, *, source: str = "geo-ops"):
    """Convert a :class:`GeoJSONGeometry` to a Shapely geometry.

    Raises :class:`~geo_common.errors.ValidationError` (taxonomy
    ``validation``) when the coordinates cannot be interpreted as a geometry,
    so malformed input fails on the shared taxonomy (Req 8.10 / 15.5).
    """
    from shapely.geometry import shape

    try:
        return shape(geometry.to_geojson())
    except Exception as exc:  # noqa: BLE001 - normalize any GEOS/parse failure
        raise ValidationError(
            "geometry could not be parsed: %s" % (str(exc) or type(exc).__name__),
            source=source,
            detail={"geometry_type": geometry.type},
            original=str(exc) or type(exc).__name__,
        ) from exc


def validate_geometry(geometry: GeoJSONGeometry, *, source: str = "geo-ops") -> GeometryValidity:
    """Report a geometry's validity, with a reason when invalid (Req 8.2, 8.3).

    Returns a :class:`GeometryValidity` whose ``valid`` flag reflects the
    GEOS-backed ``is_valid`` test (Requirement 8.2). When the geometry is
    invalid, ``reason`` carries a **non-empty** human explanation from GEOS
    (e.g. ``"Self-intersection[2 2]"``) and is ``None`` when the geometry is
    valid (Requirement 8.3; design Property 16).

    Malformed input that Shapely cannot interpret as a geometry at all raises a
    :class:`~geo_common.errors.ValidationError` (it is not a "geometry" whose
    validity can be reported).
    """
    from shapely.validation import explain_validity

    shp = _to_shape(geometry, source=source)
    is_valid = bool(shp.is_valid)
    if is_valid:
        return GeometryValidity(valid=True, reason=None)

    # Invalid: surface the GEOS explanation, guaranteeing a non-empty reason.
    reason = explain_validity(shp)
    if not reason or reason == "Valid Geometry":
        # Defensive: GEOS reported invalid but gave no/positive text. Never
        # leave an invalid geometry without a reason (Req 8.3 / Property 16).
        reason = "invalid geometry"
    return GeometryValidity(valid=False, reason=reason)


def _to_geodataframe(fc: FeatureCollection, *, source: str):
    """Build a GeoDataFrame from a FeatureCollection (geometry + properties)."""
    import geopandas as gpd
    from shapely.geometry import shape

    geometries = []
    records: List[Dict[str, Any]] = []
    for feature in fc.features:
        if feature.geometry is None:
            geometries.append(None)
        else:
            try:
                geometries.append(shape(feature.geometry.to_geojson()))
            except Exception as exc:  # noqa: BLE001
                raise ValidationError(
                    "feature geometry could not be parsed: %s"
                    % (str(exc) or type(exc).__name__),
                    source=source,
                    original=str(exc) or type(exc).__name__,
                ) from exc
        records.append(dict(feature.properties))

    # crs is intentionally left unset (None) on both sides so a join/overlay of
    # two CRS-less collections does not raise a CRS-mismatch error.
    return gpd.GeoDataFrame(records, geometry=geometries)


def _from_geodataframe(gdf) -> FeatureCollection:
    """Convert a GeoDataFrame back into a FeatureCollection."""
    from shapely.geometry import mapping

    geom_col = gdf.geometry.name
    features: List[Feature] = []
    for _, row in gdf.iterrows():
        geom = row[geom_col]
        geometry = None
        if geom is not None and not _is_missing(geom) and not geom.is_empty:
            geometry = GeoJSONGeometry.from_geojson(mapping(geom))
        properties = {
            key: _clean_value(value)
            for key, value in row.items()
            if key != geom_col
        }
        features.append(Feature(geometry=geometry, properties=properties))
    return FeatureCollection(features=features)


def _is_missing(value: Any) -> bool:
    """True when ``value`` is a pandas/NumPy missing marker (NaN/NaT/None)."""
    try:
        import pandas as pd

        result = pd.isna(value)
        # pd.isna on an array-like returns an array; treat those as present.
        return bool(result) if isinstance(result, bool) else False
    except Exception:  # noqa: BLE001
        return value is None


def _clean_value(value: Any) -> Any:
    """Normalize a cell value for JSON-friendly properties (NaN -> None)."""
    if _is_missing(value):
        return None
    # Unwrap NumPy scalar types to native Python where possible.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:  # noqa: BLE001
            return value
    return value


def spatial_join(
    left: FeatureCollection,
    right: FeatureCollection,
    *,
    predicate: str = "intersects",
    source: str = "geo-ops",
) -> FeatureCollection:
    """Attribute-join ``left`` to ``right`` by a spatial ``predicate``.

    For each left feature, attaches the attributes of every right feature whose
    geometry satisfies ``predicate`` (default ``"intersects"``), keeping the
    left geometry. Implemented with GeoPandas ``sjoin`` (an inner join), so a
    left feature with no spatial match contributes no output row.

    Raises :class:`~geo_common.errors.ValidationError` for an unsupported
    predicate.
    """
    if predicate not in SPATIAL_PREDICATES:
        raise ValidationError(
            "unsupported spatial predicate %r; expected one of: %s"
            % (predicate, ", ".join(sorted(SPATIAL_PREDICATES))),
            source=source,
            detail={"predicate": predicate},
        )

    import geopandas as gpd

    left_gdf = _to_geodataframe(left, source=source)
    right_gdf = _to_geodataframe(right, source=source)

    try:
        joined = gpd.sjoin(left_gdf, right_gdf, how="inner", predicate=predicate)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            "spatial join failed: %s" % (str(exc) or type(exc).__name__),
            source=source,
            original=str(exc) or type(exc).__name__,
        ) from exc

    # Drop the join bookkeeping column GeoPandas adds.
    joined = joined.drop(columns=[c for c in ("index_right", "index_left") if c in joined.columns])
    return _from_geodataframe(joined)


def overlay(
    a: FeatureCollection,
    b: FeatureCollection,
    *,
    op: str,
    source: str = "geo-ops",
) -> FeatureCollection:
    """Compute a geometric set operation between ``a`` and ``b``.

    ``op`` is one of ``intersection``, ``union``, ``difference``,
    ``symmetric_difference``, or ``identity`` (the GeoPandas ``overlay``
    operations). Returns the resulting features with merged attributes.

    Raises :class:`~geo_common.errors.ValidationError` for an unsupported op.
    """
    if op not in OVERLAY_OPS:
        raise ValidationError(
            "unsupported overlay op %r; expected one of: %s"
            % (op, ", ".join(sorted(OVERLAY_OPS))),
            source=source,
            detail={"op": op},
        )

    import geopandas as gpd

    a_gdf = _to_geodataframe(a, source=source)
    b_gdf = _to_geodataframe(b, source=source)

    try:
        result = gpd.overlay(a_gdf, b_gdf, how=op, keep_geom_type=False)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            "overlay failed: %s" % (str(exc) or type(exc).__name__),
            source=source,
            original=str(exc) or type(exc).__name__,
        ) from exc

    return _from_geodataframe(result)


def _require_finite_number(value: Any, *, parameter: str, source: str) -> float:
    """Validate ``value`` is a finite real number, returning it as ``float``.

    Raises :class:`~geo_common.errors.ValidationError` (taxonomy
    ``validation``, naming the bad ``parameter``) for ``bool`` (which is an
    ``int`` subclass but not a meaningful magnitude), non-numeric input, or a
    NaN/inf value. Raised **before** any geometry work (Req 8.10 / 15.5).
    """
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            "%s must be a number, got %s" % (parameter, type(value).__name__),
            source=source,
            detail={"parameter": parameter},
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(
            "%s must be finite, got %r" % (parameter, value),
            source=source,
            detail={"parameter": parameter},
        )
    return number


def buffer(
    geometry: GeoJSONGeometry,
    *,
    distance: float,
    resolution: int = 16,
    source: str = "geo-ops",
) -> GeoJSONGeometry:
    """Compute a buffer polygon around ``geometry`` at ``distance`` (Shapely/GEOS).

    Returns the buffered geometry. ``distance`` is in the units of the
    geometry's coordinates and may be negative (an erosion); ``resolution``
    controls the number of segments used to approximate a quarter circle.

    Raises :class:`~geo_common.errors.ValidationError` for a non-numeric or
    non-finite ``distance``, a non-positive ``resolution``, or coordinates
    Shapely cannot interpret as a geometry — all before any geometry work.
    """
    from shapely.geometry import mapping

    dist = _require_finite_number(distance, parameter="distance", source=source)
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 1:
        raise ValidationError(
            "resolution must be a positive integer, got %r" % (resolution,),
            source=source,
            detail={"parameter": "resolution"},
        )

    shp = _to_shape(geometry, source=source)
    try:
        buffered = shp.buffer(dist, quad_segs=resolution)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            "buffer failed: %s" % (str(exc) or type(exc).__name__),
            source=source,
            original=str(exc) or type(exc).__name__,
        ) from exc
    return GeoJSONGeometry.from_geojson(mapping(buffered))


def convex_hull(geometry: GeoJSONGeometry, *, source: str = "geo-ops") -> GeoJSONGeometry:
    """Compute the convex hull of ``geometry`` (Shapely/GEOS).

    Returns the smallest convex geometry that contains the input. Raises
    :class:`~geo_common.errors.ValidationError` for coordinates Shapely cannot
    interpret as a geometry.
    """
    from shapely.geometry import mapping

    shp = _to_shape(geometry, source=source)
    try:
        hull = shp.convex_hull
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            "convex_hull failed: %s" % (str(exc) or type(exc).__name__),
            source=source,
            original=str(exc) or type(exc).__name__,
        ) from exc
    return GeoJSONGeometry.from_geojson(mapping(hull))

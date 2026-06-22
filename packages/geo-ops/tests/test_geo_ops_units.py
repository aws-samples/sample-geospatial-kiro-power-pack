"""Focused unit tests for ``geo-ops`` (task 4.6; Requirements 8.3, 8.10).

This file complements the example tests (``test_geometry_ops.py``) and the
catalog/registration tests (``test_geo_ops_native_ops.py``) by filling three
coverage gaps without duplicating what those files assert:

* **Invalid-geometry reasons (Req 8.3)** - ``validate_geometry`` must report a
  *non-empty* reason for any invalid geometry, including the defensive fallback
  where GEOS reports invalidity but supplies no explanatory text.
* **Malformed-input validation errors (Req 8.3 surface / 8.10 / 15.5)** - the
  ``geo-ops`` tools (``transform_crs``, ``spatial_join``, ``overlay``) must
  reject structurally malformed input with a taxonomy ``ValidationError`` and
  produce no partial output.
* **Native generic-GIS ops (buffer / convex_hull)** - the operations ``geo-ops``
  implements natively on Shapely/GEOS must return a correct result on a
  happy-path call and reject malformed input (bad ``distance``/``resolution`` or
  unparseable geometry) with a taxonomy ``ValidationError`` raised before any
  geometry work.
"""

from __future__ import annotations

import pytest

from geo_common.errors import ErrorCategory, ValidationError

from geo_ops.geometry import (
    buffer,
    convex_hull,
    overlay,
    spatial_join,
    validate_geometry,
)
from geo_ops.models import Feature, FeatureCollection, GeoJSONGeometry


# --- helpers --------------------------------------------------------------

def _poly(coords) -> GeoJSONGeometry:
    return GeoJSONGeometry(type="Polygon", coordinates=coords)


def _fc(*features: Feature) -> FeatureCollection:
    return FeatureCollection(features=list(features))


UNIT_SQUARE = [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
# Two distinct self-intersecting polygons GEOS reports as invalid.
BOWTIE = [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]
FIGURE_EIGHT = [[[0, 0], [2, 2], [0, 2], [2, 0], [0, 0]]]


# =========================================================================
# Invalid-geometry reasons (Requirement 8.3)
# =========================================================================

@pytest.mark.parametrize("ring", [BOWTIE, FIGURE_EIGHT])
def test_invalid_polygons_report_nonempty_reason(ring):
    """Req 8.3: each invalid geometry is reported invalid with a real reason."""
    result = validate_geometry(_poly(ring))
    assert result.valid is False
    assert result.reason is not None and result.reason.strip() != ""
    # GEOS attributes these to a self-intersection.
    assert "self-intersection" in result.reason.lower()


def test_validity_contract_reason_present_exactly_when_invalid():
    """Req 8.3: reason is non-empty iff invalid (None when valid)."""
    valid = validate_geometry(_poly(UNIT_SQUARE))
    assert valid.valid is True and valid.reason is None

    invalid = validate_geometry(_poly(BOWTIE))
    assert invalid.valid is False and bool(invalid.reason)


def test_invalid_geometry_without_geos_text_still_gets_reason(monkeypatch):
    """Req 8.3 (defensive): an invalid geometry never lacks a reason.

    When GEOS reports the geometry invalid but ``explain_validity`` yields no
    usable text, ``validate_geometry`` must still surface a non-empty reason
    rather than ``None``/empty.
    """
    import geo_ops.geometry as geometry_mod

    monkeypatch.setattr(geometry_mod, "explain_validity", lambda shp: "", raising=False)
    # Also patch the symbol imported lazily inside the function.
    import shapely.validation as shp_validation

    monkeypatch.setattr(shp_validation, "explain_validity", lambda shp: "")

    result = validate_geometry(_poly(BOWTIE))
    assert result.valid is False
    assert result.reason is not None and result.reason.strip() != ""


# =========================================================================
# Malformed-input validation errors (Req 8.3 surface / 8.10 / 15.5)
# =========================================================================

def test_transform_crs_unknown_crs_is_validation_error():
    """An unparseable CRS surfaces as a taxonomy validation error, no output."""
    from geo_ops.transform import transform_crs

    with pytest.raises(ValidationError) as exc:
        transform_crs(_poly(UNIT_SQUARE), src_crs="EPSG:4326", dst_crs="NOT-A-CRS")
    assert exc.value.category is ErrorCategory.VALIDATION
    # Detail names the offending CRS pair without leaking anything else.
    assert exc.value.detail == {"src_crs": "EPSG:4326", "dst_crs": "NOT-A-CRS"}


def test_transform_crs_malformed_coordinates_is_validation_error():
    """A non-position coordinate structure is rejected as validation."""
    from geo_ops.transform import transform_crs

    bad = GeoJSONGeometry(type="LineString", coordinates=["x", "y"])
    with pytest.raises(ValidationError) as exc:
        transform_crs(bad, src_crs="EPSG:4326", dst_crs="EPSG:3857")
    assert exc.value.category is ErrorCategory.VALIDATION


def test_transform_crs_missing_coordinates_is_validation_error():
    """A primitive geometry with no coordinates cannot be transformed."""
    from geo_ops.transform import transform_crs

    bad = GeoJSONGeometry(type="Point", coordinates=None)
    with pytest.raises(ValidationError) as exc:
        transform_crs(bad, src_crs="EPSG:4326", dst_crs="EPSG:3857")
    assert exc.value.category is ErrorCategory.VALIDATION


def test_malformed_geometry_validation_error_retains_original_detail():
    """A geometry GEOS cannot parse maps to validation, retaining original detail."""
    with pytest.raises(ValidationError) as exc:
        validate_geometry(GeoJSONGeometry(type="Polygon", coordinates="not-coords"))
    err = exc.value
    assert err.category is ErrorCategory.VALIDATION
    assert err.source == "geo-ops"
    assert err.original is not None and err.original != ""
    assert err.detail == {"geometry_type": "Polygon"}


def test_spatial_join_malformed_feature_geometry_is_validation_error():
    """A malformed feature geometry in a join input is a validation error."""
    good = _fc(Feature(geometry=_poly(UNIT_SQUARE), properties={}))
    bad = _fc(Feature(geometry=_poly("garbage"), properties={}))
    with pytest.raises(ValidationError) as exc:
        spatial_join(good, bad, predicate="intersects")
    assert exc.value.category is ErrorCategory.VALIDATION
    assert exc.value.source == "geo-ops"


def test_overlay_malformed_feature_geometry_is_validation_error():
    """A malformed feature geometry in an overlay input is a validation error."""
    good = _fc(Feature(geometry=_poly(UNIT_SQUARE), properties={}))
    bad = _fc(Feature(geometry=_poly("garbage"), properties={}))
    with pytest.raises(ValidationError) as exc:
        overlay(good, bad, op="intersection")
    assert exc.value.category is ErrorCategory.VALIDATION
    assert exc.value.source == "geo-ops"


# =========================================================================
# Native generic-GIS ops: buffer / convex_hull (Shapely/GEOS)
# =========================================================================

def test_buffer_happy_path_grows_geometry():
    """A positive buffer of a point yields a polygon containing the origin."""
    point = GeoJSONGeometry(type="Point", coordinates=[0, 0])
    result = buffer(point, distance=1.0)
    assert result.type == "Polygon"

    # The buffered polygon should cover the original point.
    from shapely.geometry import shape, Point

    assert shape(result.to_geojson()).contains(Point(0, 0))


def test_buffer_negative_distance_erodes_to_empty():
    """A negative buffer larger than the geometry erodes it (no exception)."""
    result = buffer(_poly(UNIT_SQUARE), distance=-5.0)
    # Eroding a 1x1 square by 5 units yields an empty geometry, surfaced as a
    # valid (empty) GeoJSON geometry rather than an error.
    from shapely.geometry import shape

    assert shape(result.to_geojson()).is_empty


@pytest.mark.parametrize("bad_distance", ["1.0", None, True, float("nan"), float("inf")])
def test_buffer_invalid_distance_is_validation_error(bad_distance):
    """A non-numeric/non-finite distance is rejected before any geometry work."""
    point = GeoJSONGeometry(type="Point", coordinates=[0, 0])
    with pytest.raises(ValidationError) as exc:
        buffer(point, distance=bad_distance)
    assert exc.value.category is ErrorCategory.VALIDATION
    assert exc.value.detail == {"parameter": "distance"}


@pytest.mark.parametrize("bad_resolution", [0, -1, 2.5, True, "16"])
def test_buffer_invalid_resolution_is_validation_error(bad_resolution):
    """A non-positive-integer resolution is rejected as validation."""
    point = GeoJSONGeometry(type="Point", coordinates=[0, 0])
    with pytest.raises(ValidationError) as exc:
        buffer(point, distance=1.0, resolution=bad_resolution)
    assert exc.value.category is ErrorCategory.VALIDATION
    assert exc.value.detail == {"parameter": "resolution"}


def test_buffer_malformed_geometry_is_validation_error():
    """Coordinates Shapely cannot parse surface as a validation error."""
    with pytest.raises(ValidationError) as exc:
        buffer(_poly("garbage"), distance=1.0)
    assert exc.value.category is ErrorCategory.VALIDATION
    assert exc.value.source == "geo-ops"


def test_convex_hull_happy_path_of_point_cloud():
    """The convex hull of four corner points is the enclosing square polygon."""
    points = GeoJSONGeometry(
        type="MultiPoint", coordinates=[[0, 0], [1, 0], [1, 1], [0, 1]]
    )
    result = convex_hull(points)
    assert result.type == "Polygon"

    from shapely.geometry import shape

    hull = shape(result.to_geojson())
    assert hull.equals(shape(_poly(UNIT_SQUARE).to_geojson()))


def test_convex_hull_malformed_geometry_is_validation_error():
    """Coordinates Shapely cannot parse surface as a validation error."""
    with pytest.raises(ValidationError) as exc:
        convex_hull(_poly("garbage"))
    assert exc.value.category is ErrorCategory.VALIDATION
    assert exc.value.source == "geo-ops"

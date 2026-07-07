"""Unit tests for the ``geo-ops`` geometry operations (Req 8.2, 8.3).

Example-based coverage of the task 4.3 tools:

* ``validate_geometry`` reports validity for valid geometries (no reason) and a
  non-empty reason for invalid ones (Req 8.2, 8.3);
* malformed geometry input raises a taxonomy ``ValidationError``;
* ``spatial_join`` attaches matched attributes by predicate and rejects unknown
  predicates;
* ``overlay`` computes set operations and rejects unknown ops.
"""

from __future__ import annotations

import pytest

from geo_common.errors import ErrorCategory, ValidationError

from geo_ops.geometry import overlay, spatial_join, validate_geometry
from geo_ops.models import Feature, FeatureCollection, GeoJSONGeometry


# --- helpers --------------------------------------------------------------

def _poly(coords) -> GeoJSONGeometry:
    return GeoJSONGeometry(type="Polygon", coordinates=coords)


UNIT_SQUARE = [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
# A "bowtie" polygon that self-intersects -> invalid geometry.
BOWTIE = [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]


# --- validate_geometry: Requirements 8.2 / 8.3 ---------------------------

def test_valid_geometry_reports_valid_without_reason():
    result = validate_geometry(_poly(UNIT_SQUARE))
    assert result.valid is True
    assert result.reason is None


def test_valid_point_is_valid():
    result = validate_geometry(GeoJSONGeometry(type="Point", coordinates=[1.0, 2.0]))
    assert result.valid is True
    assert result.reason is None


def test_invalid_geometry_reports_nonempty_reason():
    result = validate_geometry(_poly(BOWTIE))
    assert result.valid is False
    assert result.reason is not None
    assert result.reason.strip() != ""  # non-empty reason exactly when invalid


def test_malformed_geometry_raises_validation_error():
    with pytest.raises(ValidationError) as exc:
        validate_geometry(GeoJSONGeometry(type="Polygon", coordinates="not-coords"))
    assert exc.value.category is ErrorCategory.VALIDATION


# --- spatial_join ---------------------------------------------------------

def _fc(*features: Feature) -> FeatureCollection:
    return FeatureCollection(features=list(features))


def test_spatial_join_attaches_matching_attributes():
    left = _fc(Feature(geometry=_poly(UNIT_SQUARE), properties={"lid": "L1"}))
    # Overlapping square -> intersects.
    right = _fc(
        Feature(
            geometry=_poly([[[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]]]),
            properties={"rid": "R1"},
        )
    )
    result = spatial_join(left, right, predicate="intersects")
    assert len(result.features) == 1
    props = result.features[0].properties
    assert props.get("lid") == "L1"
    assert props.get("rid") == "R1"


def test_spatial_join_no_match_drops_left_feature():
    left = _fc(Feature(geometry=_poly(UNIT_SQUARE), properties={"lid": "L1"}))
    right = _fc(
        Feature(
            geometry=_poly([[[10, 10], [11, 10], [11, 11], [10, 11], [10, 10]]]),
            properties={"rid": "R1"},
        )
    )
    result = spatial_join(left, right, predicate="intersects")
    assert result.features == []


def test_spatial_join_rejects_unknown_predicate():
    fc = _fc(Feature(geometry=_poly(UNIT_SQUARE), properties={}))
    with pytest.raises(ValidationError) as exc:
        spatial_join(fc, fc, predicate="nope")
    assert exc.value.category is ErrorCategory.VALIDATION


# --- overlay --------------------------------------------------------------

def test_overlay_intersection_produces_overlap_region():
    a = _fc(Feature(geometry=_poly(UNIT_SQUARE), properties={"a": 1}))
    b = _fc(
        Feature(
            geometry=_poly([[[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]]]),
            properties={"b": 2},
        )
    )
    result = overlay(a, b, op="intersection")
    assert len(result.features) == 1
    assert result.features[0].geometry is not None
    assert result.features[0].geometry.type in ("Polygon", "MultiPolygon")


def test_overlay_difference_removes_overlap():
    a = _fc(Feature(geometry=_poly(UNIT_SQUARE), properties={"a": 1}))
    b = _fc(Feature(geometry=_poly(UNIT_SQUARE), properties={"b": 2}))
    result = overlay(a, b, op="difference")
    # Subtracting an identical polygon leaves no area.
    assert result.features == []


def test_overlay_rejects_unknown_op():
    fc = _fc(Feature(geometry=_poly(UNIT_SQUARE), properties={}))
    with pytest.raises(ValidationError) as exc:
        overlay(fc, fc, op="nope")
    assert exc.value.category is ErrorCategory.VALIDATION


# --- server wiring --------------------------------------------------------

@pytest.mark.asyncio
async def test_server_registers_and_runs_tools():
    from geo_ops.server import GeoOpsServer

    server = GeoOpsServer()
    assert set(server.tool_names()) == {
        "transform_crs",
        "validate_geometry",
        "spatial_join",
        "overlay",
        "buffer",
        "convex_hull",
        "simplify",
    }
    validity = await server.validate_geometry(geometry=_poly(BOWTIE))
    assert validity.valid is False
    assert validity.reason

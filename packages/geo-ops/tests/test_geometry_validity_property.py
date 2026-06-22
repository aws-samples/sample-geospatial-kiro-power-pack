"""Property test for geometry validity reporting (design Property 16).

Feature: geospatial-power-pack, Property 16: Geometry validity reporting includes a reason when invalid

Validates: Requirements 8.2, 8.3

*For any* geometry submitted for validation, the server reports whether it is
valid (Requirement 8.2), and a **non-empty** reason is present *exactly when*
the geometry is invalid (Requirement 8.3; design Property 16). Equivalently, the
biconditional ``(reason is non-empty) <=> (not valid)`` holds for every
geometry.

Input space
-----------
``validate_geometry`` reports on geometries Shapely/GEOS can *interpret*;
structurally unparseable input is a separate ``ValidationError`` path (covered
by the example tests) and is out of scope for "a geometry whose validity can be
reported". The generators below therefore produce a wide variety of
*well-formed-but-topologically-mixed* geometries so both branches of the
biconditional are exercised:

* points / multipoints / linestrings - structurally always valid;
* closed polygon rings built from random vertices - a natural mix of valid
  rings and self-intersecting (invalid) ones;
* explicit "bowtie" self-intersecting polygons - guaranteed-invalid cases;
* axis-aligned rectangles from a bounding box - guaranteed-valid polygons;
* geometry collections composed of the above.
"""

from __future__ import annotations

from typing import List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geo_ops.geometry import validate_geometry
from geo_ops.models import GeoJSONGeometry

# Coordinates kept in a bounded, modest-precision range so every generated
# structure is interpretable by GEOS (the point is to vary *topology*, not to
# probe coordinate parsing, which is a separate ValidationError path).
_coords = st.floats(
    min_value=-1000.0,
    max_value=1000.0,
    allow_nan=False,
    allow_infinity=False,
).map(lambda v: round(v, 3))
_position = st.builds(lambda x, y: [x, y], _coords, _coords)


def _point() -> st.SearchStrategy[GeoJSONGeometry]:
    return _position.map(lambda p: GeoJSONGeometry(type="Point", coordinates=p))


def _multipoint() -> st.SearchStrategy[GeoJSONGeometry]:
    return st.lists(_position, min_size=1, max_size=6).map(
        lambda ps: GeoJSONGeometry(type="MultiPoint", coordinates=ps)
    )


def _linestring() -> st.SearchStrategy[GeoJSONGeometry]:
    return st.lists(_position, min_size=2, max_size=6).map(
        lambda ps: GeoJSONGeometry(type="LineString", coordinates=ps)
    )


def _random_polygon() -> st.SearchStrategy[GeoJSONGeometry]:
    # A closed ring from random vertices: some rings are simple (valid), others
    # self-intersect (invalid) - giving a natural mix across both branches.
    return st.lists(_position, min_size=3, max_size=6).map(
        lambda ps: GeoJSONGeometry(type="Polygon", coordinates=[[*ps, ps[0]]])
    )


def _bowtie_polygon() -> st.SearchStrategy[GeoJSONGeometry]:
    # A figure-eight ring that always self-intersects -> guaranteed invalid,
    # parameterized so many distinct invalid instances are generated.
    def build(x0, y0, w, h):
        return GeoJSONGeometry(
            type="Polygon",
            coordinates=[[
                [x0, y0],
                [x0 + w, y0 + h],
                [x0 + w, y0],
                [x0, y0 + h],
                [x0, y0],
            ]],
        )

    pos = st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False)
    size = st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False)
    return st.builds(build, pos, pos, size, size)


def _rectangle_polygon() -> st.SearchStrategy[GeoJSONGeometry]:
    # An axis-aligned rectangle (CCW) from a bounding box -> guaranteed valid.
    def build(x0, y0, w, h):
        return GeoJSONGeometry(
            type="Polygon",
            coordinates=[[
                [x0, y0],
                [x0 + w, y0],
                [x0 + w, y0 + h],
                [x0, y0 + h],
                [x0, y0],
            ]],
        )

    pos = st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False)
    size = st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False)
    return st.builds(build, pos, pos, size, size)


_single_geometries = st.one_of(
    _point(),
    _multipoint(),
    _linestring(),
    _random_polygon(),
    _bowtie_polygon(),
    _rectangle_polygon(),
)


def _geometry() -> st.SearchStrategy[GeoJSONGeometry]:
    collection = st.lists(_single_geometries, min_size=1, max_size=4).map(
        lambda gs: GeoJSONGeometry(type="GeometryCollection", geometries=gs)
    )
    return st.one_of(_single_geometries, collection)


def _reason_is_nonempty(reason) -> bool:
    return isinstance(reason, str) and reason.strip() != ""


@pytest.mark.property
@settings(deadline=None)  # max_examples (>=100) inherited from the loaded profile
@given(geometry=_geometry())
def test_validity_reason_present_exactly_when_invalid(geometry: GeoJSONGeometry) -> None:
    """Feature: geospatial-power-pack, Property 16: Geometry validity reporting includes a reason when invalid.

    Validates: Requirements 8.2, 8.3
    """
    result = validate_geometry(geometry)

    # Req 8.2: validity is always reported as a boolean for any geometry.
    assert isinstance(result.valid, bool)

    # Req 8.3 / Property 16: a non-empty reason is present *exactly when*
    # the geometry is invalid (biconditional).
    has_reason = _reason_is_nonempty(result.reason)
    assert has_reason == (not result.valid), (
        f"reason/validity mismatch: valid={result.valid!r}, reason={result.reason!r}"
    )


@pytest.mark.property
@settings(deadline=None)
@given(geometry=_bowtie_polygon())
def test_self_intersecting_polygons_always_report_a_reason(
    geometry: GeoJSONGeometry,
) -> None:
    """Self-intersecting polygons are invalid and must carry a non-empty reason.

    Feature: geospatial-power-pack, Property 16: Geometry validity reporting includes a reason when invalid

    Validates: Requirements 8.2, 8.3
    """
    result = validate_geometry(geometry)
    assert result.valid is False
    assert _reason_is_nonempty(result.reason)

"""Property test for CRS round-trip equivalence (design Property 1).

Feature: geospatial-power-pack, Property 1: CRS round-trip equivalence within documented tolerance

Validates: Requirements 8.1, 15.2

*For any* valid geometry and any pair of coordinate reference systems,
transforming the geometry to the target CRS and back to the source CRS produces
a geometry whose maximum per-vertex coordinate deviation from the original does
not exceed the documented tolerance (expressed in source-CRS units).

Documented tolerance
--------------------
The round trip is not bit-exact: each forward/inverse projection introduces
floating-point (and, for ellipsoidal projections, iterative-inverse) error. The
documented per-vertex ceiling is :data:`geo_ops.transform.CRS_ROUNDTRIP_TOLERANCE`
(``1e-6`` source-CRS units), asserted below in the *source* CRS's own units.

Input space
-----------
The source CRS is geographic WGS84 (``EPSG:4326``); the target CRS is sampled
from a representative set of global, near-global, and well-behaved projected
CRSs. Generated coordinates are constrained to longitude in [-180, 180] and
latitude in [-84, 84] - i.e. each projection's area of validity, avoiding the
polar singularities of the Mercator family - so the round trip is well defined
for every generated case (per Requirement 15.2's "documented tolerance" over
the supported domain). Geometry topology is irrelevant to a per-vertex
coordinate round trip, so points, multipoints, linestrings, polygons, and
geometry collections are all exercised.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geo_ops.models import GeoJSONGeometry
from geo_ops.transform import CRS_ROUNDTRIP_TOLERANCE, transform_crs

# Source CRS: geographic WGS84. Deviations are therefore measured in degrees.
SOURCE_CRS = "EPSG:4326"

# Target CRSs spanning global cylindrical (3857/3395/4087) projections that are
# well-behaved across the generated lon/lat domain, plus a geographic->geographic
# identity-class pair (4326) as a degenerate baseline.
TARGET_CRSS = [
    "EPSG:3857",  # Web Mercator (spherical)
    "EPSG:3395",  # World Mercator (ellipsoidal, iterative inverse)
    "EPSG:4087",  # WGS84 World Equidistant Cylindrical
    "EPSG:4326",  # geographic -> geographic (round-trip baseline)
]

# Coordinate domain: full longitude range, latitude clamped inside the Mercator
# family's validity (|lat| <= ~85) with margin so inverse projection is stable.
_lons = st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)
_lats = st.floats(min_value=-84.0, max_value=84.0, allow_nan=False, allow_infinity=False)
_positions = st.builds(lambda lon, lat: [lon, lat], _lons, _lats)


def _point() -> st.SearchStrategy[GeoJSONGeometry]:
    return _positions.map(lambda p: GeoJSONGeometry(type="Point", coordinates=p))


def _multipoint() -> st.SearchStrategy[GeoJSONGeometry]:
    return st.lists(_positions, min_size=1, max_size=8).map(
        lambda ps: GeoJSONGeometry(type="MultiPoint", coordinates=ps)
    )


def _linestring() -> st.SearchStrategy[GeoJSONGeometry]:
    return st.lists(_positions, min_size=2, max_size=8).map(
        lambda ps: GeoJSONGeometry(type="LineString", coordinates=ps)
    )


def _polygon() -> st.SearchStrategy[GeoJSONGeometry]:
    # A closed linear ring (first == last). Topological validity is not required
    # for a coordinate round-trip; we only need a well-formed ring structure.
    return st.lists(_positions, min_size=3, max_size=8).map(
        lambda ps: GeoJSONGeometry(type="Polygon", coordinates=[[*ps, ps[0]]])
    )


_single_geometries = st.one_of(_point(), _multipoint(), _linestring(), _polygon())


def _geometry() -> st.SearchStrategy[GeoJSONGeometry]:
    collection = st.lists(_single_geometries, min_size=1, max_size=4).map(
        lambda gs: GeoJSONGeometry(type="GeometryCollection", geometries=gs)
    )
    return st.one_of(_single_geometries, collection)


def _vertices(geometry: GeoJSONGeometry) -> List[Tuple[float, float]]:
    """Flatten a geometry into its ordered list of (x, y) vertices."""
    if geometry.type == "GeometryCollection":
        out: List[Tuple[float, float]] = []
        for member in geometry.geometries or []:
            out.extend(_vertices(member))
        return out
    return _flatten(geometry.coordinates)


def _flatten(coords) -> List[Tuple[float, float]]:
    # A position is a list whose first element is a (non-bool) number.
    if (
        isinstance(coords, (list, tuple))
        and len(coords) >= 2
        and isinstance(coords[0], (int, float))
        and not isinstance(coords[0], bool)
    ):
        return [(float(coords[0]), float(coords[1]))]
    out: List[Tuple[float, float]] = []
    for child in coords:
        out.extend(_flatten(child))
    return out


@pytest.mark.property
@settings(deadline=None)  # max_examples (>=100) inherited from the loaded profile
@given(geometry=_geometry(), dst_crs=st.sampled_from(TARGET_CRSS))
def test_crs_round_trip_within_documented_tolerance(
    geometry: GeoJSONGeometry, dst_crs: str
) -> None:
    """Feature: geospatial-power-pack, Property 1: CRS round-trip equivalence within documented tolerance.

    Validates: Requirements 8.1, 15.2
    """
    # Forward to the target CRS, then back to the source CRS.
    forward = transform_crs(geometry, src_crs=SOURCE_CRS, dst_crs=dst_crs)
    back = transform_crs(forward, src_crs=dst_crs, dst_crs=SOURCE_CRS)

    original = _vertices(geometry)
    roundtripped = _vertices(back)

    # The transform preserves structure: same vertex count and ordering.
    assert len(roundtripped) == len(original)

    # Maximum per-vertex coordinate deviation (in source-CRS units = degrees)
    # must not exceed the documented tolerance.
    for (ox, oy), (rx, ry) in zip(original, roundtripped):
        assert abs(rx - ox) <= CRS_ROUNDTRIP_TOLERANCE
        assert abs(ry - oy) <= CRS_ROUNDTRIP_TOLERANCE

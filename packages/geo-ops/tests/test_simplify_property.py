"""Property test for ``geo-ops.simplify`` (Douglas-Peucker contract).

Feature: geospatial-power-pack, geometry simplification (vector-complexity reduction)

Validates: the shape-preserving vertex-reduction guarantees that make
``simplify`` the recommended way to shrink a high-vertex zone perimeter before an
inline geometry op (zonal stats, CRS transform, spatial join) without discarding
concavity the way ``convex_hull`` does.

*For any* valid polygon or linestring and *any* finite, non-negative tolerance,
Douglas-Peucker simplification must satisfy the contract the tool advertises:

1. **Vertex count is non-increasing** — simplification only ever *removes*
   vertices, never adds them.
2. **Deviation is bounded by the tolerance** — every point of the result stays
   within ``tolerance`` of the input (and vice versa), i.e. the Hausdorff
   distance between input and output is ``<= tolerance`` (within float noise).
   This is the "shape-preserving" guarantee.
3. **Topology is preserved when requested** — with ``preserve_topology=True`` a
   non-empty result is a valid geometry (never self-intersecting/collapsed).

Input space
-----------
Generators produce *valid* geometries so both the Hausdorff and validity
assertions are meaningful (malformed input is a separate ``ValidationError``
path, covered by the unit tests):

* convex polygons — the convex hull of a random point cloud is always a valid,
  non-self-intersecting polygon, with edges that carry removable vertices;
* linestrings from random vertices.

Tolerances span ``0`` (a no-op) up through values large enough to remove real
vertices, kept well below the coordinate span so a polygon does not collapse to
empty under pure Douglas-Peucker.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geo_ops.geometry import simplify
from geo_ops.models import GeoJSONGeometry

# Coordinates in a wide, modest-precision range; tolerances stay small relative
# to this span so simplification removes vertices without collapsing shapes.
_coord = st.floats(
    min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False
).map(lambda v: round(v, 3))
_position = st.builds(lambda x, y: [x, y], _coord, _coord)
_tolerance = st.floats(
    min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
).map(lambda v: round(v, 4))


def _convex_polygon() -> "st.SearchStrategy[GeoJSONGeometry]":
    """A valid convex polygon: the convex hull of a random point cloud."""

    def build(points):
        from shapely.geometry import MultiPoint, mapping

        hull = MultiPoint(points).convex_hull
        return hull, GeoJSONGeometry.from_geojson(mapping(hull))

    return (
        st.lists(_position, min_size=3, max_size=12)
        .map(build)
        # Keep only clouds whose hull is a 2-D polygon (collinear clouds degrade
        # to a line/point, which is a different — still valid — assertion path).
        .filter(lambda pair: pair[0].geom_type == "Polygon" and not pair[0].is_empty)
        .map(lambda pair: pair[1])
    )


def _linestring() -> "st.SearchStrategy[GeoJSONGeometry]":
    return st.lists(_position, min_size=2, max_size=12).map(
        lambda ps: GeoJSONGeometry(type="LineString", coordinates=ps)
    )


_geometry = st.one_of(_convex_polygon(), _linestring())


def _vertex_count(shp) -> int:
    if shp.is_empty:
        return 0
    if shp.geom_type == "Polygon":
        return len(shp.exterior.coords)
    if shp.geom_type == "LineString":
        return len(shp.coords)
    return len(list(getattr(shp, "coords", [])))


@pytest.mark.property
@settings(deadline=None)  # max_examples inherited from the loaded profile
@given(geometry=_geometry, tolerance=_tolerance, preserve_topology=st.booleans())
def test_simplify_contract(
    geometry: GeoJSONGeometry, tolerance: float, preserve_topology: bool
) -> None:
    """Feature: geospatial-power-pack, geometry simplification.

    Vertex count is non-increasing, deviation is bounded by the tolerance, and
    topology is preserved when requested.
    """
    from shapely.geometry import shape

    original = shape(geometry.to_geojson())
    result = simplify(
        geometry, tolerance=tolerance, preserve_topology=preserve_topology
    )
    simplified = shape(result.to_geojson())

    # (1) Vertex count never grows.
    assert _vertex_count(simplified) <= _vertex_count(original)

    if simplified.is_empty:
        # A pure Douglas-Peucker pass can erase a thin polygon at a large
        # tolerance; topology-preserving mode never does.
        assert preserve_topology is False
        return

    # (3) preserve_topology=True preserves validity: a valid input stays valid
    # (simplification is not expected to *repair* an already-degenerate input,
    # e.g. a zero-length linestring the generator may produce).
    if preserve_topology and original.is_valid:
        assert simplified.is_valid

    # (2) The result stays within `tolerance` of the input (shape-preserving).
    # A tiny epsilon absorbs float rounding at the coordinate precision used.
    eps = 1e-6
    assert original.hausdorff_distance(simplified) <= tolerance + eps


@pytest.mark.property
@settings(deadline=None)
@given(geometry=_geometry, preserve_topology=st.booleans())
def test_simplify_zero_tolerance_is_shape_preserving(
    geometry: GeoJSONGeometry, preserve_topology: bool
) -> None:
    """tolerance=0 must not change the footprint (no vertices meaningfully move).

    Feature: geospatial-power-pack, geometry simplification.
    """
    from shapely.geometry import shape

    original = shape(geometry.to_geojson())
    result = simplify(geometry, tolerance=0.0, preserve_topology=preserve_topology)
    simplified = shape(result.to_geojson())

    assert simplified.equals(original)

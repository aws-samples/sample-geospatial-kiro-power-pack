"""Unit tests for the terrain derivatives (slope, hillshade) - pure functions.

These exercise :mod:`geo_terrain.derivatives` directly with known grids, so they
are deterministic and need no network. Server-level integration (fetch grid ->
compute) is covered in ``test_elevation.py``.
"""

from __future__ import annotations

import math

import pytest

from geo_terrain.derivatives import (
    FLAT_ASPECT,
    compute_aspect,
    compute_hillshade,
    compute_slope,
    meters_per_degree,
)


def test_meters_per_degree_shrinks_with_latitude():
    lon0, lat0 = meters_per_degree(0.0)
    lon60, lat60 = meters_per_degree(60.0)
    assert lon0 == pytest.approx(111_320.0, rel=1e-6)
    assert lat0 == pytest.approx(110_540.0, rel=1e-6)
    # cos(60deg) = 0.5, so longitude metres roughly halve.
    assert lon60 == pytest.approx(lon0 * 0.5, rel=1e-3)
    assert lat60 == lat0  # latitude metres treated as constant


def test_slope_of_flat_grid_is_zero():
    grid = [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
    slope = compute_slope(grid, cellsize_x_m=30.0, cellsize_y_m=30.0)
    assert all(abs(v) < 1e-9 for row in slope for v in row)


def test_slope_of_constant_east_ramp_is_constant_angle():
    # z increases by 10 m per column; cells are 10 m wide -> dz/dx = 1 -> 45deg.
    grid = [[0.0, 10.0, 20.0], [0.0, 10.0, 20.0], [0.0, 10.0, 20.0]]
    slope = compute_slope(grid, cellsize_x_m=10.0, cellsize_y_m=10.0)
    for row in slope:
        for v in row:
            assert v == pytest.approx(45.0, abs=1e-6)


def test_slope_propagates_none_holes():
    grid = [[0.0, 1.0, 2.0], [0.0, None, 2.0], [0.0, 1.0, 2.0]]
    slope = compute_slope(grid, cellsize_x_m=10.0, cellsize_y_m=10.0)
    # The no-data cell (and its gradient neighbours) come back as None.
    assert any(v is None for row in slope for v in row)


def test_aspect_of_flat_grid_is_flat_sentinel():
    grid = [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
    aspect = compute_aspect(grid, cellsize_x_m=30.0, cellsize_y_m=30.0)
    assert all(v == FLAT_ASPECT for row in aspect for v in row)


def test_aspect_faces_west_when_elevation_rises_east():
    # z increases toward the east -> downslope faces west -> aspect 270.
    grid = [[0.0, 10.0, 20.0], [0.0, 10.0, 20.0], [0.0, 10.0, 20.0]]
    aspect = compute_aspect(grid, cellsize_x_m=10.0, cellsize_y_m=10.0)
    for row in aspect:
        for v in row:
            assert v == pytest.approx(270.0)


def test_aspect_faces_south_when_elevation_rises_north():
    # Row 0 is north; z decreases southward -> downslope faces south -> aspect 180.
    grid = [[20.0, 20.0, 20.0], [10.0, 10.0, 10.0], [0.0, 0.0, 0.0]]
    aspect = compute_aspect(grid, cellsize_x_m=10.0, cellsize_y_m=10.0)
    for row in aspect:
        for v in row:
            assert v == pytest.approx(180.0)


def test_aspect_propagates_none_holes():
    grid = [[0.0, 1.0, 2.0], [0.0, None, 2.0], [0.0, 1.0, 2.0]]
    aspect = compute_aspect(grid, cellsize_x_m=10.0, cellsize_y_m=10.0)
    assert any(v is None for row in aspect for v in row)


def test_hillshade_of_flat_grid_is_uniform_and_in_range():
    grid = [[100.0] * 4 for _ in range(4)]
    hs = compute_hillshade(grid, cellsize_x_m=30.0, cellsize_y_m=30.0, altitude_deg=45.0)
    flat = {v for row in hs for v in row}
    assert len(flat) == 1  # uniform
    expected = round(255.0 * math.cos(math.radians(45.0)))  # zenith=45 on flat
    (value,) = flat
    assert value == pytest.approx(expected, abs=1)
    assert 0 <= value <= 255


def test_hillshade_values_are_bounded():
    grid = [[0.0, 50.0, 0.0], [50.0, 200.0, 50.0], [0.0, 50.0, 0.0]]
    hs = compute_hillshade(grid, cellsize_x_m=10.0, cellsize_y_m=10.0)
    for row in hs:
        for v in row:
            assert v is None or 0 <= v <= 255


from hypothesis import given, settings
from hypothesis import strategies as st


@pytest.mark.property
@settings(deadline=None)
@given(
    width=st.integers(min_value=2, max_value=6),
    height=st.integers(min_value=2, max_value=6),
    a=st.integers(min_value=-5, max_value=5),
    b=st.integers(min_value=-5, max_value=5),
    cx=st.sampled_from([1.0, 10.0, 30.0]),
    cy=st.sampled_from([1.0, 10.0, 30.0]),
)
def test_aspect_matches_analytic_bearing_on_a_plane(width, height, a, b, cx, cy) -> None:
    """Feature: geospatial-power-pack — aspect equals the analytic downslope bearing.

    On a planar surface ``z = a*col + b*row`` the ground gradients are exact
    constants, so the compass aspect has a closed form independent of the numpy
    ``np.gradient`` implementation: ``atan2(-a/cx, b/cy)`` (flat when a==b==0).
    """
    grid = [[float(a * col + b * row) for col in range(width)] for row in range(height)]
    aspect = compute_aspect(grid, cellsize_x_m=cx, cellsize_y_m=cy)

    if a == 0 and b == 0:
        expected = FLAT_ASPECT
    else:
        expected = math.degrees(math.atan2(-(a / cx), (b / cy))) % 360.0

    for row in aspect:
        for v in row:
            assert v is not None
            if expected == FLAT_ASPECT:
                assert v == FLAT_ASPECT
            else:
                assert v == pytest.approx(expected, abs=1e-6)

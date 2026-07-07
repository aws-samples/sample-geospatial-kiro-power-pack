"""Differential property test: numpy fast path == pure zonal engine.

Feature: geospatial-power-pack, Property: zonal backends agree

Validates: Requirements 8.7, 8.8

*For any* raster grid and set of vector zones, the optional vectorized numpy
engine (:func:`geo_raster.zonal_fast.compute_zonal_statistics_numpy`) returns a
result identical to the dependency-free pure engine
(:func:`geo_raster.zonal._compute_zonal_statistics_pure`) — same zones in order,
same no-data flags, and every statistic equal (to floating-point tolerance). This
guards against the two backends drifting, so ``backend="auto"`` is safe.

Grid/zone construction mirrors ``test_zonal_statistics_property``: small north-up
unit-cell grids and axis-aligned integer-corner rectangles, so cell centers (on
half-integers) never fall on a zone edge.
"""

from __future__ import annotations

from typing import List, Optional

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geo_common.errors import ValidationError
from geo_common import zonal as zonal_mod
from geo_common.raster_models import Feature, FeatureCollection, GeoJSONGeometry, RasterGrid
from geo_common.zonal import (
    _compute_zonal_statistics_pure,
    _validate_stats,
    compute_zonal_statistics,
)
from geo_common.zonal_fast import compute_zonal_statistics_numpy

_ALL_STATS = ["min", "max", "mean", "sum", "count", "std"]


@st.composite
def _geotransform(draw: st.DrawFn):
    """A GDAL 6-tuple: north-up, axis-flipped, or lightly rotated/sheared.

    Both backends forward-map cell centers through this affine and share the
    same ray-casting convention, so they must agree even on rotated grids
    (where cell centers no longer land on tidy half-integers).
    """
    origin_x = draw(st.sampled_from([0.0, -5.0, 100.0]))
    origin_y = draw(st.sampled_from([0.0, 5.0, 50.0]))
    px_w = draw(st.sampled_from([1.0, 0.5, 2.0]))
    px_h = draw(st.sampled_from([-1.0, -0.5, 1.0]))  # +ve => south-up
    rot = draw(st.sampled_from([0.0, 0.25, -0.3]))
    return (origin_x, px_w, rot, origin_y, rot, px_h)


@st.composite
def _grid_and_zones(draw: st.DrawFn):
    width = draw(st.integers(min_value=1, max_value=8))
    height = draw(st.integers(min_value=1, max_value=8))
    n = width * height
    # Mix integer- and fractional-valued grids so the numpy float64 cast and
    # the pure Python path are compared on both.
    element = draw(
        st.sampled_from(
            [
                st.integers(min_value=-10, max_value=10).map(float),
                st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
            ]
        )
    )
    values = draw(st.lists(element, min_size=n, max_size=n))
    nodata: Optional[float] = draw(st.one_of(st.none(), st.sampled_from([-999.0, 0.0, 5.0])))
    geotransform = draw(_geotransform())
    grid = RasterGrid(
        width=width,
        height=height,
        geotransform=geotransform,
        nodata=nodata,
        values=values,
    )

    # Sample zone corners within the grid's world extent (plus a margin) so
    # zones overlap the raster regardless of origin/scale/rotation, while some
    # still fall outside and exercise the no-data path.
    corners = [
        grid.cell_center(0, 0),
        grid.cell_center(width - 1, 0),
        grid.cell_center(0, height - 1),
        grid.cell_center(width - 1, height - 1),
    ]
    wxs = [c[0] for c in corners]
    wys = [c[1] for c in corners]
    lo_x, hi_x = min(wxs) - 2.0, max(wxs) + 2.0
    lo_y, hi_y = min(wys) - 2.0, max(wys) + 2.0
    fx = st.floats(min_value=lo_x, max_value=hi_x, allow_nan=False, allow_infinity=False)
    fy = st.floats(min_value=lo_y, max_value=hi_y, allow_nan=False, allow_infinity=False)

    zones: List[Feature] = []
    for i in range(draw(st.integers(min_value=0, max_value=5))):
        x0, x1, y0, y1 = draw(fx), draw(fx), draw(fy), draw(fy)
        if x0 == x1:
            x1 = x0 + 0.5
        if y0 == y1:
            y1 = y0 + 0.5
        min_x, max_x = sorted((float(x0), float(x1)))
        min_y, max_y = sorted((float(y0), float(y1)))
        ring = [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
            [min_x, min_y],
        ]
        zones.append(
            Feature(
                geometry=GeoJSONGeometry(type="Polygon", coordinates=[ring]),
                properties={"zone_id": f"z{i}"},
            )
        )

    stats = draw(st.lists(st.sampled_from(_ALL_STATS), min_size=1, max_size=6, unique=True))
    return grid, FeatureCollection(features=zones), stats


@pytest.mark.property
@settings(deadline=None)
@given(case=_grid_and_zones())
def test_numpy_backend_matches_pure(case) -> None:
    """Feature: geospatial-power-pack, Property: zonal backends agree.

    Validates: Requirements 8.7, 8.8
    """
    grid, zones, stats = case
    requested = _validate_stats(stats, source="test")

    pure = _compute_zonal_statistics_pure(grid, zones, requested, source="test")
    fast = compute_zonal_statistics_numpy(grid, zones, requested, source="test")

    assert len(pure) == len(fast)
    for p, f in zip(pure, fast):
        assert p.zone_id == f.zone_id
        assert p.no_data == f.no_data
        assert set(p.statistics) == set(f.statistics)
        for name in requested:
            if p.statistics[name] is None:
                assert f.statistics[name] is None
            else:
                assert f.statistics[name] == pytest.approx(p.statistics[name])


def test_multipolygon_and_holes_agree() -> None:
    """A concrete MultiPolygon-with-hole case matches between backends."""
    grid = RasterGrid(
        width=6,
        height=6,
        geotransform=(0.0, 1.0, 0.0, 6.0, 0.0, -1.0),
        nodata=None,
        values=[float(i) for i in range(36)],
    )
    # Outer square [0,5]x[1,6] with a hole [1,4]x[2,5], plus a detached square.
    geometry = GeoJSONGeometry(
        type="MultiPolygon",
        coordinates=[
            [
                [[0.0, 1.0], [5.0, 1.0], [5.0, 6.0], [0.0, 6.0], [0.0, 1.0]],
                [[1.0, 2.0], [4.0, 2.0], [4.0, 5.0], [1.0, 5.0], [1.0, 2.0]],
            ],
            [[[5.0, 0.0], [6.0, 0.0], [6.0, 1.0], [5.0, 1.0], [5.0, 0.0]]],
        ],
    )
    zones = FeatureCollection(features=[Feature(geometry=geometry, properties={"zone_id": "z"})])
    requested = _validate_stats(_ALL_STATS, source="test")

    pure = _compute_zonal_statistics_pure(grid, zones, requested, source="test")
    fast = compute_zonal_statistics_numpy(grid, zones, requested, source="test")
    assert pure[0].no_data is False
    for name in requested:
        assert fast[0].statistics[name] == pytest.approx(pure[0].statistics[name])


def _unit_grid(values, nodata=None):
    return RasterGrid(
        width=2,
        height=2,
        geotransform=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0),
        nodata=nodata,
        values=values,
    )


def _cover_zone():
    return FeatureCollection(
        features=[
            Feature(
                geometry=GeoJSONGeometry(
                    type="Polygon",
                    coordinates=[[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]]],
                ),
                properties={"zone_id": "all"},
            )
        ]
    )


def test_all_nodata_grid_is_no_data_in_both_backends() -> None:
    """A grid where every cell is nodata yields a no-data zone in both engines."""
    grid = _unit_grid([7.0, 7.0, 7.0, 7.0], nodata=7.0)
    requested = _validate_stats(_ALL_STATS, source="test")
    pure = _compute_zonal_statistics_pure(grid, _cover_zone(), requested, source="test")
    fast = compute_zonal_statistics_numpy(grid, _cover_zone(), requested, source="test")
    assert pure[0].no_data is True
    assert fast[0].no_data is True
    assert set(fast[0].statistics.values()) == {None}


def test_backend_dispatch_pure_and_numpy_agree() -> None:
    """The public dispatcher returns equal results for explicit pure/numpy backends."""
    grid = _unit_grid([1.0, 2.0, 3.0, 4.0])
    zones = _cover_zone()
    pure = compute_zonal_statistics(grid, zones, _ALL_STATS, backend="pure")
    fast = compute_zonal_statistics(grid, zones, _ALL_STATS, backend="numpy")
    for name in _ALL_STATS:
        assert fast[0].statistics[name] == pytest.approx(pure[0].statistics[name])


def test_unknown_backend_is_rejected() -> None:
    grid = _unit_grid([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValidationError):
        compute_zonal_statistics(grid, _cover_zone(), ["mean"], backend="sideways")


def test_numpy_backend_requires_numpy(monkeypatch) -> None:
    """With numpy unavailable, explicit numpy backend errors and auto falls back."""
    monkeypatch.setattr(zonal_mod, "_numpy_available", lambda: False)
    grid = _unit_grid([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValidationError):
        compute_zonal_statistics(grid, _cover_zone(), ["mean"], backend="numpy")
    # auto still succeeds via the pure fallback.
    result = compute_zonal_statistics(grid, _cover_zone(), ["mean"], backend="auto")
    assert result[0].statistics["mean"] == pytest.approx(2.5)

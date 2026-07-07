"""Property test for zonal statistics (design Property 14).

Feature: geospatial-power-pack, Property 14: Zonal statistics equal a reference computation

Validates: Requirements 8.7, 8.8

*For any* raster and any set of vector zones, each requested statistic
(minimum, maximum, mean, sum, count, population standard deviation) computed by
:func:`geo_raster.compute_zonal_statistics` equals the value computed by a
straightforward reference implementation over the raster cells whose center
overlaps the zone, and a zone with no overlapping cells yields a no-data
indication (every requested statistic ``None``) while the remaining zones still
receive statistics.

Input space
-----------
Each raster is a small (up to 6x6) north-up grid with unit cells anchored at the
origin ``(0, height)`` (geotransform ``(0, 1, 0, height, 0, -1)``), so cell
``(col, row)`` has its center at the half-integer coordinate
``(col + 0.5, height - row - 0.5)``. Cell values are drawn from a small integer
range (carried as ``float``) so an optional ``nodata`` sentinel can exactly
match — and therefore exclude — some cells (Requirement 8.7's "defined set" of
statistics is computed only over non-nodata cells).

Zones are axis-aligned rectangles with **integer** corner coordinates spanning a
range that extends a few cells beyond the grid in every direction. Because cell
centers fall on half-integers and zone edges on integers, no cell center ever
lies exactly on a zone boundary, so membership is unambiguous and the
reference's plain ``min < x < max`` rectangle test agrees with the server's
ray-casting point-in-polygon test on every case. Letting zones range outside the
grid makes some zones overlap no cells, exercising the no-data indication
(Requirement 8.8) alongside zones that do receive statistics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geo_raster import (
    Feature,
    FeatureCollection,
    GeoJSONGeometry,
    RasterGrid,
    compute_zonal_statistics,
)

STAT_NAMES: Tuple[str, ...] = ("min", "max", "mean", "sum", "count", "std")
Rect = Tuple[float, float, float, float]


def _rect_polygon(min_x: float, min_y: float, max_x: float, max_y: float) -> GeoJSONGeometry:
    """A closed, axis-aligned rectangular Polygon zone."""
    ring = [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
        [min_x, min_y],
    ]
    return GeoJSONGeometry(type="Polygon", coordinates=[ring])


def _zone_rect(feature: Feature) -> Rect:
    """Recover a rectangular zone's ``(min_x, min_y, max_x, max_y)`` bbox."""
    ring = feature.geometry.coordinates[0]  # type: ignore[union-attr]
    xs = [pt[0] for pt in ring]
    ys = [pt[1] for pt in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def _reference(grid: RasterGrid, rect: Rect, stats: List[str]) -> Dict[str, Optional[float]]:
    """A straightforward, independent reference computation.

    Collect the value of every cell whose center falls strictly inside the
    rectangle and is not the grid's ``nodata`` (or NaN), then reduce. An empty
    set of overlapping cells yields the no-data indication (Requirement 8.8).
    """
    min_x, min_y, max_x, max_y = rect
    values: List[float] = []
    for row in range(grid.height):
        for col in range(grid.width):
            value = grid.values[row * grid.width + col]
            if grid.nodata is not None and value == grid.nodata:
                continue
            if value != value:  # NaN
                continue
            x, y = grid.cell_center(col, row)
            if min_x < x < max_x and min_y < y < max_y:
                values.append(value)
    if not values:
        return {name: None for name in stats}
    mean = sum(values) / len(values)
    reduced = {
        "min": float(min(values)),
        "max": float(max(values)),
        "sum": float(sum(values)),
        "mean": mean,
        "count": float(len(values)),
        "std": (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5,
    }
    return {name: reduced[name] for name in stats}


@st.composite
def _grids(draw: st.DrawFn, *, allow_nodata: bool = True) -> RasterGrid:
    width = draw(st.integers(min_value=1, max_value=6))
    height = draw(st.integers(min_value=1, max_value=6))
    n = width * height
    values = draw(
        st.lists(
            st.integers(min_value=-10, max_value=10).map(float),
            min_size=n,
            max_size=n,
        )
    )
    nodata: Optional[float] = None
    if allow_nodata:
        # Sentinels in-range (-10/0/5) exclude some cells; -999 excludes none.
        nodata = draw(st.one_of(st.none(), st.sampled_from([-10.0, 0.0, 5.0, -999.0])))
    geotransform = (0.0, 1.0, 0.0, float(height), 0.0, -1.0)
    return RasterGrid(
        width=width,
        height=height,
        geotransform=geotransform,
        nodata=nodata,
        values=values,
    )


@st.composite
def _grid_and_zones(draw: st.DrawFn):
    grid = draw(_grids())
    w, h = grid.width, grid.height
    count = draw(st.integers(min_value=0, max_value=5))
    zones: List[Feature] = []
    coord = st.integers(min_value=-3, max_value=max(w, h) + 3)
    for i in range(count):
        x0 = draw(coord)
        x1 = draw(coord)
        y0 = draw(coord)
        y1 = draw(coord)
        if x0 == x1:
            x1 = x0 + 1
        if y0 == y1:
            y1 = y0 + 1
        min_x, max_x = sorted((float(x0), float(x1)))
        min_y, max_y = sorted((float(y0), float(y1)))
        zones.append(
            Feature(
                geometry=_rect_polygon(min_x, min_y, max_x, max_y),
                properties={"zone_id": f"z{i}"},
            )
        )
    stats = draw(st.lists(st.sampled_from(STAT_NAMES), min_size=1, max_size=5, unique=True))
    return grid, FeatureCollection(features=zones), stats


@pytest.mark.property
@settings(deadline=None)  # max_examples (>=100) inherited from the loaded profile
@given(case=_grid_and_zones())
def test_zonal_statistics_equal_reference(case) -> None:
    """Feature: geospatial-power-pack, Property 14: Zonal statistics equal a reference computation.

    Validates: Requirements 8.7, 8.8
    """
    grid, zones, stats = case
    result = compute_zonal_statistics(grid, zones, stats)

    # Every zone is returned, in order, identified by its zone_id.
    assert len(result) == len(zones.features)

    for feature, zone in zip(zones.features, result):
        assert zone.zone_id == feature.properties["zone_id"]
        assert set(zone.statistics) == set(stats)

        expected = _reference(grid, _zone_rect(feature), stats)

        if all(value is None for value in expected.values()):
            # A zone with no overlapping cells -> no-data indication (Req 8.8).
            assert zone.no_data is True
            assert set(zone.statistics.values()) == {None}
        else:
            # An overlapping zone receives statistics matching the reference
            # exactly for each requested statistic (Req 8.7).
            assert zone.no_data is False
            for name in stats:
                assert zone.statistics[name] == pytest.approx(expected[name])


@pytest.mark.property
@settings(deadline=None)
@given(grid=_grids(allow_nodata=False))
def test_empty_zone_is_no_data_while_others_receive_statistics(grid: RasterGrid) -> None:
    """Feature: geospatial-power-pack, Property 14: Zonal statistics equal a reference computation.

    Validates: Requirements 8.7, 8.8

    A zone far outside the raster overlaps no cells and must yield a no-data
    indication, while a zone covering the whole grid still receives statistics
    over every (non-nodata) cell.
    """
    w, h = grid.width, grid.height
    empty = Feature(
        geometry=_rect_polygon(w + 5.0, h + 5.0, w + 7.0, h + 7.0),
        properties={"zone_id": "empty"},
    )
    full = Feature(
        geometry=_rect_polygon(0.0, 0.0, float(w), float(h)),
        properties={"zone_id": "full"},
    )
    zones = FeatureCollection(features=[empty, full])
    stats = ["min", "max", "mean", "sum", "count"]

    result = compute_zonal_statistics(grid, zones, stats)
    assert len(result) == 2
    by_id = {zone.zone_id: zone for zone in result}

    # The empty zone is still returned, flagged no-data, every statistic None.
    assert by_id["empty"].no_data is True
    assert set(by_id["empty"].statistics.values()) == {None}

    # The full-cover zone receives statistics over all w*h cells (no nodata).
    full_zone = by_id["full"]
    assert full_zone.no_data is False
    assert full_zone.statistics["count"] == float(w * h)
    expected = _reference(grid, (0.0, 0.0, float(w), float(h)), stats)
    for name in stats:
        assert full_zone.statistics[name] == pytest.approx(expected[name])

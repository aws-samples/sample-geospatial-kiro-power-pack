"""Property test for zonal band math (per-pixel multi-asset index).

Feature: geospatial-power-pack, Property: zonal band math equals a reference

Validates: Requirements 7.12, 8.7, 8.8

*For any* pair of aligned single-band grids and any band-math expression from the
safe arithmetic subset, the per-pixel index grid produced by
:func:`geo_raster.zonal_band_math._evaluate_index` equals — cell for cell — an
independent reference computation, where a pixel is masked (``NaN``) exactly when
any contributing band is nodata/NaN or the expression hits a zero denominator.
Reducing that index over random vector zones with
:func:`geo_raster.compute_zonal_statistics` then equals a straightforward
reference reduction over the unmasked in-zone cells.

The grid/zone construction mirrors ``test_zonal_statistics_property``: small
north-up unit-cell grids anchored at ``(0, height)`` so cell ``(col, row)`` has
its center at ``(col + 0.5, height - row - 0.5)``, and axis-aligned integer-corner
rectangular zones, so no cell center ever lies on a zone boundary.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

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
from geo_raster.window_reader import _parse_expression
from geo_raster.zonal_band_math import _evaluate_index

# Each expression references bands B08 (NIR) and B04 (Red); the paired reference
# function computes the same value, returning None where the implementation
# masks the pixel (a zero denominator -> ZeroDivisionError -> NaN).
_NDVI = "(B08 - B04) / (B08 + B04)"
_DIFF = "B08 - B04"
_AVG = "(B08 + B04) / 2"

_EXPRESSIONS: Tuple[Tuple[str, Callable[[float, float], Optional[float]]], ...] = (
    (_NDVI, lambda b4, b8: None if (b8 + b4) == 0 else (b8 - b4) / (b8 + b4)),
    (_DIFF, lambda b4, b8: b8 - b4),
    (_AVG, lambda b4, b8: (b8 + b4) / 2.0),
)

Rect = Tuple[float, float, float, float]


def _band_grid(height: int, width: int, values: List[float], nodata: Optional[float]) -> RasterGrid:
    geotransform = (0.0, 1.0, 0.0, float(height), 0.0, -1.0)
    return RasterGrid(
        width=width,
        height=height,
        geotransform=geotransform,
        nodata=nodata,
        values=values,
    )


@st.composite
def _case(draw: st.DrawFn):
    width = draw(st.integers(min_value=1, max_value=5))
    height = draw(st.integers(min_value=1, max_value=5))
    n = width * height
    ints = st.integers(min_value=-8, max_value=8).map(float)
    red = draw(st.lists(ints, min_size=n, max_size=n))
    nir = draw(st.lists(ints, min_size=n, max_size=n))
    nodata_choices = st.one_of(st.none(), st.sampled_from([-999.0, 0.0, 3.0]))
    red_nodata = draw(nodata_choices)
    nir_nodata = draw(nodata_choices)
    expression, ref_fn = draw(st.sampled_from(_EXPRESSIONS))

    coord = st.integers(min_value=-3, max_value=max(width, height) + 3)
    zones: List[Feature] = []
    for i in range(draw(st.integers(min_value=0, max_value=4))):
        x0, x1, y0, y1 = draw(coord), draw(coord), draw(coord), draw(coord)
        if x0 == x1:
            x1 = x0 + 1
        if y0 == y1:
            y1 = y0 + 1
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

    return width, height, red, nir, red_nodata, nir_nodata, expression, ref_fn, zones


def _reference_index(
    height: int,
    width: int,
    red: List[float],
    nir: List[float],
    red_nodata: Optional[float],
    nir_nodata: Optional[float],
    ref_fn: Callable[[float, float], Optional[float]],
) -> List[float]:
    """Independent per-pixel index; NaN where the impl masks the pixel."""
    out: List[float] = []
    for i in range(width * height):
        v4, v8 = red[i], nir[i]
        masked = (red_nodata is not None and v4 == red_nodata) or (
            nir_nodata is not None and v8 == nir_nodata
        )
        if masked:
            out.append(math.nan)
            continue
        value = ref_fn(v4, v8)
        out.append(math.nan if value is None else float(value))
    return out


def _reference_stats(index: List[float], grid: RasterGrid, rect: Rect, stats: List[str]):
    min_x, min_y, max_x, max_y = rect
    vals: List[float] = []
    for row in range(grid.height):
        for col in range(grid.width):
            v = index[row * grid.width + col]
            if v != v:  # NaN -> masked
                continue
            x, y = grid.cell_center(col, row)
            if min_x < x < max_x and min_y < y < max_y:
                vals.append(v)
    if not vals:
        return {s: None for s in stats}
    mean = sum(vals) / len(vals)
    reduced = {
        "min": float(min(vals)),
        "max": float(max(vals)),
        "sum": float(sum(vals)),
        "mean": mean,
        "count": float(len(vals)),
        "std": (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5,
    }
    return {s: reduced[s] for s in stats}


@pytest.mark.property
@settings(deadline=None)
@given(case=_case())
def test_zonal_band_math_index_and_stats_equal_reference(case) -> None:
    """Feature: geospatial-power-pack, Property: zonal band math equals a reference.

    Validates: Requirements 7.12, 8.7, 8.8
    """
    width, height, red, nir, red_nodata, nir_nodata, expression, ref_fn, zones = case
    node, referenced = _parse_expression(expression, source_id="test")
    assert referenced == [4, 8]

    grids = {
        4: _band_grid(height, width, red, red_nodata),
        8: _band_grid(height, width, nir, nir_nodata),
    }
    index_grid = _evaluate_index(grids, referenced, node)

    expected_index = _reference_index(
        height, width, red, nir, red_nodata, nir_nodata, ref_fn
    )

    # Per-pixel index equals the reference: NaN positions match; finite values agree.
    assert len(index_grid.values) == len(expected_index)
    for got, exp in zip(index_grid.values, expected_index):
        if exp != exp:  # reference is NaN -> impl must be NaN (masked)
            assert got != got
        else:
            assert got == pytest.approx(exp)

    # Reducing the index over the zones equals an independent reference reduction.
    stats = ["min", "max", "mean", "sum", "count", "std"]
    result = compute_zonal_statistics(index_grid, FeatureCollection(features=zones), stats)
    assert len(result) == len(zones)
    for feature, zone in zip(zones, result):
        ring = feature.geometry.coordinates[0]
        xs = [pt[0] for pt in ring]
        ys = [pt[1] for pt in ring]
        rect = (min(xs), min(ys), max(xs), max(ys))
        expected = _reference_stats(expected_index, index_grid, rect, stats)
        if all(v is None for v in expected.values()):
            assert zone.no_data is True
            assert set(zone.statistics.values()) == {None}
        else:
            assert zone.no_data is False
            for name in stats:
                assert zone.statistics[name] == pytest.approx(expected[name])

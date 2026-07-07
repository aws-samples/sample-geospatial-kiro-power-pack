"""Unit tests for ``geo-raster`` zonal statistics (Requirements 8.7, 8.8).

These cover the pure compute core (:func:`geo_raster.compute_zonal_statistics`)
against a straightforward reference computation, the no-data indication for
zones with no overlapping cells (Requirement 8.8), nodata-cell exclusion, and
the validation surface (unsupported statistic, non-areal zone geometry).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pytest

from geo_common.errors import GeoError, ValidationError

from geo_raster import (
    Feature,
    FeatureCollection,
    GeoJSONGeometry,
    RasterGrid,
    ZoneStat,
    compute_zonal_statistics,
)

# A north-up 4x4 grid: origin (0, 4), 1 unit/cell, px_h = -1.
# Cell (col, row) center = (col + 0.5, 3.5 - row); value = row*4 + col (0..15).
_GT = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)


def _grid(nodata: Optional[float] = None, values: Optional[List[float]] = None) -> RasterGrid:
    vals = values if values is not None else [float(r * 4 + c) for r in range(4) for c in range(4)]
    return RasterGrid(width=4, height=4, geotransform=_GT, nodata=nodata, values=vals)


def _polygon(min_x: float, min_y: float, max_x: float, max_y: float) -> GeoJSONGeometry:
    ring = [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
        [min_x, min_y],
    ]
    return GeoJSONGeometry(type="Polygon", coordinates=[ring])


def _zone(geom: GeoJSONGeometry, zone_id: str) -> Feature:
    return Feature(geometry=geom, properties={"zone_id": zone_id})


def _reference(grid: RasterGrid, ring_box, stats):
    """A straightforward reference: collect cell-center-in-box values, reduce."""
    min_x, min_y, max_x, max_y = ring_box
    vals: List[float] = []
    for row in range(grid.height):
        for col in range(grid.width):
            v = grid.value(col, row)
            if grid.nodata is not None and v == grid.nodata:
                continue
            x, y = grid.cell_center(col, row)
            if min_x <= x <= max_x and min_y <= y <= max_y:
                vals.append(v)
    out: Dict[str, Optional[float]] = {}
    if not vals:
        return {s: None for s in stats}
    mean = sum(vals) / len(vals)
    for s in stats:
        out[s] = {
            "min": min(vals),
            "max": max(vals),
            "sum": sum(vals),
            "mean": mean,
            "count": float(len(vals)),
            "std": (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5,
        }[s]
    return out


def test_per_zone_statistics_match_reference() -> None:
    """Req 8.7: each statistic equals a straightforward reference computation."""
    grid = _grid()
    box = (0.0, 2.0, 2.0, 4.0)  # covers cells {0,1,4,5}
    fc = FeatureCollection(features=[_zone(_polygon(*box), "A")])

    stats = ["min", "max", "mean", "sum", "count"]
    result = compute_zonal_statistics(grid, fc, stats)

    assert len(result) == 1
    zone = result[0]
    assert zone.zone_id == "A"
    assert zone.no_data is False
    assert zone.statistics == _reference(grid, box, stats)
    # Concretely: {0,1,4,5} -> min 0, max 5, sum 10, mean 2.5, count 4.
    assert zone.statistics == {"min": 0.0, "max": 5.0, "mean": 2.5, "sum": 10.0, "count": 4.0}


def test_std_is_population_standard_deviation() -> None:
    """Req 8.7: ``std`` is the population standard deviation (ddof=0)."""
    grid = _grid()
    box = (0.0, 2.0, 2.0, 4.0)  # covers cells {0,1,4,5} -> values 0,1,4,5
    fc = FeatureCollection(features=[_zone(_polygon(*box), "A")])

    zone = compute_zonal_statistics(grid, fc, ["std"])[0]

    # mean 2.5; variance = (6.25+2.25+2.25+6.25)/4 = 4.25; std = sqrt(4.25).
    assert zone.no_data is False
    assert zone.statistics["std"] == pytest.approx(4.25 ** 0.5)
    assert zone.statistics["std"] == pytest.approx(_reference(grid, box, ["std"])["std"])


def test_std_of_single_cell_zone_is_zero() -> None:
    """A zone overlapping exactly one cell has zero spread (population std)."""
    grid = _grid()
    box = (0.0, 3.0, 1.0, 4.0)  # covers only cell 0 (center 0.5, 3.5)

    zone = compute_zonal_statistics(grid, FeatureCollection(features=[_zone(_polygon(*box), "A")]), ["std", "count"])[0]

    assert zone.no_data is False
    assert zone.statistics == {"std": 0.0, "count": 1.0}


def test_zone_with_no_overlapping_cells_is_no_data_others_still_returned() -> None:
    """Req 8.8: an empty zone yields no-data while other zones still get stats."""
    grid = _grid()
    inside = _zone(_polygon(0.0, 2.0, 2.0, 4.0), "inside")
    outside = _zone(_polygon(100.0, 100.0, 101.0, 101.0), "outside")
    fc = FeatureCollection(features=[outside, inside])

    result = compute_zonal_statistics(grid, fc, ["min", "max", "mean", "sum", "count"])
    by_id = {z.zone_id: z for z in result}

    # The empty zone is still returned, flagged no-data, every statistic None.
    assert by_id["outside"].no_data is True
    assert set(by_id["outside"].statistics.values()) == {None}
    # The overlapping zone still receives real statistics.
    assert by_id["inside"].no_data is False
    assert by_id["inside"].statistics["count"] == 4.0


def test_nodata_cells_are_excluded() -> None:
    """Cells equal to the grid nodata value do not contribute to a zone."""
    # Mark cells 0 and 1 (row 0, cols 0/1) as nodata; zone covers {0,1,4,5}.
    values = [float(r * 4 + c) for r in range(4) for c in range(4)]
    values[0] = -1.0
    values[1] = -1.0
    grid = _grid(nodata=-1.0, values=values)
    fc = FeatureCollection(features=[_zone(_polygon(0.0, 2.0, 2.0, 4.0), "A")])

    zone = compute_zonal_statistics(grid, fc, ["min", "max", "sum", "count"])[0]
    # Only cells 4 and 5 remain.
    assert zone.statistics == {"min": 4.0, "max": 5.0, "sum": 9.0, "count": 2.0}


def test_zone_overlapping_only_nodata_cells_is_no_data() -> None:
    """A zone overlapping only nodata cells yields a no-data indication."""
    values = [float(r * 4 + c) for r in range(4) for c in range(4)]
    values[0] = values[1] = values[4] = values[5] = -9999.0
    grid = _grid(nodata=-9999.0, values=values)
    fc = FeatureCollection(features=[_zone(_polygon(0.0, 2.0, 2.0, 4.0), "A")])

    zone = compute_zonal_statistics(grid, fc, ["mean", "count"])[0]
    assert zone.no_data is True
    assert zone.statistics == {"mean": None, "count": None}


def test_zone_id_falls_back_to_index_when_absent() -> None:
    """Zones without a zone_id/id property are identified by their position."""
    grid = _grid()
    feat = Feature(geometry=_polygon(0.0, 2.0, 2.0, 4.0), properties={})
    result = compute_zonal_statistics(grid, FeatureCollection(features=[feat]), ["count"])
    assert result[0].zone_id == "0"


def test_unsupported_statistic_is_rejected() -> None:
    """An unsupported statistic name is a validation error (Req 8.10 surface)."""
    grid = _grid()
    fc = FeatureCollection(features=[_zone(_polygon(0.0, 2.0, 2.0, 4.0), "A")])
    with pytest.raises(ValidationError) as excinfo:
        compute_zonal_statistics(grid, fc, ["median"])
    assert excinfo.value.detail == {"parameter": "stats"}


def test_empty_stats_list_is_rejected() -> None:
    grid = _grid()
    fc = FeatureCollection(features=[_zone(_polygon(0.0, 2.0, 2.0, 4.0), "A")])
    with pytest.raises(ValidationError):
        compute_zonal_statistics(grid, fc, [])


def test_non_areal_geometry_is_rejected() -> None:
    """A Point/LineString zone cannot be summarized; validation error."""
    grid = _grid()
    point = GeoJSONGeometry(type="Point", coordinates=[1.0, 1.0])
    fc = FeatureCollection(features=[_zone(point, "pt")])
    with pytest.raises(ValidationError) as excinfo:
        compute_zonal_statistics(grid, fc, ["count"])
    assert excinfo.value.detail == {"parameter": "zones"}


def test_multipolygon_zone_unions_its_parts() -> None:
    """A MultiPolygon zone aggregates cells from each of its polygons."""
    grid = _grid()
    # Two disjoint 1x1-ish boxes: one over cell {0} and one over cell {15}.
    multi = GeoJSONGeometry(
        type="MultiPolygon",
        coordinates=[
            [[[0.0, 3.0], [1.0, 3.0], [1.0, 4.0], [0.0, 4.0], [0.0, 3.0]]],  # cell 0
            [[[3.0, 0.0], [4.0, 0.0], [4.0, 1.0], [3.0, 1.0], [3.0, 0.0]]],  # cell 15
        ],
    )
    fc = FeatureCollection(features=[_zone(multi, "m")])
    zone = compute_zonal_statistics(grid, fc, ["min", "max", "count"])[0]
    assert zone.statistics == {"min": 0.0, "max": 15.0, "count": 2.0}

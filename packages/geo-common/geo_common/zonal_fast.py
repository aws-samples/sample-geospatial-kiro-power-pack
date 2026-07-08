"""Vectorized (numpy) fast path for :func:`geo_raster.zonal.compute_zonal_statistics`.

This is an **optional** accelerator, imported only when the ``numpy`` extra is
installed (``geo-raster[fast]``). It is a drop-in, semantically identical
replacement for the dependency-free engine in :mod:`geo_raster.zonal`:

* Cell membership uses the same **centroid rule** (a cell contributes when its
  center lies inside the zone polygon) and the same even-odd ray-casting
  convention as :func:`geo_raster.geometry.point_in_polygon`, just evaluated over
  every cell at once with numpy boolean arrays instead of a per-cell Python loop.
* nodata / NaN cells are excluded identically, a zone with no overlapping cells
  yields the no-data indication (Requirement 8.8), and ``std`` is the population
  standard deviation (ddof=0), matching the pure engine and rasterstats/NumPy.

The differential property test (``test_zonal_fast_differential``) asserts this
path returns exactly the pure engine's result across randomized inputs, so the
two never drift.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from geo_common.errors import ValidationError

from geo_common.geometry import geometry_bbox, iter_polygons
from geo_common.raster_models import FeatureCollection, RasterGrid, ZoneStat
from geo_common.zonal import _zone_id

__all__ = ["compute_zonal_statistics_numpy"]


def _ring_mask(xs: np.ndarray, ys: np.ndarray, ring) -> np.ndarray:
    """Even-odd ray cast for one ring, vectorized over points ``(xs, ys)``.

    Mirrors :func:`geo_raster.geometry._ring_contains`: for each edge (j -> i),
    a point's horizontal ray crosses it when the edge straddles the point's
    ``y`` and the crossing ``x`` lies to the right of the point; the parity of
    crossings determines membership.
    """
    inside = np.zeros(xs.shape, dtype=bool)
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        denom = yj - yi
        if denom != 0.0:
            straddles = (yi > ys) != (yj > ys)
            x_cross = (xj - xi) * (ys - yi) / denom + xi
            inside ^= straddles & (xs < x_cross)
        j = i
    return inside


def _polygon_mask(xs: np.ndarray, ys: np.ndarray, rings) -> np.ndarray:
    """Membership in a single polygon (exterior ring minus holes)."""
    if not rings:
        return np.zeros(xs.shape, dtype=bool)
    inside = _ring_mask(xs, ys, rings[0])
    for hole in rings[1:]:
        inside &= ~_ring_mask(xs, ys, hole)
    return inside


def _geometry_mask(xs: np.ndarray, ys: np.ndarray, polygons) -> np.ndarray:
    """Membership in any polygon of a (multi)polygon geometry."""
    member = np.zeros(xs.shape, dtype=bool)
    for rings in polygons:
        member |= _polygon_mask(xs, ys, rings)
    return member


def _reduce(name: str, values: np.ndarray) -> float:
    """Compute one statistic over a non-empty array of overlapping values."""
    if name == "count":
        return float(values.size)
    if name == "sum":
        return float(values.sum())
    if name == "min":
        return float(values.min())
    if name == "max":
        return float(values.max())
    if name == "mean":
        return float(values.mean())
    if name == "std":
        return float(values.std())  # numpy default ddof=0 == population std
    raise ValidationError(  # pragma: no cover - guarded by _validate_stats
        f"unsupported statistic {name!r}", detail={"parameter": "stats"}
    )


def compute_zonal_statistics_numpy(
    grid: RasterGrid,
    zones: FeatureCollection,
    requested: List[str],
    *,
    source: str = "geo-raster",
) -> List[ZoneStat]:
    """Vectorized equivalent of the pure zonal engine (pre-validated ``requested``)."""
    width, height = grid.width, grid.height
    values = np.asarray(grid.values, dtype=np.float64).reshape(-1)

    # Cell-center coordinates for every cell, in the grid's row-major order.
    if width and height:
        origin_x, px_w, row_rot, origin_y, col_rot, px_h = grid.geotransform
        rows, cols = np.meshgrid(
            np.arange(height, dtype=np.float64),
            np.arange(width, dtype=np.float64),
            indexing="ij",
        )
        fc = cols.reshape(-1) + 0.5
        fr = rows.reshape(-1) + 0.5
        cx = origin_x + px_w * fc + row_rot * fr
        cy = origin_y + col_rot * fc + px_h * fr

        valid = ~np.isnan(values)
        if grid.nodata is not None:
            valid &= values != grid.nodata
        xs = cx[valid]
        ys = cy[valid]
        vs = values[valid]
    else:
        xs = ys = vs = np.empty(0, dtype=np.float64)

    results: List[ZoneStat] = []
    for index, feature in enumerate(zones.features):
        zone_id = _zone_id(feature.properties or {}, index)
        if feature.geometry is None:
            raise ValidationError(
                f"zone {zone_id!r} has no geometry",
                source=source,
                detail={"parameter": "zones"},
            )
        geojson = feature.geometry.to_geojson()
        polygons = iter_polygons(geojson, source=source)

        # Restrict the point-in-polygon test to cells whose center lies in the
        # zone's bounding box: a polygon is a subset of its (closed) bbox, so
        # this never drops an interior cell — it only avoids ray-casting cells
        # that cannot possibly be inside (a large win for small zones over a
        # big window). The result is identical to testing every cell.
        if vs.size:
            bbox = geometry_bbox(geojson, source=source)
            if bbox is None:
                overlapping = vs[:0]
            else:
                min_x, min_y, max_x, max_y = bbox
                candidate = (xs >= min_x) & (xs <= max_x) & (ys >= min_y) & (ys <= max_y)
                cand_idx = np.nonzero(candidate)[0]
                if cand_idx.size:
                    inside = _geometry_mask(xs[cand_idx], ys[cand_idx], polygons)
                    overlapping = vs[cand_idx[inside]]
                else:
                    overlapping = vs[:0]
        else:
            overlapping = vs

        if overlapping.size == 0:
            results.append(
                ZoneStat(
                    zone_id=zone_id,
                    statistics={name: None for name in requested},
                    no_data=True,
                )
            )
            continue

        results.append(
            ZoneStat(
                zone_id=zone_id,
                statistics={name: _reduce(name, overlapping) for name in requested},
                no_data=False,
            )
        )
    return results

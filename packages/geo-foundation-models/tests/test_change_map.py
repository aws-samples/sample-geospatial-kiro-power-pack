"""Tests for the per-tile change_map tool (roadmap B#5), hermetic.

Two aligned in-memory COGs are built with ``geo_common.testing`` and read
through recording byte-range readers (no network). A tiny real-weight stand-in
(a DeterministicLocalBackend re-labelled with a non-stand-in backend_id) drives
the calibrated path so change varies with content and identical dates score 0.
Coverage: grid shape + bbox, identical-vs-different dates, the honesty gate
(refuse / caveat under the deterministic stand-in), mis-registration, AOI
overlap, tile-count cap, and a randomized invariant property.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geo_common.errors import ValidationError
from geo_common.testing import RecordingByteRangeReader, build_cog

from geo_foundation_models.change_map import change_map
from geo_foundation_models.embedding import DeterministicLocalBackend

# 4x4 asset, north-up, covering x/y in [0, 4].
_GT = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
_AOI = (0.0, 0.0, 4.0, 4.0)


class _FakeRealBackend(DeterministicLocalBackend):
    """Content-dependent deterministic vectors, but labelled as a real backend."""

    backend_id = "test-real"


def _cog(values, *, width=4, height=4, dtype="uint16"):
    return build_cog(
        width=width, height=height, tile_width=2, tile_height=2,
        bands=[values], dtype=dtype, geotransform=_GT,
    )


def _readers(a_values, b_values, **kw):
    return {
        "a": RecordingByteRangeReader(_cog(a_values, **kw)),
        "b": RecordingByteRangeReader(_cog(b_values, **kw)),
    }


async def _run(a_values, b_values, *, backend=None, **kw):
    return await change_map(
        raster_href_a="s3://amzn-s3-demo-bucket/a.tif",
        raster_href_b="s3://amzn-s3-demo-bucket/b.tif",
        model="Clay",
        aoi_bbox=_AOI,
        tile_size=2,
        backend=backend,
        readers=_readers(a_values, b_values),
        **kw,
    )


# --- grid shape + geometry ------------------------------------------------

async def test_change_map_grid_shape_and_bbox() -> None:
    a = list(range(16))
    b = [v * 2 for v in range(16)]
    result = await _run(a, b, backend=_FakeRealBackend())

    assert result.rows == 2 and result.cols == 2
    assert result.cell_count == 4 and len(result.cells) == 4
    assert result.backend == "test-real"
    assert result.calibrated is True and result.caveat is None
    assert result.model == "Clay" and result.dimension == 768

    # Every cell has a bbox inside the AOI and a change in [0, 1].
    for cell in result.cells:
        assert 0.0 <= cell.change <= 1.0
        min_x, min_y, max_x, max_y = cell.bbox
        assert _AOI[0] <= min_x < max_x <= _AOI[2]
        assert _AOI[1] <= min_y < max_y <= _AOI[3]

    # Grid indices cover a full 2x2.
    assert {(c.row, c.col) for c in result.cells} == {(0, 0), (0, 1), (1, 0), (1, 1)}


async def test_identical_dates_score_zero_change() -> None:
    values = list(range(16))
    result = await _run(values, values, backend=_FakeRealBackend())
    assert all(cell.change == 0.0 for cell in result.cells)
    assert result.structure_only is False


async def test_different_dates_produce_nonzero_change_somewhere() -> None:
    rng = random.Random(7)
    a = [rng.randint(0, 255) for _ in range(16)]
    b = [rng.randint(0, 255) for _ in range(16)]
    result = await _run(a, b, backend=_FakeRealBackend())
    assert any(cell.change > 0.0 for cell in result.cells)


# --- per-zone reduction ---------------------------------------------------

def _fc(*polys):
    from geo_common.raster_models import Feature, FeatureCollection, GeoJSONGeometry

    feats = []
    for zid, ring in polys:
        feats.append(Feature(
            geometry=GeoJSONGeometry(type="Polygon", coordinates=[ring]),
            properties={"zone_id": zid},
        ))
    return FeatureCollection(features=feats)


async def test_change_map_per_zone_reduction() -> None:
    a = list(range(16))
    b = [v * 2 for v in range(16)]
    # Left half (x in [0,2]) contains 2 tile centers; a detached box contains none.
    left = ("left", [[0, 0], [2, 0], [2, 4], [0, 4], [0, 0]])
    empty = ("empty", [[10, 10], [11, 10], [11, 11], [10, 11], [10, 10]])
    result = await _run(a, b, backend=_FakeRealBackend(), zones=_fc(left, empty))

    assert result.zones is not None and len(result.zones) == 2
    by_id = {z.zone_id: z for z in result.zones}
    assert by_id["left"].tile_count == 2
    assert by_id["left"].mean_change is not None
    assert 0.0 <= by_id["left"].mean_change <= 1.0
    assert by_id["left"].max_change >= by_id["left"].mean_change
    # A zone containing no tile center is a no-data indication.
    assert by_id["empty"].tile_count == 0
    assert by_id["empty"].mean_change is None


async def test_change_map_without_zones_has_none_zone_summary() -> None:
    result = await _run(list(range(16)), [v + 1 for v in range(16)],
                        backend=_FakeRealBackend())
    assert result.zones is None


# --- honesty gate ---------------------------------------------------------

async def test_standin_backend_sets_caveat_and_uncalibrated() -> None:
    a = list(range(16))
    b = [v + 1 for v in range(16)]
    result = await _run(a, b, backend=None)  # deterministic stand-in
    assert result.backend == "deterministic-local"
    assert result.calibrated is False
    assert result.caveat is not None and "calibrated" in result.caveat.lower()
    # It still returns a full grid.
    assert result.cell_count == 4


async def test_require_real_backend_refuses_under_standin() -> None:
    with pytest.raises(ValidationError) as exc:
        await _run(list(range(16)), list(range(16)), backend=None,
                   require_real_backend=True)
    assert exc.value.detail.get("parameter") == "require_real_backend"


# --- guards ---------------------------------------------------------------

async def test_misregistered_assets_are_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        await change_map(
            raster_href_a="s3://b/a.tif", raster_href_b="s3://b/b.tif",
            model="Clay", aoi_bbox=_AOI, tile_size=2, backend=_FakeRealBackend(),
            readers={
                "a": RecordingByteRangeReader(_cog(list(range(16)))),
                "b": RecordingByteRangeReader(_cog(list(range(36)), width=6, height=6)),
            },
        )
    assert exc.value.detail.get("parameter") == "raster_href_b"


async def test_non_overlapping_aoi_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        await change_map(
            raster_href_a="s3://b/a.tif", raster_href_b="s3://b/b.tif",
            model="Clay", aoi_bbox=(100.0, 100.0, 101.0, 101.0), tile_size=2,
            backend=_FakeRealBackend(),
            readers=_readers(list(range(16)), list(range(16))),
        )
    assert exc.value.detail.get("parameter") == "aoi_bbox"


async def test_tile_count_cap_is_enforced() -> None:
    with pytest.raises(ValidationError) as exc:
        await change_map(
            raster_href_a="s3://b/a.tif", raster_href_b="s3://b/b.tif",
            model="Clay", aoi_bbox=_AOI, tile_size=1, max_tiles=3,
            backend=_FakeRealBackend(),
            readers=_readers(list(range(16)), list(range(16))),
        )
    assert exc.value.detail.get("parameter") == "max_tiles"


async def test_out_of_range_band_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        await change_map(
            raster_href_a="s3://b/a.tif", raster_href_b="s3://b/b.tif",
            model="Clay", aoi_bbox=_AOI, tile_size=2, bands=[2],
            backend=_FakeRealBackend(),
            readers=_readers(list(range(16)), list(range(16))),
        )
    assert exc.value.detail.get("parameter") == "bands"


# --- property: grid invariants --------------------------------------------

@pytest.mark.property
@settings(deadline=None, max_examples=40)
@given(
    a=st.lists(st.integers(min_value=0, max_value=255), min_size=16, max_size=16),
    b=st.lists(st.integers(min_value=0, max_value=255), min_size=16, max_size=16),
    tile_size=st.integers(min_value=1, max_value=4),
)
async def test_change_map_invariants(a, b, tile_size) -> None:
    """Feature: geospatial-power-pack, per-tile change map (roadmap B#5).

    For any co-registered pair and tile size: the grid is rows*cols cells, every
    change is in [0,1], and identical dates score exactly 0 everywhere.
    """
    result = await change_map(
        raster_href_a="s3://b/a.tif", raster_href_b="s3://b/b.tif",
        model="Clay", aoi_bbox=_AOI, tile_size=tile_size, max_tiles=64,
        backend=_FakeRealBackend(),
        readers=_readers(a, b),
    )
    assert result.cell_count == result.rows * result.cols
    assert all(0.0 <= c.change <= 1.0 for c in result.cells)

    same = await change_map(
        raster_href_a="s3://b/a.tif", raster_href_b="s3://b/b.tif",
        model="Clay", aoi_bbox=_AOI, tile_size=tile_size, max_tiles=64,
        backend=_FakeRealBackend(),
        readers=_readers(a, a),
    )
    assert all(c.change == 0.0 for c in same.cells)

"""Unit tests for ``detect_change`` and ``segment`` (Req 9.5, 9.9).

Example-based coverage of the task 7.2 contract:

* ``detect_change`` returns a scalar in ``[0.0, 1.0]`` for equal-dimension
  embeddings, with identical embeddings yielding ``0.0`` and higher values for
  greater change (9.5); a dimensionality mismatch raises a validation error and
  produces no measure (9.9).
* ``segment`` produces a SAMGeo-style mask, validating the tile first (9.7-style
  reuse) and honoring point/box prompts.
"""

from __future__ import annotations

import pytest

from geo_common.errors import ErrorCategory, ValidationError

from geo_foundation_models.change import detect_change
from geo_foundation_models.segmentation import SAMGEO_MODEL_NAME, segment
from geo_foundation_models.models import RasterTile, SegmentationMask


def _tile(width: int = 8, height: int = 8, bands: int = 1, fmt: str = "GTiff", data=None) -> RasterTile:
    return RasterTile(width=width, height=height, bands=bands, format=fmt, data=data)


# --- Requirement 9.5: normalized change measure --------------------------

def test_identical_embeddings_yield_zero_change():
    emb = [0.1, 0.2, 0.3, 0.4]
    assert detect_change(emb, emb) == 0.0


def test_identical_zero_embeddings_yield_zero_change():
    assert detect_change([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) == 0.0


def test_opposite_embeddings_yield_maximum_change():
    a = [1.0, 0.0, 0.0]
    b = [-1.0, 0.0, 0.0]
    assert detect_change(a, b) == pytest.approx(1.0)


def test_orthogonal_embeddings_yield_midpoint_change():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert detect_change(a, b) == pytest.approx(0.5)


def test_change_is_in_unit_interval():
    a = [3.0, -2.0, 0.5, 7.0]
    b = [-1.0, 4.0, 2.0, -3.0]
    change = detect_change(a, b)
    assert 0.0 <= change <= 1.0


def test_more_change_gives_higher_value():
    base = [1.0, 0.0, 0.0]
    small = [0.9, 0.1, 0.0]
    large = [-0.5, 0.5, 0.7]
    assert detect_change(base, small) < detect_change(base, large)


def test_scale_invariant_for_collinear_embeddings():
    a = [1.0, 2.0, 3.0]
    b = [2.0, 4.0, 6.0]  # same direction, different magnitude
    assert detect_change(a, b) == pytest.approx(0.0, abs=1e-9)


# --- Requirement 9.9: dimensionality mismatch rejected -------------------

def test_dimensionality_mismatch_rejected():
    with pytest.raises(ValidationError) as exc:
        detect_change([0.1, 0.2, 0.3], [0.1, 0.2])
    assert exc.value.category is ErrorCategory.VALIDATION
    assert exc.value.detail == {"dimension_a": 3, "dimension_b": 2}


def test_mismatch_produces_no_measure():
    try:
        detect_change([1.0], [1.0, 2.0])
    except ValidationError as exc:
        assert exc.category is ErrorCategory.VALIDATION
    else:  # pragma: no cover
        pytest.fail("expected ValidationError for dimensionality mismatch")


# --- SAMGeo-style segmentation -------------------------------------------

def test_segment_returns_full_mask_for_uniform_tile():
    result = segment(_tile(width=4, height=4))
    assert isinstance(result, SegmentationMask)
    assert result.model == SAMGEO_MODEL_NAME
    assert result.width == 4 and result.height == 4
    assert len(result.mask) == 16
    # No pixel data -> a single whole-tile segment.
    assert result.num_segments == 1
    assert set(result.mask) == {1}


def test_segment_automatic_partitions_by_intensity():
    # Left half dark, right half bright -> at least two segments.
    width, height = 4, 2
    rows = []
    for _ in range(height):
        rows.extend([0.0, 0.0, 250.0, 250.0])
    result = segment(_tile(width=width, height=height, data=rows))
    assert result.num_segments >= 2
    assert len(result.mask) == width * height


def test_segment_point_prompts_make_one_region_each():
    result = segment(
        _tile(width=8, height=8),
        prompts=[
            {"type": "point", "x": 1, "y": 1},
            {"type": "point", "x": 6, "y": 6},
        ],
    )
    assert result.num_segments == 2
    assert set(result.mask) == {1, 2}  # full Voronoi cover, no background


def test_segment_box_prompt_labels_only_inside_box():
    result = segment(
        _tile(width=8, height=8),
        prompts=[{"type": "box", "x_min": 2, "y_min": 2, "x_max": 5, "y_max": 5}],
    )
    assert result.num_segments == 1
    # 4x4 box region labeled 1, the rest background 0.
    assert result.mask.count(1) == 16
    assert result.mask.count(0) == 64 - 16


def test_segment_validates_tile_before_segmenting():
    with pytest.raises(ValidationError) as exc:
        segment(_tile(width=0, height=8))
    assert exc.value.category is ErrorCategory.VALIDATION


def test_segment_oversized_tile_rejected():
    with pytest.raises(ValidationError) as exc:
        segment(_tile(width=2048, height=8))
    assert exc.value.category is ErrorCategory.VALIDATION


def test_segment_out_of_bounds_point_prompt_rejected():
    with pytest.raises(ValidationError) as exc:
        segment(_tile(width=8, height=8), prompts=[{"type": "point", "x": 99, "y": 1}])
    assert exc.value.category is ErrorCategory.VALIDATION


def test_segment_unsupported_prompt_type_rejected():
    with pytest.raises(ValidationError) as exc:
        segment(_tile(width=8, height=8), prompts=[{"type": "scribble", "x": 1, "y": 1}])
    assert exc.value.category is ErrorCategory.VALIDATION


# --- Response-size guardrail: oversized mask is refused with guidance ------


def test_segment_tile_over_mask_pixel_cap_rejected():
    """A tile exceeding the mask-pixel cap is refused so no huge mask returns."""
    from geo_foundation_models.segmentation import DEFAULT_MAX_MASK_PIXELS

    # 512x512 = 262,144 pixels > the 65,536 default cap, but still within the
    # 1024x1024 validate_tile dimension limit, so the cap (not the dimension
    # check) is what rejects it.
    with pytest.raises(ValidationError) as exc:
        segment(_tile(width=512, height=512))
    assert exc.value.category is ErrorCategory.VALIDATION
    assert exc.value.detail["parameter"] == "max_mask_pixels"
    assert exc.value.detail["pixel_count"] == 512 * 512
    assert exc.value.detail["max_mask_pixels"] == DEFAULT_MAX_MASK_PIXELS


def test_segment_mask_pixel_cap_is_configurable():
    """Raising the cap allows a larger tile to be segmented."""
    result = segment(_tile(width=300, height=300), max_mask_pixels=300 * 300)
    assert isinstance(result, SegmentationMask)
    assert len(result.mask) == 300 * 300


# --- Server tool registration --------------------------------------------

@pytest.mark.asyncio
async def test_server_registers_and_runs_new_tools():
    from geo_foundation_models.server import GeoFoundationModelsServer

    server = GeoFoundationModelsServer()
    assert "detect_change" in server.tool_names()
    assert "segment" in server.tool_names()

    change = await server.detect_change(embedding_a=[1.0, 0.0], embedding_b=[1.0, 0.0])
    assert change == 0.0

    mask = await server.segment(tile=_tile(width=4, height=4))
    assert isinstance(mask, SegmentationMask)
    assert mask.num_segments == 1

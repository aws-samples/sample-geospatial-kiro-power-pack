"""Unit tests for foundation-model tool validation (Task 7.5; Req 9.7, 9.9).

These example-based tests focus on the *rejection* contracts of the
Foundation_Model_Server, complementing the broader behavioral tests in
``test_embed_tile.py`` and ``test_detect_change_segment.py``:

* **Requirement 9.7** - an empty, oversized, or unsupported-format imagery tile
  is rejected with an ``Error_Taxonomy`` validation error that *identifies the
  validation failure*, and **no embedding is produced**. The tests exercise both
  the standalone :func:`validate_tile` guard and the full :func:`embed_tile`
  entry point, and assert on the structured ``detail`` so the error genuinely
  identifies the failing attribute.
* **Requirement 9.9** - two embeddings of differing dimensionality submitted to
  :func:`detect_change` are rejected with a validation error identifying the
  dimensionality mismatch, and **no change measure is produced**.
"""

from __future__ import annotations

import pytest

from geo_common.errors import ErrorCategory, ValidationError

from geo_foundation_models.change import detect_change
from geo_foundation_models.embedding import (
    DEFAULT_SUPPORTED_FORMATS,
    embed_tile,
    validate_tile,
)
from geo_foundation_models.models import MAX_TILE_DIMENSION, RasterTile


def _tile(
    width: int = 64,
    height: int = 64,
    bands: int = 3,
    fmt: str = "GTiff",
    data=None,
) -> RasterTile:
    return RasterTile(width=width, height=height, bands=bands, format=fmt, data=data)


# --- Requirement 9.7: tile validation failures ---------------------------


@pytest.mark.parametrize(
    "tile, bad_attr",
    [
        (_tile(width=0), "width"),
        (_tile(height=0), "height"),
        (_tile(bands=0), "bands"),
        (_tile(width=0, height=0, bands=0), "width"),
    ],
)
def test_validate_tile_rejects_empty_tile_with_detail(tile, bad_attr):
    """An empty tile is rejected and the error detail identifies the failure."""
    with pytest.raises(ValidationError) as exc:
        validate_tile(tile)
    err = exc.value
    assert err.category is ErrorCategory.VALIDATION
    # The failure is identified: the offending zero attribute is in the detail.
    assert err.detail is not None
    assert err.detail.get(bad_attr) == 0
    assert "empty" in err.message.lower()


@pytest.mark.parametrize(
    "tile",
    [
        _tile(width=MAX_TILE_DIMENSION + 1, height=64),
        _tile(width=64, height=MAX_TILE_DIMENSION + 1),
        _tile(width=4096, height=4096),
    ],
)
def test_validate_tile_rejects_oversized_tile_with_detail(tile):
    """An oversized tile is rejected and the limit is reported in the detail."""
    with pytest.raises(ValidationError) as exc:
        validate_tile(tile)
    err = exc.value
    assert err.category is ErrorCategory.VALIDATION
    assert err.detail is not None
    assert err.detail.get("max_dimension") == MAX_TILE_DIMENSION
    assert "oversized" in err.message.lower()


# Note: an empty-string format cannot be constructed (RasterTile.format has
# min_length=1); the whitespace-only case covers the "effectively empty after
# strip()" format at the validate_tile layer.
@pytest.mark.parametrize("fmt", ["bmp", "tiff8", "exr", "   "])
def test_validate_tile_rejects_unsupported_format_with_detail(fmt):
    """An unsupported raster format is rejected with the format in the detail."""
    with pytest.raises(ValidationError) as exc:
        validate_tile(_tile(fmt=fmt))
    err = exc.value
    assert err.category is ErrorCategory.VALIDATION
    assert err.detail is not None
    assert err.detail.get("format") == fmt
    assert sorted(err.detail.get("supported_formats")) == sorted(
        {f.lower() for f in DEFAULT_SUPPORTED_FORMATS}
    )


def test_validate_tile_rejects_empty_inlined_data():
    """Inlined-but-empty pixel data counts as an empty tile (Req 9.7)."""
    with pytest.raises(ValidationError) as exc:
        validate_tile(_tile(data=[]))
    assert exc.value.category is ErrorCategory.VALIDATION
    assert exc.value.detail == {"data_length": 0}


def test_validate_tile_rejects_mismatched_inlined_data_length():
    """Inlined data whose length disagrees with the geometry is rejected."""
    with pytest.raises(ValidationError) as exc:
        validate_tile(_tile(width=4, height=4, bands=1, data=[0.0] * 7))
    err = exc.value
    assert err.category is ErrorCategory.VALIDATION
    assert err.detail == {"data_length": 7, "expected_length": 16}


def test_validate_tile_accepts_supported_formats_case_insensitively():
    """A valid tile passes validation regardless of format casing (no raise)."""
    for fmt in ("GTiff", "geotiff", "COG", "PNG", "JpEg", "numpy"):
        assert validate_tile(_tile(fmt=fmt)) is None


@pytest.mark.parametrize(
    "tile",
    [
        _tile(width=0),  # empty
        _tile(height=0),  # empty
        _tile(bands=0),  # empty
        _tile(width=MAX_TILE_DIMENSION + 1),  # oversized
        _tile(fmt="bmp"),  # unsupported format
        _tile(data=[]),  # empty inlined data
    ],
)
def test_embed_tile_rejection_produces_no_embedding(tile):
    """A rejected tile raises and yields no EmbeddingResult (Req 9.7)."""
    result = None
    with pytest.raises(ValidationError) as exc:
        result = embed_tile(tile, "Clay")
    assert exc.value.category is ErrorCategory.VALIDATION
    assert result is None  # nothing was assigned/returned


def test_embed_tile_validates_before_embedding():
    """Selection is fine but a bad tile still aborts before any embedding work."""
    # A known model (selection passes) with an oversized tile (validation fails).
    with pytest.raises(ValidationError) as exc:
        embed_tile(_tile(width=5000, height=5000), "SatCLIP")
    assert exc.value.category is ErrorCategory.VALIDATION
    assert "oversized" in exc.value.message.lower()


# --- Requirement 9.9: dimensionality-mismatch rejection ------------------


@pytest.mark.parametrize(
    "a, b, dim_a, dim_b",
    [
        ([0.1, 0.2, 0.3], [0.1, 0.2], 3, 2),
        ([1.0], [1.0, 2.0, 3.0], 1, 3),
        ([], [1.0], 0, 1),
        ([0.0] * 768, [0.0] * 256, 768, 256),
    ],
)
def test_detect_change_rejects_dimensionality_mismatch(a, b, dim_a, dim_b):
    """Mismatched dimensionality is rejected and both lengths are identified."""
    with pytest.raises(ValidationError) as exc:
        detect_change(a, b)
    err = exc.value
    assert err.category is ErrorCategory.VALIDATION
    assert err.detail == {"dimension_a": dim_a, "dimension_b": dim_b}
    assert "mismatch" in err.message.lower()


def test_detect_change_mismatch_produces_no_measure():
    """The mismatch path raises rather than returning any scalar measure."""
    measure = None
    with pytest.raises(ValidationError) as exc:
        measure = detect_change([1.0, 2.0, 3.0, 4.0], [1.0, 2.0])
    assert exc.value.category is ErrorCategory.VALIDATION
    assert measure is None

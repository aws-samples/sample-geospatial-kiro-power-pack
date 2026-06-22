"""Property test for the normalized embedding change measure.

Feature: geospatial-power-pack, Property 12: Change measure is normalized

This module validates :func:`geo_foundation_models.change.detect_change`
against Property 12 of the design (Validates: Requirements 9.5):

*For any* two equal-dimensionality embeddings, the change-detection result is a
scalar in the range ``0.0`` to ``1.0``, and two identical embeddings yield the
minimum change value (``0.0``).

The minimum-case count (>= 100 generated examples) is inherited from the
``geospatial-power-pack`` Hypothesis profile loaded by the root ``conftest.py``.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geo_foundation_models.change import detect_change


# Finite floats in a wide-but-bounded range. Embedding components are real
# numbers; bounding the magnitude keeps cosine arithmetic well-conditioned
# while still exercising positive, negative, zero, and mixed-sign vectors.
_components = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def equal_dimension_pairs(draw: st.DrawFn) -> tuple[list[float], list[float]]:
    """Draw two embeddings that share the same (>= 1) dimensionality."""
    dimension = draw(st.integers(min_value=1, max_value=64))
    embedding_a = draw(st.lists(_components, min_size=dimension, max_size=dimension))
    embedding_b = draw(st.lists(_components, min_size=dimension, max_size=dimension))
    return embedding_a, embedding_b


@pytest.mark.property
@settings(deadline=None)  # max_examples (>=100) inherited from the loaded profile
@given(pair=equal_dimension_pairs())
def test_change_measure_is_normalized(pair: tuple[list[float], list[float]]) -> None:
    """Feature: geospatial-power-pack, Property 12: Change measure is normalized.

    Validates: Requirements 9.5
    """
    embedding_a, embedding_b = pair

    # (1) For any equal-dimension pair the result is a scalar in [0.0, 1.0].
    change = detect_change(embedding_a, embedding_b)
    assert isinstance(change, float)
    assert 0.0 <= change <= 1.0

    # (2) Identical embeddings yield the minimum change value, exactly 0.0.
    assert detect_change(embedding_a, embedding_a) == 0.0
    assert detect_change(embedding_b, embedding_b) == 0.0


@pytest.mark.property
@settings(deadline=None)
@given(
    embedding=st.lists(_components, min_size=1, max_size=64),
)
def test_identical_embeddings_yield_zero_change(embedding: list[float]) -> None:
    """Identical embeddings of any dimensionality yield exactly 0.0 change.

    Feature: geospatial-power-pack, Property 12: Change measure is normalized
    Validates: Requirements 9.5
    """
    change = detect_change(embedding, embedding)
    assert change == 0.0
    assert math.isfinite(change)

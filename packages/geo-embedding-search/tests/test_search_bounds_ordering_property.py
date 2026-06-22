"""Property test for vector search bounds and ordering (design Property 11).

Feature: geospatial-power-pack, Property 11: Vector search is bounded and ordered

This module validates :func:`geo_embedding_search.store.search_embeddings` against
Property 11 of the design (Validates: Requirements 9.4):

*For any* set of stored embeddings, any query embedding, and any ``K`` in
``1``-``1,000``, the search returns **at most ``K``** results ordered by
**non-increasing** similarity score, and returns an **empty** result set when no
embeddings are stored.

The minimum-case count (>= 100 generated examples) is inherited from the
``geospatial-power-pack`` Hypothesis profile loaded by the root ``conftest.py``.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geo_embedding_search.models import VectorSearchHit
from geo_embedding_search.store import (
    MAX_K,
    MIN_K,
    InMemoryVectorStore,
    cosine_similarity,
    search_embeddings,
    store_embedding,
)


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
def _embeddings(draw: st.DrawFn) -> list[float]:
    """Draw a single non-empty, finite embedding (any dimensionality >= 1)."""
    dimension = draw(st.integers(min_value=1, max_value=32))
    return draw(st.lists(_components, min_size=dimension, max_size=dimension))


@st.composite
def _search_cases(
    draw: st.DrawFn,
) -> tuple[list[list[float]], list[float], int]:
    """Draw a (stored embeddings, query embedding, K) triple.

    The stored set may be empty (to exercise the empty-store rule) and its
    members may differ in dimensionality from one another and from the query
    (``cosine_similarity`` is total), so the property is checked across the
    whole input space described by Requirement 9.4.
    """
    stored = draw(st.lists(_embeddings(), min_size=0, max_size=25))
    query = draw(_embeddings())
    k = draw(st.integers(min_value=MIN_K, max_value=MAX_K))
    return stored, query, k


@pytest.mark.property
@settings(deadline=None)  # max_examples (>=100) inherited from the loaded profile
@given(case=_search_cases())
def test_vector_search_is_bounded_and_ordered(
    case: tuple[list[list[float]], list[float], int],
) -> None:
    """Feature: geospatial-power-pack, Property 11: Vector search is bounded and ordered.

    Validates: Requirements 9.4
    """
    stored, query, k = case

    store = InMemoryVectorStore()
    for embedding in stored:
        store_embedding(embedding, store=store)

    hits = search_embeddings(query, k, store=store)

    # (1) Empty store => empty result set, regardless of query or K.
    if not stored:
        assert hits == []
        return

    # (2) Bounded: at most K results, and never more than what is stored.
    assert len(hits) <= k
    assert len(hits) == min(k, len(stored))

    # (3) Ordered by non-increasing similarity score.
    similarities = [hit.similarity for hit in hits]
    assert all(
        similarities[i] >= similarities[i + 1] for i in range(len(similarities) - 1)
    )

    # (4) Each reported similarity equals the cosine score for that record, and
    #     every returned hit is a distinct, well-formed VectorSearchHit.
    seen_ids: set[str] = set()
    for hit in hits:
        assert isinstance(hit, VectorSearchHit)
        assert hit.id not in seen_ids
        seen_ids.add(hit.id)
        assert hit.similarity == pytest.approx(
            cosine_similarity(query, hit.embedding)
        )

    # (5) The returned hits are genuinely the top-scoring records: no withheld
    #     record outscores the smallest similarity that was returned.
    returned = {hit.id for hit in hits}
    all_scored = sorted(
        (cosine_similarity(query, rec.embedding) for rec in store.records()),
        reverse=True,
    )
    assert similarities == pytest.approx(all_scored[: len(hits)])


@pytest.mark.property
@settings(deadline=None)
@given(query=_embeddings(), k=st.integers(min_value=MIN_K, max_value=MAX_K))
def test_empty_store_returns_empty_for_any_query_and_k(
    query: list[float], k: int
) -> None:
    """Empty store yields an empty result for any query embedding and any K.

    Feature: geospatial-power-pack, Property 11: Vector search is bounded and ordered
    Validates: Requirements 9.4
    """
    store = InMemoryVectorStore()
    assert search_embeddings(query, k, store=store) == []

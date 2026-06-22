"""Unit tests for ``geo-embedding-search`` embedding storage and search (task 15.1).

Covers the behavior task 15.1 requires of
:mod:`geo_embedding_search.store` and the :class:`GeoEmbeddingSearchServer` tools:

* Req 9.3 - ``store_embedding`` persists an embedding plus metadata and returns
  a confirmation that the record is retrievable by subsequent queries.
* Req 9.4 - ``search_embeddings`` returns up to K (1-1000) stored embeddings
  ordered by descending similarity, and an empty result when nothing is stored.
* Req 9.8 - a failed persistence raises an ``Error_Taxonomy`` error and leaves
  no partial record.

Property-based coverage of the bound/ordering (Property 11) and the
store-then-retrieve round-trip (Property 13) lives in the dedicated property
tests (tasks 15.2 / 15.3).
"""

from __future__ import annotations

import pytest

from geo_common.errors import ErrorCategory, UpstreamError, ValidationError

from geo_embedding_search.models import StoreConfirmation, VectorSearchHit
from geo_embedding_search.store import (
    InMemoryVectorStore,
    VectorStoreBackend,
    cosine_similarity,
    search_embeddings,
    store_embedding,
)
from geo_embedding_search.server import GeoEmbeddingSearchServer


# --- store_embedding: persistence + retrievability (Req 9.3) ---------------


def test_store_embedding_persists_and_confirms_retrievable() -> None:
    """Req 9.3: a stored embedding is confirmed retrievable by later queries."""
    store = InMemoryVectorStore()
    confirmation = store_embedding([0.1, 0.2, 0.3], {"area": "scene-1"}, store=store)

    assert isinstance(confirmation, StoreConfirmation)
    assert confirmation.retrievable is True
    assert confirmation.dimension == 3
    assert store.contains(confirmation.id)

    # The record is actually retrievable by a subsequent query.
    hits = search_embeddings([0.1, 0.2, 0.3], 5, store=store)
    assert [hit.id for hit in hits] == [confirmation.id]
    assert hits[0].metadata == {"area": "scene-1"}


def test_store_embedding_rejects_empty_embedding_with_no_record() -> None:
    """An empty embedding is a validation error and stores nothing."""
    store = InMemoryVectorStore()
    with pytest.raises(ValidationError) as exc_info:
        store_embedding([], {"k": "v"}, store=store)
    assert exc_info.value.category is ErrorCategory.VALIDATION
    assert len(store) == 0


def test_store_embedding_rejects_non_finite_embedding() -> None:
    """A non-finite component is rejected as validation, no record stored."""
    store = InMemoryVectorStore()
    with pytest.raises(ValidationError):
        store_embedding([0.1, float("nan")], {}, store=store)
    assert len(store) == 0


# --- store_embedding: no partial record on failure (Req 9.8) ---------------


class _FailingStore(VectorStoreBackend):
    """A backend whose ``add`` always raises, to exercise Req 9.8."""

    def __init__(self) -> None:
        self.deleted: list[str] = []

    def add(self, record) -> None:  # noqa: ANN001
        raise RuntimeError("backend unavailable")

    def contains(self, record_id: str) -> bool:
        return False

    def delete(self, record_id: str) -> None:
        self.deleted.append(record_id)

    def records(self):
        return []

    def __len__(self) -> int:
        return 0


class _DroppingStore(VectorStoreBackend):
    """A backend that accepts ``add`` but never retains the record (Req 9.8)."""

    def __init__(self) -> None:
        self.added: list[str] = []
        self.deleted: list[str] = []

    def add(self, record) -> None:  # noqa: ANN001
        self.added.append(record.id)  # silently dropped

    def contains(self, record_id: str) -> bool:
        return False

    def delete(self, record_id: str) -> None:
        self.deleted.append(record_id)

    def records(self):
        return []

    def __len__(self) -> int:
        return 0


def test_store_embedding_persistence_failure_leaves_no_partial_record() -> None:
    """Req 9.8: a raising backend yields a taxonomy error and rolls back."""
    store = _FailingStore()
    with pytest.raises(UpstreamError) as exc_info:
        store_embedding([1.0, 2.0], {"a": 1}, store=store)
    assert exc_info.value.category is ErrorCategory.UPSTREAM
    # Rollback attempted so no partial record can remain.
    assert store.deleted, "store_embedding must roll back a failed write"


def test_store_embedding_silent_drop_is_a_failure_with_no_record() -> None:
    """Req 9.8: a store that does not retain the record is a persistence failure."""
    store = _DroppingStore()
    with pytest.raises(UpstreamError):
        store_embedding([1.0, 2.0], {}, store=store)
    assert store.deleted, "a non-retained write must be rolled back"


# --- search_embeddings: bounds, ordering, empty store (Req 9.4) ------------


def test_search_empty_store_returns_empty() -> None:
    """Req 9.4: querying an empty store returns an empty result set."""
    store = InMemoryVectorStore()
    assert search_embeddings([1.0, 0.0], 10, store=store) == []


def test_search_orders_by_descending_similarity_and_caps_at_k() -> None:
    """Req 9.4: results are ordered by descending similarity and capped at K."""
    store = InMemoryVectorStore()
    # Query is the unit vector along x. Records ordered by closeness to it.
    near = store_embedding([1.0, 0.0], {"label": "near"}, store=store)
    mid = store_embedding([1.0, 1.0], {"label": "mid"}, store=store)
    far = store_embedding([0.0, 1.0], {"label": "far"}, store=store)

    hits = search_embeddings([1.0, 0.0], 2, store=store)
    assert len(hits) == 2  # capped at K=2
    assert [hit.id for hit in hits] == [near.id, mid.id]
    # Non-increasing similarity ordering.
    assert hits[0].similarity >= hits[1].similarity
    assert far.id not in {hit.id for hit in hits}


@pytest.mark.parametrize("bad_k", [0, -1, 1001, 5000])
def test_search_rejects_out_of_range_k(bad_k: int) -> None:
    """Req 9.4: K outside [1, 1000] is a validation error."""
    store = InMemoryVectorStore()
    store_embedding([1.0, 0.0], {}, store=store)
    with pytest.raises(ValidationError) as exc_info:
        search_embeddings([1.0, 0.0], bad_k, store=store)
    assert exc_info.value.category is ErrorCategory.VALIDATION


def test_search_never_returns_more_than_stored() -> None:
    """At most ``min(k, stored)`` hits are returned."""
    store = InMemoryVectorStore()
    store_embedding([1.0, 0.0], {}, store=store)
    store_embedding([0.0, 1.0], {}, store=store)
    hits = search_embeddings([1.0, 1.0], 1000, store=store)
    assert len(hits) == 2


# --- cosine_similarity ------------------------------------------------------


def test_cosine_identical_vectors_is_one() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_cosine_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_zero_vector_is_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# --- server tool wiring -----------------------------------------------------


def test_server_registers_both_tools() -> None:
    server = GeoEmbeddingSearchServer()
    assert set(server.tool_names()) == {"store_embedding", "search_embeddings"}


async def test_server_store_then_search_round_trip() -> None:
    """The server tools round-trip a record through the configured store."""
    server = GeoEmbeddingSearchServer()
    confirmation = await server.store_embedding(
        embedding=[0.5, 0.5, 0.5], metadata={"scene": "abc"}
    )
    assert confirmation.retrievable is True

    hits = await server.search_embeddings(query=[0.5, 0.5, 0.5], k=3)
    assert len(hits) == 1
    assert isinstance(hits[0], VectorSearchHit)
    assert hits[0].id == confirmation.id
    assert hits[0].metadata == {"scene": "abc"}
    assert hits[0].similarity == pytest.approx(1.0)

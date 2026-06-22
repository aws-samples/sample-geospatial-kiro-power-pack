"""Property test for the embedding store-then-retrieve round-trip (task 15.3).

Feature: geospatial-power-pack
Property 13: Embedding store-then-retrieve round-trip

*For any* embedding and its metadata, storing the record and then querying for
it returns the record with its metadata intact, and a failed persistence stores
no partial record.

**Validates: Requirements 9.3, 9.8**

The repository-wide Hypothesis profile (see root ``conftest.py``) enforces a
minimum of 100 generated cases per property, so these tests inherit that floor
without repeating ``@settings(max_examples=...)``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from hypothesis import given
from hypothesis import strategies as st

from geo_common.errors import ErrorCategory, UpstreamError

from geo_embedding_search.models import StoreConfirmation
from geo_embedding_search.store import (
    InMemoryVectorStore,
    VectorStoreBackend,
    search_embeddings,
    store_embedding,
)


# --- Generators -------------------------------------------------------------
# Finite, bounded floats keep cosine-norm arithmetic free of overflow while
# still exercising the full sign/magnitude space of an embedding component.
_finite_floats = st.floats(
    min_value=-1.0e6,
    max_value=1.0e6,
    allow_nan=False,
    allow_infinity=False,
)

# A valid embedding for store_embedding: non-empty, all finite.
_embeddings = st.lists(_finite_floats, min_size=1, max_size=16)

# JSON-ish metadata values whose equality is stable (no NaN), so an intact
# round-trip can be asserted with ``==``.
_metadata_values = st.one_of(
    st.text(max_size=20),
    st.integers(min_value=-1_000_000, max_value=1_000_000),
    st.booleans(),
    st.none(),
    _finite_floats,
)
_metadata = st.dictionaries(st.text(max_size=10), _metadata_values, max_size=6)


# --- Property 13a: store-then-retrieve returns the record intact (Req 9.3) --
@given(embedding=_embeddings, metadata=_metadata)
def test_stored_record_is_returned_intact_on_query(
    embedding: List[float], metadata: Dict[str, Any]
) -> None:
    """Req 9.3: a stored record round-trips intact through a subsequent query.

    Storing an embedding plus its metadata and then querying for that same
    embedding returns the stored record with its embedding and metadata intact.
    """
    store = InMemoryVectorStore()
    confirmation = store_embedding(embedding, metadata, store=store)

    assert isinstance(confirmation, StoreConfirmation)
    assert confirmation.retrievable is True
    assert confirmation.dimension == len(embedding)

    # Query for the record just stored; k=1 suffices since it is the only one.
    hits = search_embeddings(embedding, 1, store=store)

    assert len(hits) == 1
    hit = hits[0]
    assert hit.id == confirmation.id
    # The record round-trips intact: embedding and metadata are unchanged.
    assert hit.embedding == [float(v) for v in embedding]
    assert hit.metadata == metadata


@given(
    target_embedding=_embeddings,
    target_metadata=_metadata,
    others=st.lists(st.tuples(_embeddings, _metadata), max_size=8),
)
def test_target_record_retrievable_among_other_records(
    target_embedding: List[float],
    target_metadata: Dict[str, Any],
    others: List[tuple],
) -> None:
    """Req 9.3: a target record stays retrievable intact among other records.

    With several other embeddings also stored, the target record remains
    discoverable by id and its embedding/metadata are returned unchanged.
    """
    store = InMemoryVectorStore()
    for other_embedding, other_metadata in others:
        store_embedding(other_embedding, other_metadata, store=store)

    confirmation = store_embedding(target_embedding, target_metadata, store=store)

    # Search the whole store; k is capped at the legal maximum (1000).
    k = min(len(store), 1000)
    hits = search_embeddings(target_embedding, k, store=store)

    matching = [hit for hit in hits if hit.id == confirmation.id]
    assert len(matching) == 1, "stored record must be retrievable by its id"
    hit = matching[0]
    assert hit.embedding == [float(v) for v in target_embedding]
    assert hit.metadata == target_metadata


# --- Property 13b: a failed persistence stores no partial record (Req 9.8) --
class _FailingStore(VectorStoreBackend):
    """A backend whose ``add`` always raises and that retains nothing.

    Tracks attempted writes and rollback deletes so the test can assert that no
    partial record survives a persistence failure (Requirement 9.8).
    """

    def __init__(self) -> None:
        self.records_map: Dict[str, Any] = {}
        self.deleted: List[str] = []

    def add(self, record) -> None:  # noqa: ANN001
        # Fail before retaining anything: a persistence error must leave nothing.
        raise RuntimeError("backend unavailable")

    def contains(self, record_id: str) -> bool:
        return record_id in self.records_map

    def delete(self, record_id: str) -> None:
        self.deleted.append(record_id)
        self.records_map.pop(record_id, None)

    def records(self) -> List[Any]:
        return list(self.records_map.values())

    def __len__(self) -> int:
        return len(self.records_map)


@given(embedding=_embeddings, metadata=_metadata)
def test_failed_persistence_stores_no_partial_record(
    embedding: List[float], metadata: Dict[str, Any]
) -> None:
    """Req 9.8: a persistence failure raises a taxonomy error and stores nothing.

    When the configured store fails to persist, ``store_embedding`` raises an
    ``Error_Taxonomy`` error and no partial record remains in the store.
    """
    store = _FailingStore()

    with pytest.raises(UpstreamError) as exc_info:
        store_embedding(embedding, metadata, store=store)

    assert exc_info.value.category is ErrorCategory.UPSTREAM
    # No partial record: the store holds nothing and is not queryable for it.
    assert len(store) == 0
    assert store.records() == []
    assert search_embeddings(embedding, 1, store=store) == []

"""Tests for the managed OpenSearch vector store (``[opensearch]`` extra).

These use an injected in-memory fake client that mimics the ``opensearch-py``
surface the backend touches, so they run with no cluster and no network. They
validate add/contains/delete and that ``native_search`` re-scores candidates
with the shared cosine metric and returns non-increasing similarity. A live
check against a real managed cluster is gated separately by
``RUN_LIVE_OPENSEARCH=1`` (see ``test_opensearch_live.py``).
"""

from __future__ import annotations

import pytest

from geo_common.errors import ValidationError

from geo_embedding_search.opensearch_store import OpenSearchVectorStore
from geo_embedding_search.store import search_embeddings, store_embedding


class _FakeIndices:
    def __init__(self, parent: "_FakeClient") -> None:
        self._parent = parent

    def exists(self, index: str) -> bool:
        return index in self._parent.indices_created

    def create(self, index: str, body=None) -> None:
        self._parent.indices_created.add(index)
        self._parent.docs.setdefault(index, {})


class _FakeClient:
    """Minimal in-memory stand-in for an ``opensearch-py`` client."""

    def __init__(self) -> None:
        self.docs: dict = {}
        self.indices_created: set = set()
        self.indices = _FakeIndices(self)

    def index(self, index: str, id: str, body: dict, refresh: bool = False) -> None:
        self.docs.setdefault(index, {})[id] = body

    def exists(self, index: str, id: str) -> bool:
        return id in self.docs.get(index, {})

    def delete(self, index: str, id: str, ignore=None) -> None:
        self.docs.get(index, {}).pop(id, None)

    def count(self, index: str) -> dict:
        return {"count": len(self.docs.get(index, {}))}

    def search(self, index: str, body: dict) -> dict:
        # The backend re-scores with exact cosine, so returning every doc as a
        # candidate is a faithful (if unindexed) stand-in for k-NN retrieval.
        hits = [
            {"_id": doc_id, "_source": source}
            for doc_id, source in self.docs.get(index, {}).items()
        ]
        return {"hits": {"hits": hits}}


def _store() -> OpenSearchVectorStore:
    return OpenSearchVectorStore(client=_FakeClient(), index="test-embeddings")


def test_store_and_search_round_trip() -> None:
    store = _store()
    store_embedding([1.0, 0.0, 0.0], {"label": "east"}, store=store, record_id="e")
    store_embedding([0.0, 1.0, 0.0], {"label": "north"}, store=store, record_id="n")

    hits = search_embeddings([1.0, 0.0, 0.0], k=2, store=store)
    assert [h.id for h in hits] == ["e", "n"]
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-9)
    assert hits[0].similarity >= hits[1].similarity
    assert hits[0].metadata["label"] == "east"


def test_contains_delete_and_len() -> None:
    store = _store()
    store_embedding([0.1, 0.2], {}, store=store, record_id="a")
    assert store.contains("a") is True
    assert len(store) == 1
    store.delete("a")
    assert store.contains("a") is False
    assert len(store) == 0


def test_dimension_mismatch_is_rejected() -> None:
    store = _store()
    store_embedding([1.0, 2.0, 3.0], store=store, record_id="x")
    with pytest.raises(ValidationError):
        store_embedding([1.0, 2.0], store=store, record_id="y")


def test_top_k_truncates() -> None:
    store = _store()
    for i in range(5):
        vec = [0.0, 0.0, 0.0]
        vec[i % 3] = 1.0
        store_embedding(vec, {"i": i}, store=store, record_id="r%d" % i)
    hits = search_embeddings([1.0, 0.0, 0.0], k=2, store=store)
    assert len(hits) == 2
    assert hits[0].similarity >= hits[1].similarity


def test_missing_url_and_client_raises() -> None:
    with pytest.raises(ValidationError):
        OpenSearchVectorStore()  # no client, no url

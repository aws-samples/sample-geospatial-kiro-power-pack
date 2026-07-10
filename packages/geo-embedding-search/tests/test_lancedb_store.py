"""Tests for the LanceDB-backed vector store (``[lancedb]`` extra).

Skipped unless ``lancedb`` is importable. These exercise the real embedded
engine against a temporary on-disk URI (no server), so they are hermetic and
deterministic while validating genuine COPC-free vector persistence + search.
"""

from __future__ import annotations

import pytest

pytest.importorskip("lancedb")
pytest.importorskip("pyarrow")

from geo_common.errors import ValidationError

from geo_embedding_search.lancedb_store import LanceDBVectorStore
from geo_embedding_search.store import search_embeddings, store_embedding


def _store(tmp_path) -> LanceDBVectorStore:
    return LanceDBVectorStore(uri=str(tmp_path / "lancedb"))


def test_store_and_search_round_trip(tmp_path) -> None:
    store = _store(tmp_path)
    conf = store_embedding([1.0, 0.0, 0.0], {"label": "east"}, store=store)
    assert conf.retrievable and conf.dimension == 3
    store_embedding([0.0, 1.0, 0.0], {"label": "north"}, store=store)

    hits = search_embeddings([1.0, 0.0, 0.0], k=2, store=store)
    assert len(hits) == 2
    # Nearest is the identical vector, at (clamped) cosine ~1.0.
    assert hits[0].metadata["label"] == "east"
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-6)
    assert hits[0].similarity >= hits[1].similarity


def test_contains_delete_and_len(tmp_path) -> None:
    store = _store(tmp_path)
    conf = store_embedding([0.1, 0.2, 0.3], {"k": "v"}, store=store, record_id="rec-1")
    assert conf.id == "rec-1"
    assert store.contains("rec-1") is True
    assert len(store) == 1
    store.delete("rec-1")
    assert store.contains("rec-1") is False
    assert len(store) == 0


def test_empty_store_search_is_empty(tmp_path) -> None:
    store = _store(tmp_path)
    assert search_embeddings([1.0, 2.0], k=5, store=store) == []


def test_dimension_mismatch_is_rejected(tmp_path) -> None:
    store = _store(tmp_path)
    store_embedding([1.0, 2.0, 3.0], store=store)
    with pytest.raises(ValidationError):
        store_embedding([1.0, 2.0], store=store)  # wrong dimension


def test_upsert_same_id_replaces(tmp_path) -> None:
    store = _store(tmp_path)
    store_embedding([1.0, 0.0], {"v": 1}, store=store, record_id="dup")
    store_embedding([0.0, 1.0], {"v": 2}, store=store, record_id="dup")
    assert len(store) == 1
    hits = search_embeddings([0.0, 1.0], k=1, store=store)
    assert hits[0].metadata["v"] == 2


def test_persists_across_reopen(tmp_path) -> None:
    uri = str(tmp_path / "lancedb")
    first = LanceDBVectorStore(uri=uri)
    store_embedding([1.0, 0.0, 0.0], {"label": "keep"}, store=first, record_id="p1")
    # A fresh handle to the same URI sees the durably-written record.
    reopened = LanceDBVectorStore(uri=uri)
    assert reopened.contains("p1") is True
    hits = search_embeddings([1.0, 0.0, 0.0], k=1, store=reopened)
    assert hits and hits[0].metadata["label"] == "keep"

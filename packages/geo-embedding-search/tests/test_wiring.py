"""Tests for backend selection in :func:`default_vector_store`."""

from __future__ import annotations

import pytest

from geo_embedding_search.store import InMemoryVectorStore
from geo_embedding_search.wiring import default_vector_store


def test_default_is_in_memory(monkeypatch) -> None:
    monkeypatch.delenv("OPENSEARCH_URL", raising=False)
    monkeypatch.delenv("LANCEDB_URI", raising=False)
    assert isinstance(default_vector_store(), InMemoryVectorStore)


def test_lancedb_uri_selects_lancedb(monkeypatch, tmp_path) -> None:
    pytest.importorskip("lancedb")
    from geo_embedding_search.lancedb_store import LanceDBVectorStore

    monkeypatch.delenv("OPENSEARCH_URL", raising=False)
    monkeypatch.setenv("LANCEDB_URI", str(tmp_path / "lancedb"))
    assert isinstance(default_vector_store(), LanceDBVectorStore)


def test_opensearch_url_selects_opensearch(monkeypatch) -> None:
    pytest.importorskip("opensearchpy")
    from geo_embedding_search.opensearch_store import OpenSearchVectorStore

    # Constructing the client is offline (no connection until a call is made).
    monkeypatch.setenv("OPENSEARCH_URL", "https://opensearch.example.invalid")
    monkeypatch.delenv("LANCEDB_URI", raising=False)
    assert isinstance(default_vector_store(), OpenSearchVectorStore)

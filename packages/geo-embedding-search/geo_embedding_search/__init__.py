"""geo-embedding-search: Pillar C (expansion) GeoAI MCP server.

Exposes embedding storage and similarity search over a configured vector store
(OpenSearch / LanceDB in production; a deterministic in-memory store by
default). Task 15.1 delivers ``store_embedding`` (persist an embedding plus
metadata with retrievability confirmation and no partial record on failure -
Req 9.3, 9.8) and ``search_embeddings`` (return up to K stored embeddings by
descending similarity, empty when none are stored - Req 9.4), together with the
storage data models. Task 15.4 registers the server's Resource Catalog entries
(one per capability - Req 2.1, 11.3) and its Optional OpenSearch ``mcp.json``
credential specs (Req 16.1).
"""

from __future__ import annotations

from geo_embedding_search.models import (
    StoreConfirmation,
    StoredRecord,
    VectorSearchHit,
)
from geo_embedding_search.store import (
    MAX_K,
    MIN_K,
    InMemoryVectorStore,
    VectorStoreBackend,
    cosine_similarity,
    search_embeddings,
    store_embedding,
)
from geo_embedding_search.server import GeoEmbeddingSearchServer

__all__ = [
    # Data models (task 15.1)
    "StoredRecord",
    "StoreConfirmation",
    "VectorSearchHit",
    # Store core
    "VectorStoreBackend",
    "InMemoryVectorStore",
    "cosine_similarity",
    "store_embedding",
    "search_embeddings",
    "MIN_K",
    "MAX_K",
    # Server
    "GeoEmbeddingSearchServer",
]

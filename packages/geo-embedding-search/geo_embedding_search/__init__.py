"""geo-embedding-search: Pillar C (expansion) GeoAI MCP server.

Exposes embedding storage and similarity search over a configured vector store:
a deterministic in-memory store by default, the local durable
:class:`LanceDBVectorStore` (``[lancedb]`` extra) when ``LANCEDB_URI`` is set, or
a managed/hosted :class:`OpenSearchVectorStore` (``[opensearch]`` extra) when
``OPENSEARCH_URL`` is set. ``store_embedding`` persists an embedding plus
metadata with retrievability confirmation and no partial record on failure
(Req 9.3, 9.8); ``search_embeddings`` returns up to K stored embeddings by
descending similarity, empty when none are stored (Req 9.4). The server also
registers its Resource Catalog entries (one per capability - Req 2.1, 11.3) and
its Optional OpenSearch ``mcp.json`` credential specs (Req 16.1).
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
from geo_embedding_search.lancedb_store import LanceDBVectorStore
from geo_embedding_search.opensearch_store import OpenSearchVectorStore
from geo_embedding_search.wiring import default_vector_store
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
    # Optional real backends + wiring
    "LanceDBVectorStore",
    "OpenSearchVectorStore",
    "default_vector_store",
    # Server
    "GeoEmbeddingSearchServer",
]

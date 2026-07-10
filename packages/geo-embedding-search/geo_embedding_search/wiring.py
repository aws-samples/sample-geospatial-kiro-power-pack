"""Select the configured vector store from the environment.

:func:`default_vector_store` is what the server's ``main()`` uses to choose a
backend, mirroring how the rest of the pack activates optional capabilities by
configuration (e.g. ``geo-query`` wiring Athena from AWS creds):

* ``OPENSEARCH_URL`` set  -> the managed/hosted
  :class:`~geo_embedding_search.opensearch_store.OpenSearchVectorStore` (cloud
  scale; requires the ``[opensearch]`` extra).
* else ``LANCEDB_URI`` set -> the local, durable
  :class:`~geo_embedding_search.lancedb_store.LanceDBVectorStore` (requires the
  ``[lancedb]`` extra).
* else -> the zero-config
  :class:`~geo_embedding_search.store.InMemoryVectorStore` (process-local,
  non-durable) so the server always starts with nothing to configure.

A backend selected by its env var but missing its extra raises a clear
``Error_Taxonomy`` error naming the extra (from the backend constructor).
"""

from __future__ import annotations

import os

from geo_embedding_search.store import InMemoryVectorStore, VectorStoreBackend

__all__ = ["default_vector_store"]


def default_vector_store() -> VectorStoreBackend:
    """Return the vector store configured via the environment (see module docs)."""
    url = os.environ.get("OPENSEARCH_URL")
    if url:
        from geo_embedding_search.opensearch_store import OpenSearchVectorStore

        return OpenSearchVectorStore(
            url=url,
            username=os.environ.get("OPENSEARCH_USERNAME"),
            password=os.environ.get("OPENSEARCH_PASSWORD"),
        )

    lancedb_uri = os.environ.get("LANCEDB_URI")
    if lancedb_uri:
        from geo_embedding_search.lancedb_store import LanceDBVectorStore

        return LanceDBVectorStore(uri=lancedb_uri)

    return InMemoryVectorStore()

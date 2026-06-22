"""The ``geo-embedding-search`` MCP server (Pillar C, expansion).

This module wires the embedding store/search core
(:mod:`geo_embedding_search.store`) into the shared
:class:`~geo_common.server.BaseGeoServer` contract and exposes
``store_embedding`` and ``search_embeddings`` as MCP tools.

The server owns one configured vector store (an
:class:`~geo_embedding_search.store.InMemoryVectorStore` by default; OpenSearch /
LanceDB backends substitute without changing the tool contract). It inherits
the taxonomy error mapping from :class:`~geo_common.server.BaseGeoServer`, so a
persistence or query failure maps onto exactly one ``Error_Taxonomy`` category
(Req 9.8, 11.5).

Task 15.4 registers the server's Resource Catalog entries (one per capability -
Req 2.1, 11.3) and its ``mcp.json`` credential specs (Req 16.1). The three
OpenSearch keys are all :attr:`CredentialClassification.OPTIONAL`: the default
LanceDB-style local store works without them, so an absent value never blocks
startup (Req 16.5). The bundled vector stores (OpenSearch, LanceDB) are openly
licensed, so every catalog entry's tier is :attr:`OpennessTier.OPEN`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from geo_common.models import (
    CatalogEntry,
    CredentialClassification,
    CredentialSpec,
    OpennessTier,
)
from geo_common.server import BaseGeoServer

from geo_embedding_search.models import StoreConfirmation, VectorSearchHit
from geo_embedding_search.store import (
    InMemoryVectorStore,
    VectorStoreBackend,
    search_embeddings as _search_embeddings,
    store_embedding as _store_embedding,
)

__all__ = ["GeoEmbeddingSearchServer", "main"]

#: ``uvx`` command that installs this server (Req 2.6, 6.2; bundle-manifest).
INSTALL_COMMAND = "uvx geo-embedding-search"

#: Optional OpenSearch connection keys (Req 16.1; bundle-manifest). The default
#: local LanceDB-style store works without any of them, so each is
#: :attr:`CredentialClassification.OPTIONAL` and never blocks startup (Req 16.5).
OPENSEARCH_URL_KEY = "OPENSEARCH_URL"
OPENSEARCH_USERNAME_KEY = "OPENSEARCH_USERNAME"
OPENSEARCH_PASSWORD_KEY = "OPENSEARCH_PASSWORD"
OPENSEARCH_SOURCE = "OpenSearch"


class GeoEmbeddingSearchServer(BaseGeoServer):
    """Pillar C (expansion) server for embedding storage + similarity search.

    Holds one configured vector store and registers ``store_embedding`` and
    ``search_embeddings`` as MCP tools. ``store_embedding`` persists an
    embedding plus metadata and confirms retrievability with no partial record
    on failure (Req 9.3, 9.8); ``search_embeddings`` returns up to K
    (``1``-``1000``) stored embeddings ordered by descending similarity, and an
    empty result when nothing is stored (Req 9.4).
    """

    pillar = "C"
    server_name = "geo-embedding-search"
    version = "0.2.0"

    def __init__(
        self,
        *,
        store: Optional[VectorStoreBackend] = None,
        http=None,
    ) -> None:
        super().__init__(http=http)
        self.store: VectorStoreBackend = store if store is not None else InMemoryVectorStore()
        self.register_tool("store_embedding", self.store_embedding)
        self.register_tool("search_embeddings", self.search_embeddings)

    # ------------------------------------------------------------------
    # Catalog + credential declaration (Req 2.1, 11.3, 16.1)
    # ------------------------------------------------------------------

    def catalog_entries(self) -> "List[CatalogEntry]":
        """The Pillar C capabilities this server registers (Req 2.1, 11.3).

        One :class:`~geo_common.models.CatalogEntry` per exposed tool -
        ``store_embedding`` and ``search_embeddings`` - each naming this server
        as ``provider_server`` (Req 11.3) and recording the entry name, pillar,
        capability description, and ``Openness_Tier`` required by the Resource
        Catalog (Req 2.1). The bundled vector stores (OpenSearch, LanceDB) are
        openly licensed, so each entry's tier is :attr:`OpennessTier.OPEN`.
        """
        provider = self.server_name
        return [
            CatalogEntry(
                name="store_embedding",
                pillar=self.pillar,
                capability_description=(
                    "Persist an embedding plus metadata to the configured "
                    "vector store (OpenSearch / LanceDB), confirming the record "
                    "is retrievable and leaving no partial record on failure."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=provider,
                installed=True,
            ),
            CatalogEntry(
                name="search_embeddings",
                pillar=self.pillar,
                capability_description=(
                    "Semantic/similarity search returning up to K (1-1000) "
                    "stored embeddings ordered by descending similarity, or an "
                    "empty result when none are stored."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=provider,
                installed=True,
            ),
        ]

    def required_credentials(self) -> "List[CredentialSpec]":
        """The ``mcp.json`` keys this server reads, with classification (Req 16.1).

        Declares the three Optional OpenSearch connection keys
        (``OPENSEARCH_URL``, ``OPENSEARCH_USERNAME``, ``OPENSEARCH_PASSWORD``).
        Because each is :attr:`CredentialClassification.OPTIONAL`, an absent
        value never blocks startup (Req 16.5): the default local LanceDB-style
        store works without any OpenSearch configuration.
        """
        return [
            CredentialSpec(
                source=OPENSEARCH_SOURCE,
                mcp_json_key=OPENSEARCH_URL_KEY,
                classification=CredentialClassification.OPTIONAL,
            ),
            CredentialSpec(
                source=OPENSEARCH_SOURCE,
                mcp_json_key=OPENSEARCH_USERNAME_KEY,
                classification=CredentialClassification.OPTIONAL,
            ),
            CredentialSpec(
                source=OPENSEARCH_SOURCE,
                mcp_json_key=OPENSEARCH_PASSWORD_KEY,
                classification=CredentialClassification.OPTIONAL,
            ),
        ]

    async def store_embedding(
        self, *, embedding: List[float], metadata: Optional[Dict[str, Any]] = None
    ) -> StoreConfirmation:
        """Persist an embedding + metadata, confirming retrievability (Req 9.3/9.8).

        Validates the embedding before writing, persists it to the configured
        store, and returns a :class:`StoreConfirmation` only once the store
        reports the record retrievable by subsequent queries. A persistence
        failure raises an ``Error_Taxonomy`` error and leaves no partial record
        (Requirement 9.8).
        """
        return _store_embedding(embedding, metadata, store=self.store)

    async def search_embeddings(
        self, *, query: List[float], k: int
    ) -> "List[VectorSearchHit]":
        """Return up to ``k`` stored embeddings by descending similarity (Req 9.4).

        ``k`` must be between 1 and 1,000 (otherwise an ``Error_Taxonomy``
        validation error). Returns at most ``k`` hits ordered by non-increasing
        similarity to ``query``, or an empty list when nothing is stored.
        """
        return _search_embeddings(query, k, store=self.store)


def main() -> None:
    """Console entry point: serve geo-embedding-search over MCP stdio.

    Runs the startup credential guard, then serves this server's registered
    tools over stdin/stdout via the shared geo-common MCP runtime until the
    client disconnects.
    """
    GeoEmbeddingSearchServer().run()

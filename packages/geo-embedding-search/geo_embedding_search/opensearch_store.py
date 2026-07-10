"""OpenSearch-backed vector store (the managed/hosted ``[opensearch]`` backend).

:class:`OpenSearchVectorStore` is the **cloud-scale** backend for
``geo-embedding-search``. It is a thin *client* for a **managed/hosted**
OpenSearch (e.g. Amazon OpenSearch Service) reached over HTTPS - the pack never
runs or bundles the OpenSearch engine locally (a local cluster would be far too
heavy for a ``uvx``-light install and would violate the pack's "bring compute to
the data / delegate heavy jobs to the cloud" principle). The weight lives in the
managed service; only the light ``opensearch-py`` client ships, behind the
``[opensearch]`` extra, and it is configured through the single ``mcp.json``
credential surface (``OPENSEARCH_URL`` / ``OPENSEARCH_USERNAME`` /
``OPENSEARCH_PASSWORD``).

It implements the :class:`~geo_embedding_search.store.VectorStoreBackend`
contract and overrides
:meth:`~geo_embedding_search.store.VectorStoreBackend.native_search` to run a
k-NN query against the service. To keep the similarity metric identical to the
rest of the pack, the returned candidate vectors are re-scored with the shared
:func:`~geo_embedding_search.store.cosine_similarity` (rather than trusting the
engine's version-dependent k-NN ``_score`` scale) and ordered by non-increasing
cosine similarity.

Like any real vector index, the index fixes the embedding **dimensionality**;
a later embedding of a different length is rejected with a
:class:`~geo_common.errors.ValidationError`.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from geo_common.errors import UpstreamError, ValidationError

from geo_embedding_search.models import StoredRecord, VectorSearchHit
from geo_embedding_search.store import cosine_similarity

__all__ = ["OpenSearchVectorStore", "DEFAULT_INDEX_NAME"]

_SOURCE = "geo-embedding-search"

#: Default OpenSearch index holding the embedding documents.
DEFAULT_INDEX_NAME = "geo-embeddings"

#: How many k-NN candidates to fetch before exact cosine re-scoring. A small
#: multiple of ``k`` keeps recall high while the re-score stays cheap.
_CANDIDATE_MULTIPLIER = 4


class OpenSearchVectorStore:
    """A client for a managed OpenSearch vector index (``[opensearch]`` extra).

    Persists each :class:`StoredRecord` as a document ``{vector, metadata}`` and
    searches via a k-NN query re-scored with the shared cosine metric. The
    index's vector dimension is fixed on first write.
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        index: str = DEFAULT_INDEX_NAME,
        username: Optional[str] = None,
        password: Optional[str] = None,
        dimension: Optional[int] = None,
        client: Any = None,
        verify_certs: bool = True,
    ) -> None:
        self._index = index
        self._dimension = dimension
        if client is not None:
            # Injected client (used by unit tests with a fake).
            self._client = client
        else:
            if not url:
                raise ValidationError(
                    "OpenSearch backend requires a URL (set OPENSEARCH_URL)",
                    source=_SOURCE,
                    detail={"parameter": "url"},
                )
            try:
                from opensearchpy import OpenSearch
            except ModuleNotFoundError as exc:  # pragma: no cover - import guard
                raise UpstreamError(
                    "the OpenSearch backend requires the 'opensearch' extra: "
                    "pip install 'geo-embedding-search[opensearch]'",
                    source=_SOURCE,
                    original=str(exc),
                ) from exc
            http_auth = (username, password) if username and password else None
            self._client = OpenSearch(
                hosts=[url], http_auth=http_auth, verify_certs=verify_certs
            )

    # -- helpers ---------------------------------------------------------

    def _check_dimension(self, dim: int) -> None:
        if self._dimension is not None and dim != self._dimension:
            raise ValidationError(
                "embedding dimension %d does not match the store dimension %d; "
                "an OpenSearch-backed store requires a consistent dimension"
                % (dim, self._dimension),
                source=_SOURCE,
                detail={"dimension": dim, "expected": self._dimension},
            )

    def _ensure_index(self, dim: int) -> None:
        if self._client.indices.exists(index=self._index):
            if self._dimension is None:
                self._dimension = dim
            return
        body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene",
                        },
                    },
                    "metadata": {"type": "object", "enabled": True},
                }
            },
        }
        self._client.indices.create(index=self._index, body=body)
        self._dimension = dim

    # -- VectorStoreBackend contract ------------------------------------

    def add(self, record: StoredRecord) -> None:
        dim = len(record.embedding)
        self._check_dimension(dim)
        self._ensure_index(dim)
        self._client.index(
            index=self._index,
            id=record.id,
            body={
                "vector": [float(v) for v in record.embedding],
                "metadata": dict(record.metadata or {}),
            },
            refresh=True,
        )

    def contains(self, record_id: str) -> bool:
        return bool(self._client.exists(index=self._index, id=record_id))

    def delete(self, record_id: str) -> None:
        self._client.delete(
            index=self._index, id=record_id, ignore=[404]
        )

    def records(self) -> "List[StoredRecord]":
        """Enumerate documents (used only by the portable fallback path)."""
        try:
            resp = self._client.search(
                index=self._index,
                body={"query": {"match_all": {}}, "size": 1000},
            )
        except Exception:  # pragma: no cover - index may not exist yet
            return []
        return [_hit_to_record(h) for h in _hits(resp)]

    def native_search(
        self, query: Sequence[float], k: int
    ) -> "Optional[List[VectorSearchHit]]":
        vector = [float(v) for v in query]
        self._check_dimension(len(vector))
        candidates = max(k * _CANDIDATE_MULTIPLIER, k)
        try:
            resp = self._client.search(
                index=self._index,
                body={
                    "size": candidates,
                    "query": {"knn": {"vector": {"vector": vector, "k": candidates}}},
                },
            )
        except Exception as exc:  # noqa: BLE001 - map engine/transport errors
            raise UpstreamError(
                "OpenSearch k-NN query failed",
                source=_SOURCE,
                original=str(exc),
            ) from exc
        # Re-score with the shared cosine metric so ordering matches the rest of
        # the pack regardless of the engine's k-NN score scale, then take top-k.
        scored = []
        for hit in _hits(resp):
            record = _hit_to_record(hit)
            scored.append((cosine_similarity(vector, record.embedding), record))
        scored.sort(key=lambda item: -item[0])
        return [
            VectorSearchHit(
                id=record.id,
                similarity=score,
                embedding=list(record.embedding),
                metadata=dict(record.metadata),
            )
            for score, record in scored[:k]
        ]

    def __len__(self) -> int:
        try:
            resp = self._client.count(index=self._index)
        except Exception:  # pragma: no cover - index may not exist yet
            return 0
        return int(resp.get("count", 0))


def _hits(response: Any) -> "List[dict]":
    """Extract the hit list from an OpenSearch search response."""
    if not isinstance(response, dict):
        return []
    outer = response.get("hits")
    if not isinstance(outer, dict):
        return []
    inner = outer.get("hits")
    return inner if isinstance(inner, list) else []


def _hit_to_record(hit: dict) -> StoredRecord:
    """Rebuild a :class:`StoredRecord` from an OpenSearch hit."""
    source = hit.get("_source") if isinstance(hit, dict) else {}
    source = source if isinstance(source, dict) else {}
    vector = source.get("vector") or []
    metadata = source.get("metadata")
    return StoredRecord(
        id=str(hit.get("_id")),
        embedding=[float(v) for v in vector],
        metadata=metadata if isinstance(metadata, dict) else {},
    )

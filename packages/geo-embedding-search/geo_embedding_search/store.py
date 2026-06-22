"""Embedding storage and similarity search core (Req 9.3, 9.4, 9.8).

This module holds the logic-bearing core of task 15.1, kept free of any
MCP plumbing so it is easy to test and reuse:

* :class:`VectorStoreBackend` - the pluggable "configured vector store"
  interface (design.md names OpenSearch and LanceDB as production backends).
  A backend only has to persist, look up, and enumerate
  :class:`~geo_embedding_search.models.StoredRecord` values.
* :class:`InMemoryVectorStore` - the default backend: a deterministic,
  dependency-free store used for local runs and tests. Production backends are
  substituted without changing the tool contract below.
* :func:`store_embedding` - validates an embedding, persists it plus its
  metadata to the configured store, and returns a
  :class:`~geo_embedding_search.models.StoreConfirmation` confirming the record is
  retrievable by subsequent queries (Requirement 9.3). If persistence fails it
  raises an ``Error_Taxonomy`` error and leaves **no** partial record
  (Requirement 9.8).
* :func:`search_embeddings` - returns up to ``k`` (``1``-``1000``) stored
  records ordered by non-increasing similarity to a query embedding, and an
  empty list when nothing is stored (Requirement 9.4, Property 11).

Cosine similarity is used as the similarity score. It is made **total** (well
defined for any pair of finite vectors, including differing dimensionalities)
by computing the dot product over the overlapping components and dividing by
the full vector norms, so search produces a deterministic non-increasing
ordering for *any* query against *any* stored set (Property 11). Two identical
vectors score ``1.0`` - the maximum - so a stored record round-trips to the top
of its own query (Property 13).
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Dict, List, Optional, Sequence

from geo_common.errors import UpstreamError, ValidationError

from geo_embedding_search.models import (
    StoreConfirmation,
    StoredRecord,
    VectorSearchHit,
)

__all__ = [
    "VectorStoreBackend",
    "InMemoryVectorStore",
    "cosine_similarity",
    "store_embedding",
    "search_embeddings",
    "MIN_K",
    "MAX_K",
]

#: Source identifier used on errors raised by this server.
_SOURCE = "geo-embedding-search"

#: Inclusive bounds on the result count ``k`` accepted by ``search_embeddings``
#: (Requirement 9.4: "K is between 1 and 1,000").
MIN_K = 1
MAX_K = 1000


class VectorStoreBackend:
    """The configured vector store interface (Req 9.3 / 9.4).

    A backend persists :class:`StoredRecord` values keyed by ``id`` and lets the
    core enumerate them for similarity search. The default implementation is
    :class:`InMemoryVectorStore`; production deployments substitute an
    OpenSearch- or LanceDB-backed implementation without changing the
    ``store_embedding`` / ``search_embeddings`` contract.
    """

    def add(self, record: StoredRecord) -> None:  # pragma: no cover - interface
        """Persist ``record``. Must raise on failure rather than partially write."""
        raise NotImplementedError

    def contains(self, record_id: str) -> bool:  # pragma: no cover - interface
        """Return whether a record with ``record_id`` is persisted."""
        raise NotImplementedError

    def delete(self, record_id: str) -> None:  # pragma: no cover - interface
        """Remove ``record_id`` if present; a no-op when it is absent."""
        raise NotImplementedError

    def records(self) -> Sequence[StoredRecord]:  # pragma: no cover - interface
        """Return all persisted records, in a deterministic order."""
        raise NotImplementedError

    def __len__(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError


class InMemoryVectorStore(VectorStoreBackend):
    """A deterministic, dependency-free vector store (default backend).

    Keeps records in insertion order so similarity ties break deterministically
    and so :meth:`records` enumeration is stable across calls. ``add`` is
    effectively atomic (a single dict assignment), so a failure cannot leave a
    partial record; the :func:`store_embedding` flow additionally rolls back on
    any error to uphold Requirement 9.8 for backends that are not atomic.
    """

    def __init__(self) -> None:
        self._records: "Dict[str, StoredRecord]" = {}

    def add(self, record: StoredRecord) -> None:
        self._records[record.id] = record

    def contains(self, record_id: str) -> bool:
        return record_id in self._records

    def delete(self, record_id: str) -> None:
        self._records.pop(record_id, None)

    def records(self) -> "List[StoredRecord]":
        # Insertion order is preserved by dict in Python 3.7+.
        return list(self._records.values())

    def get(self, record_id: str) -> Optional[StoredRecord]:
        """Return the record for ``record_id`` or ``None`` when absent."""
        return self._records.get(record_id)

    def __len__(self) -> int:
        return len(self._records)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Total cosine similarity between two finite vectors.

    Defined for any two vectors, including differing lengths: the dot product is
    taken over the overlapping components and divided by the **full** L2 norms
    of each vector. The result lies in ``[-1.0, 1.0]``; two identical vectors
    score exactly ``1.0`` (the maximum), and a zero-norm vector scores ``0.0``
    (cosine is otherwise undefined). This totality is what lets
    :func:`search_embeddings` produce a deterministic non-increasing ordering
    for any query against any stored set (Property 11).
    """
    if a is b or list(a) == list(b):
        # Identical vectors are maximally similar; short-circuit so the score is
        # exactly 1.0 rather than a floating-point epsilon away (Property 13),
        # except the degenerate all-zero case handled below.
        norm_sq = math.fsum(x * x for x in a)
        if norm_sq > 0.0:
            return 1.0

    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = math.fsum(a[i] * b[i] for i in range(n))
    norm_a = math.sqrt(math.fsum(x * x for x in a))
    norm_b = math.sqrt(math.fsum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    similarity = dot / (norm_a * norm_b)
    # Guard against floating-point drift outside the valid cosine range.
    return max(-1.0, min(1.0, similarity))


def _validate_embedding(embedding: Sequence[float], *, field: str) -> None:
    """Reject an empty or non-finite embedding (Req 9.7-style validation).

    Raises an ``Error_Taxonomy`` :class:`~geo_common.errors.ValidationError`
    when ``embedding`` is empty or contains a non-finite value, so a bad input
    produces no stored record and no search.
    """
    if len(embedding) == 0:
        raise ValidationError(
            "empty %s: an embedding must have at least one component" % field,
            source=_SOURCE,
            detail={"field": field, "dimension": 0},
        )
    for index, value in enumerate(embedding):
        if not math.isfinite(value):
            raise ValidationError(
                "invalid %s: component %d is not a finite number" % (field, index),
                source=_SOURCE,
                detail={"field": field, "index": index},
            )


def store_embedding(
    embedding: Sequence[float],
    metadata: Optional[Dict[str, Any]] = None,
    *,
    store: VectorStoreBackend,
    record_id: Optional[str] = None,
) -> StoreConfirmation:
    """Persist an embedding + metadata and confirm it is retrievable (Req 9.3/9.8).

    Validates the embedding (non-empty, all finite) **before** touching the
    store, so malformed input yields no record. It then writes the record to the
    configured ``store`` and verifies the store reports it present; the returned
    :class:`StoreConfirmation` therefore means the record is retrievable by
    subsequent queries (Requirement 9.3).

    If the store raises while persisting, or reports the record absent after the
    write, the record is rolled back (deleted) and an ``Error_Taxonomy``
    :class:`~geo_common.errors.UpstreamError` is raised, so persistence failure
    leaves **no** partial record (Requirement 9.8).

    ``record_id`` lets a caller supply a stable id; when omitted a unique id is
    generated. ``metadata`` defaults to an empty mapping.
    """
    _validate_embedding(embedding, field="embedding")

    rid = record_id if record_id is not None else uuid.uuid4().hex
    record = StoredRecord(
        id=rid,
        embedding=[float(v) for v in embedding],
        metadata=dict(metadata or {}),
    )

    try:
        store.add(record)
        # Confirm retrievability by subsequent queries (Req 9.3): the store must
        # report the record present. A backend that silently dropped the write
        # is treated as a persistence failure below.
        retrievable = store.contains(record.id)
    except ValidationError:
        # Validation failures are already taxonomy-classified; surface as-is.
        raise
    except Exception as exc:  # noqa: BLE001 - map any backend error onto taxonomy
        _rollback(store, record.id)
        raise UpstreamError(
            "failed to persist embedding to the vector store",
            source=_SOURCE,
            detail={"id": record.id},
            original=_safe_detail(exc),
        ) from exc

    if not retrievable:
        # The store accepted the write but cannot return the record: no partial
        # record may remain (Req 9.8).
        _rollback(store, record.id)
        raise UpstreamError(
            "vector store did not retain the embedding; no record was stored",
            source=_SOURCE,
            detail={"id": record.id},
        )

    return StoreConfirmation(
        id=record.id,
        dimension=record.dimension,
        retrievable=True,
    )


def search_embeddings(
    query: Sequence[float],
    k: int,
    *,
    store: VectorStoreBackend,
) -> "List[VectorSearchHit]":
    """Return up to ``k`` stored embeddings by descending similarity (Req 9.4).

    ``k`` must be an integer in ``[1, 1000]``; anything else raises an
    ``Error_Taxonomy`` :class:`~geo_common.errors.ValidationError`. The query
    embedding must be non-empty and finite.

    Scores every stored record with :func:`cosine_similarity` against ``query``,
    orders the records by **non-increasing** similarity (ties broken by stable
    insertion order), and returns the top ``k`` as
    :class:`VectorSearchHit` values (Property 11). When the store is empty the
    result is an empty list (Requirement 9.4).
    """
    _validate_k(k)
    _validate_embedding(query, field="query")

    records = list(store.records())
    if not records:
        return []

    # Score each record; enumerate index as a stable tie-breaker so equal
    # similarities preserve insertion order deterministically.
    scored = [
        (cosine_similarity(query, record.embedding), index, record)
        for index, record in enumerate(records)
    ]
    # Sort by descending similarity, then ascending insertion index. Negating
    # the index is unnecessary because we want ascending order for ties.
    scored.sort(key=lambda item: (-item[0], item[1]))

    top = scored[:k]
    return [
        VectorSearchHit(
            id=record.id,
            similarity=score,
            embedding=list(record.embedding),
            metadata=dict(record.metadata),
        )
        for score, _index, record in top
    ]


def _validate_k(k: int) -> None:
    """Reject a result count outside ``[1, 1000]`` (Requirement 9.4)."""
    # Guard against bool (a subclass of int) and non-integers.
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValidationError(
            "result count k must be an integer between %d and %d" % (MIN_K, MAX_K),
            source=_SOURCE,
            detail={"k": k},
        )
    if k < MIN_K or k > MAX_K:
        raise ValidationError(
            "result count k=%d is out of range; k must be between %d and %d"
            % (k, MIN_K, MAX_K),
            source=_SOURCE,
            detail={"k": k, "min": MIN_K, "max": MAX_K},
        )


def _rollback(store: VectorStoreBackend, record_id: str) -> None:
    """Best-effort removal of a partially written record (Requirement 9.8)."""
    try:
        store.delete(record_id)
    except Exception:  # noqa: BLE001 - rollback must not mask the original error
        pass


def _safe_detail(exc: Exception) -> str:
    """A secret-free description of an exception for ``original`` retention."""
    text = str(exc).strip()
    return text if text else type(exc).__name__

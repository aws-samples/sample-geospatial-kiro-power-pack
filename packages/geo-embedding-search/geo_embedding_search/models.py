"""Data models for the ``geo-embedding-search`` server (Pillar C, expansion).

Task 15.1 needs three small data types that describe what the embedding store
holds and what its two tools return:

* :class:`StoredRecord` - one persisted embedding plus its metadata. It is the
  unit the configured vector store keeps and what ``store_embedding`` writes
  (Requirement 9.3). The record owns a stable ``id`` so a later query can name
  the exact record it round-trips (Property 13).
* :class:`StoreConfirmation` - the value ``store_embedding`` returns. It
  confirms the record was persisted *and is retrievable by subsequent queries*
  (Requirement 9.3): ``retrievable`` is only ``True`` once the store reports the
  record present, and a failed persistence raises rather than returning a
  confirmation, so there is never a partial record (Requirement 9.8).
* :class:`VectorSearchHit` - one result of ``search_embeddings``: the matched
  record's ``id``, its ``similarity`` score against the query, and the stored
  ``embedding`` and ``metadata`` so the hit round-trips the record intact
  (Requirement 9.4, Property 13).

Python 3.9 compatibility: this module uses ``from __future__ import
annotations`` together with ``typing`` generics so the pydantic models resolve
their annotations on 3.9+.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

__all__ = [
    "StoredRecord",
    "StoreConfirmation",
    "VectorSearchHit",
]


class StoredRecord(BaseModel):
    """One persisted embedding plus its metadata (Requirement 9.3).

    ``id`` is the stable identifier the store keys the record by, so a
    subsequent query can return *this* record and its metadata intact
    (Property 13). ``dimension`` always equals ``len(embedding)``.
    """

    id: str = Field(min_length=1)
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def dimension(self) -> int:
        """The embedding dimensionality (``len(embedding)``)."""
        return len(self.embedding)


class StoreConfirmation(BaseModel):
    """Confirmation that an embedding was persisted and is retrievable (Req 9.3).

    Returned by ``store_embedding`` only on success. ``retrievable`` is ``True``
    once the configured vector store reports the record present, so the
    confirmation means "a subsequent query can find this record". On a
    persistence failure ``store_embedding`` raises an ``Error_Taxonomy`` error
    instead of returning this model, and no partial record is left behind
    (Requirement 9.8).
    """

    id: str = Field(min_length=1)
    dimension: int = Field(ge=0)
    retrievable: bool


class VectorSearchHit(BaseModel):
    """One hit returned by ``search_embeddings`` (Requirement 9.4).

    Carries the matched record's ``id``, the ``similarity`` score against the
    query embedding (higher is more similar; results are ordered by
    non-increasing ``similarity`` - Property 11), and the stored ``embedding``
    and ``metadata`` so the hit round-trips the persisted record intact
    (Property 13).
    """

    id: str = Field(min_length=1)
    similarity: float
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

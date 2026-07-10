"""LanceDB-backed vector store (the local, durable ``[lancedb]`` backend).

:class:`LanceDBVectorStore` is the local/embedded durable backend for
``geo-embedding-search``: an on-disk vector index with no server to run, so it
fits the pack's "light, modular install" principle (the heavy engine is not a
service you operate - LanceDB is embedded). It is opt-in via the ``[lancedb]``
extra and activated by pointing ``LANCEDB_URI`` at a directory.

It implements the :class:`~geo_embedding_search.store.VectorStoreBackend`
contract and overrides :meth:`~geo_embedding_search.store.VectorStoreBackend.native_search`
to push the top-``k`` cosine search down to LanceDB's index rather than scoring
every record in Python.

Like any real vector index, LanceDB fixes the embedding **dimensionality** per
table: the dimension is set by the first stored embedding (or an explicit
``dimension``), and a later embedding of a different length is rejected with an
``Error_Taxonomy`` :class:`~geo_common.errors.ValidationError`. (The default
in-memory store tolerates mixed dimensions; a durable index cannot.)

``lancedb`` and ``pyarrow`` are imported lazily so the base package installs
without them; a missing dependency raises a clear error naming the extra.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Sequence

from geo_common.errors import UpstreamError, ValidationError

from geo_embedding_search.models import StoredRecord, VectorSearchHit

__all__ = ["LanceDBVectorStore", "DEFAULT_TABLE_NAME"]

_SOURCE = "geo-embedding-search"

#: Default LanceDB table name holding the embedding records.
DEFAULT_TABLE_NAME = "embeddings"


class LanceDBVectorStore:
    """A durable, embedded vector store backed by LanceDB (``[lancedb]`` extra).

    Persists each :class:`StoredRecord` as a row ``{id, vector, metadata_json}``
    and searches via LanceDB's cosine metric. The table's vector dimension is
    fixed on first write; mismatched dimensions are rejected.
    """

    def __init__(
        self,
        uri: str,
        *,
        table_name: str = DEFAULT_TABLE_NAME,
        dimension: Optional[int] = None,
    ) -> None:
        try:
            import lancedb  # noqa: F401  (lazy: only when this backend is used)
        except ModuleNotFoundError as exc:  # pragma: no cover - import-guard path
            raise UpstreamError(
                "the LanceDB backend requires the 'lancedb' extra: "
                "pip install 'geo-embedding-search[lancedb]'",
                source=_SOURCE,
                original=str(exc),
            ) from exc

        self._lancedb = lancedb
        self._uri = uri
        self._table_name = table_name
        self._dimension = dimension
        self._db = lancedb.connect(uri)
        self._table = None
        if table_name in self._existing_table_names():
            self._table = self._db.open_table(table_name)
            if self._dimension is None:
                self._dimension = self._infer_dimension(self._table)

    # -- helpers ---------------------------------------------------------

    def _existing_table_names(self) -> "List[str]":
        """Existing table names, tolerating LanceDB API differences across versions.

        Newer LanceDB returns a ``ListTablesResponse`` (with a ``.tables`` list)
        from ``list_tables()``; older versions expose a plain list via
        ``table_names()``. Prefer the response object's ``.tables`` when present,
        else fall back to the plain list.
        """
        lister = getattr(self._db, "list_tables", None)
        if lister is not None:
            try:
                result = lister()
            except TypeError:  # pragma: no cover - signature variance
                result = None
            if result is not None:
                tables = getattr(result, "tables", None)
                if tables is not None:
                    return list(tables)
                if isinstance(result, (list, tuple)):
                    return list(result)
        return list(self._db.table_names())

    @staticmethod
    def _infer_dimension(table: Any) -> "Optional[int]":
        """Read the fixed vector dimension off an existing table's schema."""
        try:
            field = table.schema.field("vector")
            # A fixed-size list carries its length as ``list_size``.
            return int(field.type.list_size)
        except Exception:  # pragma: no cover - defensive; schema variations
            return None

    def _check_dimension(self, dim: int) -> None:
        if self._dimension is not None and dim != self._dimension:
            raise ValidationError(
                "embedding dimension %d does not match the store dimension %d; "
                "a LanceDB-backed store requires a consistent dimension"
                % (dim, self._dimension),
                source=_SOURCE,
                detail={"dimension": dim, "expected": self._dimension},
            )

    def _ensure_table(self, dim: int) -> Any:
        """Return the table, creating it (dimension ``dim``) on first write."""
        import pyarrow as pa

        if self._table is not None:
            return self._table
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
                pa.field("metadata_json", pa.string()),
            ]
        )
        self._table = self._db.create_table(self._table_name, schema=schema)
        self._dimension = dim
        return self._table

    # -- VectorStoreBackend contract ------------------------------------

    def add(self, record: StoredRecord) -> None:
        dim = len(record.embedding)
        self._check_dimension(dim)
        table = self._ensure_table(dim)
        # Upsert: an id is unique, so drop any prior row before inserting.
        self.delete(record.id)
        table.add(
            [
                {
                    "id": record.id,
                    "vector": [float(v) for v in record.embedding],
                    "metadata_json": json.dumps(record.metadata or {}),
                }
            ]
        )

    def contains(self, record_id: str) -> bool:
        if self._table is None:
            return False
        return self._table.count_rows(filter=_id_filter(record_id)) > 0

    def delete(self, record_id: str) -> None:
        if self._table is None:
            return
        self._table.delete(_id_filter(record_id))

    def records(self) -> "List[StoredRecord]":
        """Enumerate all rows (used only by the portable fallback path)."""
        if self._table is None:
            return []
        rows = self._table.to_arrow().to_pylist()
        return [_row_to_record(row) for row in rows]

    def native_search(
        self, query: Sequence[float], k: int
    ) -> "Optional[List[VectorSearchHit]]":
        if self._table is None:
            return []
        self._check_dimension(len(query))
        results = (
            self._table.search([float(v) for v in query])
            .metric("cosine")
            .limit(k)
            .to_list()
        )
        hits: "List[VectorSearchHit]" = []
        for row in results:
            # LanceDB reports cosine *distance* in ``_distance`` (0 = identical);
            # cosine similarity is 1 - distance, clamped to the valid range.
            distance = float(row.get("_distance", 0.0))
            similarity = max(-1.0, min(1.0, 1.0 - distance))
            record = _row_to_record(row)
            hits.append(
                VectorSearchHit(
                    id=record.id,
                    similarity=similarity,
                    embedding=list(record.embedding),
                    metadata=dict(record.metadata),
                )
            )
        return hits

    def __len__(self) -> int:
        if self._table is None:
            return 0
        return int(self._table.count_rows())


def _id_filter(record_id: str) -> str:
    """A safe LanceDB SQL filter matching one id (single quotes escaped)."""
    escaped = record_id.replace("'", "''")
    return "id = '%s'" % escaped


def _row_to_record(row: dict) -> StoredRecord:
    """Rebuild a :class:`StoredRecord` from a LanceDB row mapping."""
    raw_meta = row.get("metadata_json") or "{}"
    try:
        metadata = json.loads(raw_meta)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        metadata = {}
    vector = row.get("vector") or []
    return StoredRecord(
        id=str(row.get("id")),
        embedding=[float(v) for v in vector],
        metadata=metadata if isinstance(metadata, dict) else {},
    )

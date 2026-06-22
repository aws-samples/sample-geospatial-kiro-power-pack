"""The DuckDB Spatial query engine for ``geo-query`` (open, credential-free).

:class:`DuckDBEngine` is a concrete :class:`~geo_query.engine.QueryEngine` that
runs spatial SQL in-process with DuckDB and its ``spatial`` extension. It is the
open default engine: no credential, no account, and DuckDB reads cloud-optimized
data (Parquet/GeoParquet, CSV, GeoJSON) directly from local paths or ``s3://`` /
``https://`` URLs, so queries run *over data in place* (Requirement 8.6).

DuckDB is an **optional** dependency (the ``[duckdb]`` extra). It is imported
lazily so the rest of ``geo-query`` - and its test suite, which injects a mock
engine - never requires DuckDB to be installed. :func:`default_query_engines`
wires this engine when DuckDB is importable, and the server's ``main()`` uses it
so a deployed ``geo-query`` runs DuckDB out of the box once the extra is
installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from geo_common.errors import NotFoundError, UpstreamError, ValidationError

from geo_query.engine import QueryEngine
from geo_query.models import ResultSet

__all__ = ["DuckDBEngine", "default_query_engines"]


class DuckDBEngine(QueryEngine):
    """Execute spatial SQL in-process with DuckDB + the ``spatial`` extension.

    On each query a fresh in-memory connection is opened (and closed), the
    ``spatial`` extension is loaded, any ``sources`` (logical name -> data
    location) are exposed as views over the file/URL, and the query is run. The
    column names and rows are returned as a :class:`ResultSet` attributed to
    ``duckdb``. DuckDB's ``httpfs`` extension is loaded best-effort so ``s3://``
    / ``https://`` data locations work when reachable.
    """

    name = "duckdb"

    def execute(
        self, *, query: str, sources: Optional[Mapping[str, str]] = None
    ) -> ResultSet:
        try:
            import duckdb  # noqa: PLC0415 - optional dependency, imported lazily
        except ImportError as exc:  # pragma: no cover - exercised only without duckdb
            raise UpstreamError(
                "the DuckDB engine requires the 'duckdb' package; install the "
                "geo-query[duckdb] extra to enable it",
                source=self.name,
                original=str(exc),
            ) from exc

        con = duckdb.connect(database=":memory:")
        try:
            self._load_extensions(con)
            try:
                # Registering a source view and running the query can both fail
                # on a bad source location (a missing file/URL), so classify
                # them together.
                self._register_sources(con, sources)
                cursor = con.execute(query)
            except Exception as exc:  # noqa: BLE001 - DuckDB IO/parse/bind error
                raise self._classify(duckdb, exc) from exc
            columns = [d[0] for d in (cursor.description or [])]
            rows = [list(_jsonable(v) for v in row) for row in cursor.fetchall()]
        finally:
            con.close()

        return ResultSet.from_rows(engine=self.name, columns=columns, rows=rows)

    def _classify(self, duckdb: Any, exc: Exception) -> Exception:
        """Map a DuckDB failure onto the taxonomy, preserving its message.

        A source that cannot be opened (a missing file or unreachable URL named
        in ``sources``) is a ``NOT_FOUND`` - the resource does not exist - and
        carries DuckDB's own precise message (e.g. ``IO Error: No files found``)
        rather than surfacing as a generic unmapped upstream error. Any other
        failure is treated as invalid query input (``VALIDATION``). All messages
        are the engine's own and secret-free.
        """
        message = str(exc)
        io_exc = getattr(duckdb, "IOException", None)
        is_missing_source = (
            (io_exc is not None and isinstance(exc, io_exc))
            or "No files found" in message
            or "No such file" in message
            or "IO Error" in message
        )
        if is_missing_source:
            return NotFoundError(
                "DuckDB could not open a source: %s" % message,
                source=self.name,
                detail={"engine": self.name},
            )
        return ValidationError(
            "DuckDB rejected the query: %s" % message,
            source=self.name,
            detail={"engine": self.name},
        )

    @staticmethod
    def _load_extensions(con: Any) -> None:
        """Load the ``spatial`` extension (and ``httpfs`` best-effort)."""
        con.execute("INSTALL spatial; LOAD spatial;")
        try:
            con.execute("INSTALL httpfs; LOAD httpfs;")
        except Exception:  # noqa: BLE001 - offline / no httpfs: local queries still work
            pass

    @staticmethod
    def _register_sources(con: Any, sources: Optional[Mapping[str, str]]) -> None:
        """Expose each ``name -> location`` source as a queryable view.

        DuckDB reads Parquet/CSV/GeoJSON directly from a path or URL, so each
        source becomes ``CREATE VIEW <name> AS SELECT * FROM '<location>'``.
        Identifiers/locations are quoted to avoid injection via the mapping.
        """
        if not sources:
            return
        for name, location in sources.items():
            ident = '"%s"' % str(name).replace('"', '""')
            loc = str(location).replace("'", "''")
            con.execute("CREATE OR REPLACE VIEW %s AS SELECT * FROM '%s'" % (ident, loc))


def _jsonable(value: Any) -> Any:
    """Coerce a DuckDB cell to a JSON-serializable value (best effort)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def default_query_engines() -> "Dict[str, QueryEngine]":
    """Return the auto-wired engine set: DuckDB when importable, else empty.

    Used by the server's ``main()`` so a deployed ``geo-query`` runs the open
    DuckDB engine out of the box once the ``[duckdb]`` extra is installed.
    Credentialed engines (Athena) remain inject-only.
    """
    try:
        import duckdb  # noqa: F401, PLC0415
    except ImportError:
        return {}
    return {"duckdb": DuckDBEngine()}

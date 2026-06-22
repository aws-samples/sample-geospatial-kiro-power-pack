"""The Databricks warehouse engine for ``geo-warehouse`` (credentialed).

:class:`DatabricksEngine` is a concrete :class:`~geo_warehouse.engine.WarehouseEngine`
that runs spatial SQL (Databricks' ``ST_*`` / H3 geospatial functions) against a
Databricks SQL warehouse and returns a uniform
:class:`~geo_warehouse.models.ResultSet`.

The driver (``databricks-sql-connector``) is an **optional** dependency (the
``[databricks]`` extra), imported lazily so the rest of ``geo-warehouse`` - and
its test suite, which injects a fake connection - never requires it.

Connection convention (shared across all warehouse engines):
``DATABRICKS_CONNECTION`` holds a JSON object of connection parameters *or a path
to a JSON file*: ``server_hostname`` and ``http_path`` (of the SQL warehouse),
plus an auth credential - either ``access_token`` (a personal access token) or
OAuth machine-to-machine (``client_id``/``client_secret``). An optional
``catalog`` may be set. Databricks uses HTTPS transport. The per-call
``connection`` argument is the **schema** (database) to query (non-secret); it
overrides any ``schema`` in the JSON. Credential values are never echoed into
errors.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from geo_common.errors import AuthenticationError, GeoError, UpstreamError

from geo_warehouse._connection import (
    classify_warehouse_error,
    jsonable,
    load_connection_config,
)
from geo_warehouse.engine import WarehouseEngine
from geo_warehouse.models import ResultSet

__all__ = [
    "DatabricksEngine",
    "DATABRICKS_CONNECTION_KEY",
    "databricks_engine_if_configured",
]

#: The ``mcp.json`` key holding the Databricks connection (JSON or path to JSON).
DATABRICKS_CONNECTION_KEY = "DATABRICKS_CONNECTION"


def _safe_close(obj: Any) -> None:
    if obj is None:
        return
    try:
        obj.close()
    except Exception:  # noqa: BLE001 - close failures must not mask results/errors
        pass


class DatabricksEngine(WarehouseEngine):
    """Execute spatial SQL against a Databricks SQL warehouse.

    A ``connect`` factory may be injected (``connect=...``) for tests; otherwise
    a connection is opened lazily from :data:`DATABRICKS_CONNECTION_KEY`.
    """

    name = "databricks"

    def __init__(
        self,
        *,
        connect: Optional[Any] = None,
        connection_env: str = DATABRICKS_CONNECTION_KEY,
    ) -> None:
        self._connect_override = connect
        self._connection_env = connection_env

    def execute(self, *, query: str, connection: str) -> ResultSet:
        conn = self._open(schema=connection)
        try:
            cursor = conn.cursor()
            try:
                cursor.execute(query)
                description = cursor.description
                columns = [str(col[0]) for col in (description or [])]
                rows = (
                    [[jsonable(v) for v in row] for row in cursor.fetchall()]
                    if description
                    else []
                )
            finally:
                _safe_close(cursor)
        except GeoError:
            raise
        except Exception as exc:  # noqa: BLE001 - driver error
            raise classify_warehouse_error(
                exc, source=self.name, auth_key=self._connection_env
            ) from exc
        finally:
            _safe_close(conn)
        return ResultSet.from_rows(engine=self.name, columns=columns, rows=rows)

    def _open(self, *, schema: str) -> Any:
        """Open a Databricks SQL connection (injected factory or the real driver)."""
        if self._connect_override is not None:
            return self._connect_override(schema=schema)

        raw = os.environ.get(self._connection_env)
        if not (raw and raw.strip()):
            raise AuthenticationError(
                "Databricks is not configured: set %s in mcp.json"
                % self._connection_env,
                source=self.name,
                detail={"engine": self.name, "mcp_json_key": self._connection_env},
            )

        try:
            import databricks.sql  # noqa: PLC0415 - optional dependency
        except ImportError as exc:  # pragma: no cover - exercised only w/o driver
            raise UpstreamError(
                "the Databricks engine requires the 'databricks-sql-connector' "
                "package; install the geo-warehouse[databricks] extra to enable it",
                source=self.name,
                original=str(exc),
            ) from exc

        config = load_connection_config(
            raw, env_key=self._connection_env, source=self.name
        )
        params = dict(config)
        if schema:
            params["schema"] = schema
        try:
            return databricks.sql.connect(**params)
        except Exception as exc:  # noqa: BLE001 - connect/auth failure
            raise classify_warehouse_error(
                exc, source=self.name, auth_key=self._connection_env
            ) from exc


def databricks_engine_if_configured() -> "Optional[WarehouseEngine]":
    """Return a :class:`DatabricksEngine` when ``DATABRICKS_CONNECTION`` is set, else ``None``."""
    raw = os.environ.get(DATABRICKS_CONNECTION_KEY)
    if raw and raw.strip():
        return DatabricksEngine()
    return None

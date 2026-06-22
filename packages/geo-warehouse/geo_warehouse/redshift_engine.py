"""The Amazon Redshift warehouse engine for ``geo-warehouse`` (credentialed).

:class:`RedshiftEngine` is a concrete :class:`~geo_warehouse.engine.WarehouseEngine`
that runs spatial SQL (Redshift's ``GEOMETRY`` type + ``ST_*`` functions) against
Amazon Redshift and returns a uniform :class:`~geo_warehouse.models.ResultSet`.

The driver (``redshift-connector``, Amazon's pure-Python client) is an
**optional** dependency (the ``[redshift]`` extra), imported lazily so the rest
of ``geo-warehouse`` - and its test suite, which injects a fake connection -
never requires it.

Connection convention (shared across all warehouse engines):
``REDSHIFT_CONNECTION`` holds a JSON object of connection parameters *or a path
to a JSON file* (keep the secret in a ``0600`` file, not inline). The keys are
passed straight to ``redshift_connector.connect`` - so password auth
(``host``/``port``/``user``/``password``) and the more secure **IAM auth**
(``iam: true`` with ``cluster_identifier``/``region``/``db_user``, or
``is_serverless`` for Redshift Serverless) both work. TLS is forced on
(``ssl: true``) unless explicitly overridden. The per-call ``connection``
argument is the **database** to query (non-secret); it overrides any
``database`` in the JSON. Credential values are never echoed into errors.
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

__all__ = ["RedshiftEngine", "REDSHIFT_CONNECTION_KEY", "redshift_engine_if_configured"]

#: The ``mcp.json`` key holding the Redshift connection (JSON or path to JSON).
REDSHIFT_CONNECTION_KEY = "REDSHIFT_CONNECTION"


def _safe_close(obj: Any) -> None:
    """Close a cursor/connection, ignoring any close-time error."""
    if obj is None:
        return
    try:
        obj.close()
    except Exception:  # noqa: BLE001 - close failures must not mask results/errors
        pass


class RedshiftEngine(WarehouseEngine):
    """Execute spatial SQL against Amazon Redshift.

    A ``connect`` factory may be injected (``connect=...``) so the engine is
    exercisable without the driver or a real cluster; otherwise a connection is
    opened lazily from the :data:`REDSHIFT_CONNECTION_KEY` parameters.
    """

    name = "redshift"

    def __init__(
        self,
        *,
        connect: Optional[Any] = None,
        connection_env: str = REDSHIFT_CONNECTION_KEY,
    ) -> None:
        self._connect_override = connect
        self._connection_env = connection_env

    def execute(self, *, query: str, connection: str) -> ResultSet:
        conn = self._open(database=connection)
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

    def _open(self, *, database: str) -> Any:
        """Open a Redshift connection (injected factory or the real driver)."""
        if self._connect_override is not None:
            return self._connect_override(database=database)

        raw = os.environ.get(self._connection_env)
        if not (raw and raw.strip()):
            raise AuthenticationError(
                "Amazon Redshift is not configured: set %s in mcp.json"
                % self._connection_env,
                source=self.name,
                detail={"engine": self.name, "mcp_json_key": self._connection_env},
            )

        try:
            import redshift_connector  # noqa: PLC0415 - optional dependency
        except ImportError as exc:  # pragma: no cover - exercised only w/o driver
            raise UpstreamError(
                "the Redshift engine requires the 'redshift-connector' package; "
                "install the geo-warehouse[redshift] extra to enable it",
                source=self.name,
                original=str(exc),
            ) from exc

        config = load_connection_config(
            raw, env_key=self._connection_env, source=self.name
        )
        params = dict(config)
        if database:
            params["database"] = database
        params.setdefault("ssl", True)  # enforce TLS unless explicitly overridden
        try:
            return redshift_connector.connect(**params)
        except Exception as exc:  # noqa: BLE001 - connect/auth failure
            raise classify_warehouse_error(
                exc, source=self.name, auth_key=self._connection_env
            ) from exc


def redshift_engine_if_configured() -> "Optional[WarehouseEngine]":
    """Return a :class:`RedshiftEngine` when ``REDSHIFT_CONNECTION`` is set, else ``None``."""
    raw = os.environ.get(REDSHIFT_CONNECTION_KEY)
    if raw and raw.strip():
        return RedshiftEngine()
    return None

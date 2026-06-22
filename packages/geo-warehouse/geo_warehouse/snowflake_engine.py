"""The Snowflake warehouse engine for ``geo-warehouse`` (credentialed).

:class:`SnowflakeEngine` is a concrete :class:`~geo_warehouse.engine.WarehouseEngine`
that runs spatial SQL (Snowflake's ``GEOGRAPHY``/``GEOMETRY`` types + ``ST_*``
functions) against Snowflake and returns a uniform
:class:`~geo_warehouse.models.ResultSet`.

The driver (``snowflake-connector-python``) is an **optional** dependency (the
``[snowflake]`` extra), imported lazily so the rest of ``geo-warehouse`` - and
its test suite, which injects a fake connection - never requires it.

Connection convention (shared across all warehouse engines):
``SNOWFLAKE_CONNECTION`` holds a JSON object of connection parameters *or a path
to a JSON file*. The keys are passed straight to ``snowflake.connector.connect``
- so password auth (``account``/``user``/``password``) and the more secure
**key-pair** (``private_key_file``) or **SSO/OAuth** (``authenticator``) flows
all work; ``warehouse``/``role``/``schema`` are optional. Snowflake's transport
is always TLS. The per-call ``connection`` argument is the **database** to query
(non-secret); it overrides any ``database`` in the JSON. Credential values are
never echoed into errors.
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
    "SnowflakeEngine",
    "SNOWFLAKE_CONNECTION_KEY",
    "snowflake_engine_if_configured",
]

#: The ``mcp.json`` key holding the Snowflake connection (JSON or path to JSON).
SNOWFLAKE_CONNECTION_KEY = "SNOWFLAKE_CONNECTION"


def _safe_close(obj: Any) -> None:
    if obj is None:
        return
    try:
        obj.close()
    except Exception:  # noqa: BLE001 - close failures must not mask results/errors
        pass


class SnowflakeEngine(WarehouseEngine):
    """Execute spatial SQL against Snowflake.

    A ``connect`` factory may be injected (``connect=...``) for tests; otherwise
    a connection is opened lazily from :data:`SNOWFLAKE_CONNECTION_KEY`.
    """

    name = "snowflake"

    def __init__(
        self,
        *,
        connect: Optional[Any] = None,
        connection_env: str = SNOWFLAKE_CONNECTION_KEY,
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
        """Open a Snowflake connection (injected factory or the real driver)."""
        if self._connect_override is not None:
            return self._connect_override(database=database)

        raw = os.environ.get(self._connection_env)
        if not (raw and raw.strip()):
            raise AuthenticationError(
                "Snowflake is not configured: set %s in mcp.json"
                % self._connection_env,
                source=self.name,
                detail={"engine": self.name, "mcp_json_key": self._connection_env},
            )

        try:
            import snowflake.connector  # noqa: PLC0415 - optional dependency
        except ImportError as exc:  # pragma: no cover - exercised only w/o driver
            raise UpstreamError(
                "the Snowflake engine requires the 'snowflake-connector-python' "
                "package; install the geo-warehouse[snowflake] extra to enable it",
                source=self.name,
                original=str(exc),
            ) from exc

        config = load_connection_config(
            raw, env_key=self._connection_env, source=self.name
        )
        params = dict(config)
        if database:
            params["database"] = database
        try:
            return snowflake.connector.connect(**params)
        except Exception as exc:  # noqa: BLE001 - connect/auth failure
            raise classify_warehouse_error(
                exc, source=self.name, auth_key=self._connection_env
            ) from exc


def snowflake_engine_if_configured() -> "Optional[WarehouseEngine]":
    """Return a :class:`SnowflakeEngine` when ``SNOWFLAKE_CONNECTION`` is set, else ``None``."""
    raw = os.environ.get(SNOWFLAKE_CONNECTION_KEY)
    if raw and raw.strip():
        return SnowflakeEngine()
    return None

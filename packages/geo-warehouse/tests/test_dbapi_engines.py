"""Layer-1 tests for the DBAPI warehouse engines (Redshift, Snowflake, Databricks).

These three concrete engines share the same shape (open a connection, run a
cursor, map results to ``ResultSet``, classify driver errors), so they're tested
together against a **fake connection** injected via the engines' ``connect``
factory - no driver and no account required. Covered: result translation, the
non-secret ``connection`` routing arg (database/schema), the
error -> ``Error_Taxonomy`` mapping (shared classifier), connection cleanup, the
auto-wiring gates, and end-to-end execution through the server.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from geo_common.errors import (
    AuthenticationError,
    ErrorCategory,
    NotFoundError,
    UpstreamError,
    ValidationError,
)

from geo_warehouse import GeoWarehouseServer, default_warehouse_engines
from geo_warehouse.databricks_engine import (
    DATABRICKS_CONNECTION_KEY,
    DatabricksEngine,
    databricks_engine_if_configured,
)
from geo_warehouse.redshift_engine import (
    REDSHIFT_CONNECTION_KEY,
    RedshiftEngine,
    redshift_engine_if_configured,
)
from geo_warehouse.snowflake_engine import (
    SNOWFLAKE_CONNECTION_KEY,
    SnowflakeEngine,
    snowflake_engine_if_configured,
)
from geo_warehouse.models import ResultSet


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "BIGQUERY_CREDENTIALS",
        "BIGQUERY_USE_ADC",
        "GOOGLE_APPLICATION_CREDENTIALS",
        REDSHIFT_CONNECTION_KEY,
        SNOWFLAKE_CONNECTION_KEY,
        DATABRICKS_CONNECTION_KEY,
    ):
        monkeypatch.delenv(key, raising=False)


# --- Fake DBAPI connection/cursor ------------------------------------------


class _FakeCursor:
    def __init__(
        self,
        *,
        description: Optional[List[tuple]] = None,
        rows: Optional[List[List[Any]]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self.description = description
        self._rows = rows or []
        self._error = error
        self.executed: List[str] = []
        self.closed = False

    def execute(self, query: str) -> None:
        self.executed.append(query)
        if self._error is not None:
            raise self._error

    def fetchall(self) -> List[List[Any]]:
        return list(self._rows)

    def close(self) -> None:
        self.closed = True


class _FakeConn:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def close(self) -> None:
        self.closed = True


def _factory(conn: _FakeConn, *, record: Optional[List[Dict[str, Any]]] = None):
    def connect(**kwargs: Any) -> _FakeConn:
        if record is not None:
            record.append(kwargs)
        return conn

    return connect


# (EngineClass, routing kwarg name, engine name, env key, gate fn)
ENGINES = [
    (RedshiftEngine, "database", "redshift", REDSHIFT_CONNECTION_KEY, redshift_engine_if_configured),
    (SnowflakeEngine, "database", "snowflake", SNOWFLAKE_CONNECTION_KEY, snowflake_engine_if_configured),
    (DatabricksEngine, "schema", "databricks", DATABRICKS_CONNECTION_KEY, databricks_engine_if_configured),
]
_IDS = [e[2] for e in ENGINES]


# --- Result translation + routing ------------------------------------------


@pytest.mark.parametrize("EngineCls, routing, name, env_key, gate", ENGINES, ids=_IDS)
def test_execute_returns_result_set_and_routes_connection(
    EngineCls, routing, name, env_key, gate
) -> None:
    cursor = _FakeCursor(
        description=[("geom",), ("n",)],
        rows=[["POINT(0 0)", 1], ["POINT(1 1)", 2]],
    )
    conn = _FakeConn(cursor)
    record: List[Dict[str, Any]] = []
    engine = EngineCls(connect=_factory(conn, record=record))

    result = engine.execute(query="SELECT geom, n FROM t", connection="my_db")

    assert isinstance(result, ResultSet)
    assert result.engine == name
    assert result.columns == ["geom", "n"]
    assert result.rows == [["POINT(0 0)", 1], ["POINT(1 1)", 2]]
    assert result.row_count == 2
    assert cursor.executed == ["SELECT geom, n FROM t"]
    # The non-secret connection arg is routed to the right driver kwarg.
    assert record[0].get(routing) == "my_db"
    # Connection + cursor are always closed.
    assert conn.closed is True
    assert cursor.closed is True


@pytest.mark.parametrize("EngineCls, routing, name, env_key, gate", ENGINES, ids=_IDS)
def test_empty_result_is_valid(EngineCls, routing, name, env_key, gate) -> None:
    engine = EngineCls(connect=_factory(_FakeConn(_FakeCursor(description=[("n",)], rows=[]))))
    result = engine.execute(query="SELECT n FROM t WHERE 1=0", connection="db")
    assert result.row_count == 0
    assert result.columns == ["n"]


@pytest.mark.parametrize("EngineCls, routing, name, env_key, gate", ENGINES, ids=_IDS)
def test_no_result_set_returns_empty(EngineCls, routing, name, env_key, gate) -> None:
    # description=None (e.g. a statement with no result set) -> no fetch, empty.
    engine = EngineCls(connect=_factory(_FakeConn(_FakeCursor(description=None))))
    result = engine.execute(query="CALL something()", connection="db")
    assert result.columns == []
    assert result.rows == []


# --- Error mapping (shared classifier) -------------------------------------


@pytest.mark.parametrize("EngineCls, routing, name, env_key, gate", ENGINES, ids=_IDS)
@pytest.mark.parametrize(
    "message, expected",
    [
        ("syntax error at or near \"FROM\"", ErrorCategory.VALIDATION),
        ("relation \"t\" does not exist", ErrorCategory.NOT_FOUND),
        (
            "[UNRESOLVED_COLUMN.WITHOUT_SUGGESTION] A column, variable, or "
            "function parameter with name FROM cannot be resolved. "
            "SQLSTATE: 42703",
            ErrorCategory.NOT_FOUND,
        ),
        ("password authentication failed for user", ErrorCategory.AUTHENTICATION),
        ("connection reset by peer", ErrorCategory.UPSTREAM),
    ],
)
def test_execute_error_mapping(
    EngineCls, routing, name, env_key, gate, message, expected
) -> None:
    cursor = _FakeCursor(error=RuntimeError(message))
    engine = EngineCls(connect=_factory(_FakeConn(cursor)))
    with pytest.raises(Exception) as exc_info:
        engine.execute(query="SELECT 1", connection="db")
    assert exc_info.value.category is expected


@pytest.mark.parametrize("EngineCls, routing, name, env_key, gate", ENGINES, ids=_IDS)
def test_auth_error_names_connection_key(
    EngineCls, routing, name, env_key, gate
) -> None:
    cursor = _FakeCursor(error=RuntimeError("password authentication failed"))
    engine = EngineCls(connect=_factory(_FakeConn(cursor)))
    with pytest.raises(AuthenticationError) as exc_info:
        engine.execute(query="SELECT 1", connection="db")
    assert exc_info.value.detail is not None
    assert exc_info.value.detail["mcp_json_key"] == env_key


# --- Auto-wiring gates -----------------------------------------------------


@pytest.mark.parametrize("EngineCls, routing, name, env_key, gate", ENGINES, ids=_IDS)
def test_gate_none_without_credential(
    EngineCls, routing, name, env_key, gate, monkeypatch
) -> None:
    monkeypatch.delenv(env_key, raising=False)
    assert gate() is None


@pytest.mark.parametrize("EngineCls, routing, name, env_key, gate", ENGINES, ids=_IDS)
def test_gate_wires_when_credential_present(
    EngineCls, routing, name, env_key, gate, monkeypatch
) -> None:
    monkeypatch.setenv(env_key, '{"host": "h", "user": "u"}')
    engine = gate()
    assert isinstance(engine, EngineCls)
    # And it shows up in the aggregated set.
    assert name in default_warehouse_engines()


def test_default_warehouse_engines_aggregates_multiple(monkeypatch) -> None:
    monkeypatch.setenv(REDSHIFT_CONNECTION_KEY, '{"host": "h", "user": "u"}')
    monkeypatch.setenv(SNOWFLAKE_CONNECTION_KEY, '{"account": "a", "user": "u"}')
    engines = default_warehouse_engines()
    assert {"redshift", "snowflake"} <= set(engines)
    assert "databricks" not in engines  # not configured


# --- Credential guard (no driver, env unset) -------------------------------


@pytest.mark.parametrize("EngineCls, routing, name, env_key, gate", ENGINES, ids=_IDS)
def test_missing_connection_is_authentication_error(
    EngineCls, routing, name, env_key, gate, monkeypatch
) -> None:
    monkeypatch.delenv(env_key, raising=False)
    engine = EngineCls()  # no injected connect -> must read env
    with pytest.raises(AuthenticationError) as exc_info:
        engine.execute(query="SELECT 1", connection="db")
    assert exc_info.value.detail is not None
    assert exc_info.value.detail["mcp_json_key"] == env_key


# --- End-to-end through the server -----------------------------------------


@pytest.mark.parametrize("EngineCls, routing, name, env_key, gate", ENGINES, ids=_IDS)
async def test_server_runs_engine_via_injected_connection(
    EngineCls, routing, name, env_key, gate
) -> None:
    cursor = _FakeCursor(description=[("n",)], rows=[[1]])
    engine = EngineCls(connect=_factory(_FakeConn(cursor)))
    server = GeoWarehouseServer(engines={name: engine})
    result = await server.warehouse_spatial_sql(
        query="SELECT 1 AS n", engine=name, connection="my_db"
    )
    assert result.engine == name
    assert result.rows == [[1]]

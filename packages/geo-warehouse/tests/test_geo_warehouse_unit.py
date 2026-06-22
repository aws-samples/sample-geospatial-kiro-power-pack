"""Focused unit tests for ``geo-warehouse`` (task 12.2).

Task 12.1 implemented ``warehouse_spatial_sql`` and its companion suite
``test_warehouse_spatial_sql.py`` covers the full surface (catalog, credential
declarations, taxonomy mapping). This module is the task-12.2 deliverable: a
tight unit suite pinning the **two** behaviours the task calls out, exercised
entirely against a mocked :class:`~geo_warehouse.engine.WarehouseEngine` so no
proprietary driver is required:

* **Query execution (Requirement 8.6)** - a valid invocation forwards the
  spatial SQL ``query`` and ``connection`` to the configured engine *unchanged*
  and returns the engine's :class:`~geo_warehouse.models.ResultSet` to the
  caller, for each of the five supported engines and for the empty-result case.
* **The credential guard (Requirement 8.6; Req 3.6, 3.8, 10.5)** - invoking a
  supported engine whose proprietary credential is not configured is denied
  with an ``Error_Taxonomy`` :class:`~geo_common.errors.AuthenticationError`
  that *names* the engine's ``mcp.json`` key (never a secret value) and the
  query is **never executed** (the engine's ``execute`` is not called).

The repo's ``asyncio_mode = "auto"`` lets the ``async def`` tests run directly.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import pytest

from geo_common.errors import AuthenticationError, ErrorCategory

from geo_warehouse.engine import SUPPORTED_ENGINES, WarehouseEngine
from geo_warehouse.models import ResultSet
from geo_warehouse.server import GeoWarehouseServer


class _RecordingEngine(WarehouseEngine):
    """A mock engine that records every ``execute`` call (no real driver).

    ``result`` is the :class:`ResultSet` returned on success; when omitted a
    deterministic two-row set tagged with the engine name is produced so a
    happy-path call always yields a well-formed result.
    """

    def __init__(self, name: str, *, result: Optional[ResultSet] = None) -> None:
        self.name = name
        self._result = result
        #: Every (query, connection) pair this engine was asked to execute.
        self.calls: List[Tuple[str, str]] = []

    def execute(self, *, query: str, connection: str) -> ResultSet:
        self.calls.append((query, connection))
        if self._result is not None:
            return self._result
        return ResultSet.from_rows(
            engine=self.name,
            columns=["wkt", "n"],
            rows=[["POINT(0 0)", 1], ["POINT(1 1)", 2]],
        )


# ---------------------------------------------------------------------------
# Query execution against a mocked engine (Requirement 8.6)
# ---------------------------------------------------------------------------


async def test_query_runs_on_configured_engine_and_returns_result_set() -> None:
    """Req 8.6: a valid query executes on the configured engine and its rows
    are returned to the caller unchanged."""
    engine = _RecordingEngine("bigquery")
    server = GeoWarehouseServer(engines={"bigquery": engine})

    result = await server.warehouse_spatial_sql(
        query="SELECT ST_AsText(geom) AS wkt, COUNT(*) AS n FROM t GROUP BY 1",
        engine="bigquery",
        connection="project.dataset",
    )

    assert isinstance(result, ResultSet)
    assert result.engine == "bigquery"
    assert result.columns == ["wkt", "n"]
    assert result.row_count == 2
    assert result.rows == [["POINT(0 0)", 1], ["POINT(1 1)", 2]]


async def test_query_and_connection_are_forwarded_to_engine_unchanged() -> None:
    """Req 8.6: the exact SQL string and connection identifier reach the engine
    verbatim - the server is a thin, faithful executor."""
    engine = _RecordingEngine("snowflake")
    server = GeoWarehouseServer(engines={"snowflake": engine})

    query = "SELECT ST_Distance(a.g, b.g) FROM a JOIN b ON a.id = b.id"
    connection = "account/warehouse/db/schema"
    await server.warehouse_spatial_sql(query=query, engine="snowflake", connection=connection)

    # Executed exactly once, with the inputs passed through unchanged.
    assert engine.calls == [(query, connection)]


@pytest.mark.parametrize("engine_name", list(SUPPORTED_ENGINES))
async def test_every_supported_engine_executes_against_its_mock(engine_name: str) -> None:
    """Req 8.6: all five proprietary engines accept a query and return a
    result set attributed to that engine when configured."""
    engine = _RecordingEngine(engine_name)
    server = GeoWarehouseServer(engines={engine_name: engine})

    result = await server.warehouse_spatial_sql(
        query="SELECT 1", engine=engine_name, connection="conn"
    )

    assert result.engine == engine_name
    assert engine.calls == [("SELECT 1", "conn")]


async def test_empty_result_set_is_returned_for_a_no_match_query() -> None:
    """Req 8.6: a query matching no rows yields a valid empty ResultSet (not an
    error)."""
    empty = ResultSet.from_rows(engine="redshift", columns=["geom"], rows=[])
    engine = _RecordingEngine("redshift", result=empty)
    server = GeoWarehouseServer(engines={"redshift": engine})

    result = await server.warehouse_spatial_sql(
        query="SELECT geom FROM t WHERE 1 = 0", engine="redshift", connection="conn"
    )

    assert result.row_count == 0
    assert result.rows == []
    assert engine.calls == [("SELECT geom FROM t WHERE 1 = 0", "conn")]


async def test_only_the_selected_engine_executes_when_several_are_configured() -> None:
    """Req 8.6: with multiple engines configured, only the selected one runs."""
    bq = _RecordingEngine("bigquery")
    sf = _RecordingEngine("snowflake")
    server = GeoWarehouseServer(engines={"bigquery": bq, "snowflake": sf})

    await server.warehouse_spatial_sql(query="SELECT 1", engine="snowflake", connection="conn")

    assert sf.calls == [("SELECT 1", "conn")]
    assert bq.calls == []  # the unselected engine was never touched


# ---------------------------------------------------------------------------
# Credential guard against a mocked engine (Req 8.6; 3.6, 3.8, 10.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine_name", list(SUPPORTED_ENGINES))
async def test_unconfigured_credential_denies_with_authentication_error_naming_key(
    engine_name: str,
) -> None:
    """Credential guard: a supported engine with no configured credential is
    denied with an AuthenticationError naming its mcp.json key, for every
    engine."""
    server = GeoWarehouseServer(engines={})  # no credentials configured

    with pytest.raises(AuthenticationError) as exc_info:
        await server.warehouse_spatial_sql(
            query="SELECT 1", engine=engine_name, connection="conn"
        )

    err = exc_info.value
    assert err.category is ErrorCategory.AUTHENTICATION
    assert err.detail is not None
    spec = SUPPORTED_ENGINES[engine_name]
    # The mcp.json key is named so the user knows which credential to set.
    assert err.detail.get("mcp_json_key") == spec.mcp_json_key
    assert err.detail.get("engine") == engine_name


async def test_credential_guard_does_not_execute_the_query() -> None:
    """Credential guard: when the *selected* engine is unconfigured, no query is
    executed - even though another engine happens to be configured."""
    other = _RecordingEngine("bigquery")
    # bigquery is configured; the caller selects the unconfigured snowflake engine.
    server = GeoWarehouseServer(engines={"bigquery": other})

    with pytest.raises(AuthenticationError):
        await server.warehouse_spatial_sql(query="SELECT 1", engine="snowflake", connection="conn")

    # Nothing ran: the configured engine was not used as a fallback.
    assert other.calls == []


async def test_authentication_error_never_exposes_the_connection_secret() -> None:
    """Req 3.8: the guard names the credential key but never echoes the secret
    connection string supplied by the caller."""
    server = GeoWarehouseServer(engines={})
    secret_connection = "user:sup3r-secret-pw@host:5439/db"

    with pytest.raises(AuthenticationError) as exc_info:
        await server.warehouse_spatial_sql(
            query="SELECT 1", engine="databricks", connection=secret_connection
        )

    err = exc_info.value
    assert secret_connection not in str(err)
    assert secret_connection not in repr(err)
    assert secret_connection not in str(err.detail)
    assert secret_connection not in (err.original or "")

"""Tests for ``geo-warehouse``'s ``warehouse_spatial_sql`` (task 12.1).

Covers Requirement 8.6 (execute spatial SQL against the configured engine and
return the result set), the proprietary License-Needed credential declarations
and Proprietary catalog registration (Req 2.1, 11.3, 16.1, 16.8), and the
taxonomy error mapping (Req 11.2, 11.5) - all exercised against a mock engine so
no proprietary driver is required.
"""

from __future__ import annotations

import httpx
import pytest

from geo_common.errors import (
    AuthenticationError,
    ErrorCategory,
    GeoError,
    NetworkError,
    UpstreamError,
    ValidationError,
)
from geo_common.models import CredentialClassification, OpennessTier

from geo_warehouse.engine import SUPPORTED_ENGINES, WarehouseEngine
from geo_warehouse.models import ResultSet
from geo_warehouse.server import INSTALL_COMMAND, GeoWarehouseServer


class _MockEngine(WarehouseEngine):
    """A controllable in-memory engine for tests (no proprietary driver)."""

    def __init__(self, name: str, *, result: ResultSet | None = None, raises: Exception | None = None) -> None:
        self.name = name
        self._result = result
        self._raises = raises
        self.calls: list[tuple[str, str]] = []

    def execute(self, *, query: str, connection: str) -> ResultSet:
        self.calls.append((query, connection))
        if self._raises is not None:
            raise self._raises
        if self._result is not None:
            return self._result
        return ResultSet.from_rows(
            engine=self.name,
            columns=["geom", "n"],
            rows=[["POINT(0 0)", 1], ["POINT(1 1)", 2]],
        )


def _server_with(engine_name: str = "bigquery", **kwargs) -> tuple[GeoWarehouseServer, _MockEngine]:
    engine = _MockEngine(engine_name, **kwargs)
    server = GeoWarehouseServer(engines={engine_name: engine})
    return server, engine


# --- Query execution (Req 8.6) --------------------------------------------


async def test_executes_query_against_configured_engine_and_returns_result_set() -> None:
    """Req 8.6: a valid query runs on the configured engine and returns its rows."""
    server, engine = _server_with("snowflake")
    result = await server.warehouse_spatial_sql(
        query="SELECT ST_AsText(geom) FROM t",
        engine="snowflake",
        connection="acct/db/schema",
    )
    assert isinstance(result, ResultSet)
    assert result.engine == "snowflake"
    assert result.row_count == 2
    assert result.columns == ["geom", "n"]
    # The query/connection were passed through to the engine unchanged.
    assert engine.calls == [("SELECT ST_AsText(geom) FROM t", "acct/db/schema")]


async def test_empty_query_result_is_valid() -> None:
    """Req 8.6: a query that matches nothing returns an empty ResultSet."""
    empty = ResultSet.from_rows(engine="snowflake", columns=["geom"], rows=[])
    server, _ = _server_with("snowflake", result=empty)
    result = await server.warehouse_spatial_sql(
        query="SELECT geom FROM t WHERE 1=0", engine="snowflake", connection="conn"
    )
    assert result.row_count == 0
    assert result.rows == []


@pytest.mark.parametrize("engine_name", list(SUPPORTED_ENGINES))
async def test_each_supported_engine_executes(engine_name: str) -> None:
    """Req 8.6: all five proprietary engines are accepted and execute."""
    server, _ = _server_with(engine_name)
    result = await server.warehouse_spatial_sql(
        query="SELECT 1", engine=engine_name, connection="conn"
    )
    assert result.engine == engine_name


# --- Input validation -> VALIDATION (Req 8.6) -----------------------------


@pytest.mark.parametrize("bad_query", ["", "   ", "\n"])
async def test_blank_query_is_validation_error(bad_query: str) -> None:
    server, engine = _server_with()
    with pytest.raises(ValidationError) as exc_info:
        await server.warehouse_spatial_sql(query=bad_query, engine="bigquery", connection="conn")
    assert exc_info.value.category is ErrorCategory.VALIDATION
    assert engine.calls == []  # nothing executed


async def test_blank_connection_is_validation_error() -> None:
    server, engine = _server_with()
    with pytest.raises(ValidationError) as exc_info:
        await server.warehouse_spatial_sql(query="SELECT 1", engine="bigquery", connection="  ")
    assert exc_info.value.category is ErrorCategory.VALIDATION
    assert engine.calls == []


async def test_unknown_engine_is_validation_error() -> None:
    server, _ = _server_with()
    with pytest.raises(ValidationError) as exc_info:
        await server.warehouse_spatial_sql(query="SELECT 1", engine="postgres", connection="conn")
    assert exc_info.value.category is ErrorCategory.VALIDATION


# --- Credential guard (Req 3.6, 3.8, 10.5) --------------------------------


async def test_unconfigured_engine_denied_naming_mcp_json_key() -> None:
    """A known engine with no configured credential -> AuthenticationError naming the key."""
    server = GeoWarehouseServer(engines={})  # nothing configured
    with pytest.raises(AuthenticationError) as exc_info:
        await server.warehouse_spatial_sql(query="SELECT 1", engine="redshift", connection="conn")
    err = exc_info.value
    assert err.category is ErrorCategory.AUTHENTICATION
    # Names the mcp.json key, never a secret value.
    assert err.detail is not None
    assert err.detail["mcp_json_key"] == SUPPORTED_ENGINES["redshift"].mcp_json_key


def test_server_starts_without_any_engine_credentials() -> None:
    """Req 16.5: License-Needed credentials never block startup."""
    server = GeoWarehouseServer()
    server.start(configured_keys=[])
    assert server.started is True


# --- Credential declaration (Req 16.1, 3.7) -------------------------------


def test_required_credentials_are_license_needed_with_references() -> None:
    server = GeoWarehouseServer()
    specs = server.required_credentials()
    assert {s.mcp_json_key for s in specs} == {s.mcp_json_key for s in SUPPORTED_ENGINES.values()}
    for spec in specs:
        assert spec.classification is CredentialClassification.LICENSE_NEEDED
        assert spec.license_reference  # non-empty licensing reference (Req 3.7)


# --- Catalog registration (Req 2.1, 11.3) ---------------------------------


def test_catalog_entries_are_proprietary_and_self_provided() -> None:
    server = GeoWarehouseServer()
    entries = server.catalog_entries()
    assert len(entries) == len(SUPPORTED_ENGINES)
    for entry in entries:
        assert entry.pillar == "B"
        assert entry.provider_server == "geo-warehouse"
        assert entry.openness_tier is OpennessTier.PROPRIETARY
        assert 1 <= len(entry.capability_description) <= 500


def test_catalog_marks_configured_engine_installed() -> None:
    server, _ = _server_with("databricks")
    by_name = {e.name: e for e in server.catalog_entries()}
    configured = by_name["warehouse_spatial_sql:databricks"]
    other = by_name["warehouse_spatial_sql:snowflake"]
    assert configured.installed is True and configured.install_command is None
    assert other.installed is False and other.install_command == INSTALL_COMMAND


# --- Error mapping onto the taxonomy (Req 11.2, 11.5) ---------------------


async def test_engine_geoerror_passes_through_unchanged() -> None:
    boom = AuthenticationError("rejected token", source="bigquery")
    server, _ = _server_with("bigquery", raises=boom)
    with pytest.raises(AuthenticationError) as exc_info:
        await server.warehouse_spatial_sql(query="SELECT 1", engine="bigquery", connection="conn")
    assert exc_info.value is boom


async def test_unmapped_driver_error_maps_to_upstream() -> None:
    server, _ = _server_with("snowflake", raises=RuntimeError("driver exploded"))
    with pytest.raises(UpstreamError) as exc_info:
        await server.warehouse_spatial_sql(query="SELECT 1", engine="snowflake", connection="conn")
    err = exc_info.value
    assert err.category is ErrorCategory.UPSTREAM
    assert err.original is not None and "driver exploded" in err.original


async def test_transport_failure_maps_to_network() -> None:
    request = httpx.Request("POST", "https://bigquery.googleapis.com")
    server, _ = _server_with("bigquery", raises=httpx.ConnectError("refused", request=request))
    with pytest.raises(NetworkError) as exc_info:
        await server.warehouse_spatial_sql(query="SELECT 1", engine="bigquery", connection="conn")
    assert exc_info.value.category is ErrorCategory.NETWORK


def test_constructing_server_with_unknown_engine_key_raises() -> None:
    with pytest.raises(ValueError):
        GeoWarehouseServer(engines={"oracle": _MockEngine("oracle")})


def test_warehouse_spatial_sql_is_registered_tool() -> None:
    server = GeoWarehouseServer()
    assert "warehouse_spatial_sql" in server.tool_names()


def test_result_set_rejects_ragged_rows() -> None:
    """ResultSet keeps rows rectangular and row_count consistent."""
    with pytest.raises(Exception):
        ResultSet(engine="bigquery", columns=["a", "b"], rows=[[1]], row_count=1)
    with pytest.raises(Exception):
        ResultSet(engine="bigquery", columns=["a"], rows=[[1]], row_count=2)

"""Tests for ``geo-query``'s ``spatial_sql`` (task 14.4).

Covers Requirement 8.6 (execute spatial SQL against the configured engine and
return the result set), input validation onto the taxonomy (Req 8.6), the
per-engine catalog registration at the right openness tier and the Optional
Athena credential declaration (Req 2.1, 11.3, 16.1, 16.5), and the taxonomy
error mapping (Req 11.2, 11.5) - all exercised against a mock engine so no
driver is required.

The repo's ``asyncio_mode = "auto"`` lets the ``async def`` tests run directly.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

import httpx
import pytest

from geo_common.errors import (
    AuthenticationError,
    ErrorCategory,
    NetworkError,
    UpstreamError,
    ValidationError,
)
from geo_common.models import CredentialClassification, OpennessTier

from geo_query.engine import DEFAULT_ENGINE, SUPPORTED_ENGINES, QueryEngine
from geo_query.models import ResultSet
from geo_query.server import INSTALL_COMMAND, GeoQueryServer


class _MockEngine(QueryEngine):
    """A controllable in-memory engine for tests (no real driver).

    ``result`` is the :class:`ResultSet` returned on success; when omitted a
    deterministic two-row set tagged with the engine name is produced. ``raises``
    lets a test force a failure to exercise error mapping. Every call's
    ``(query, sources)`` is recorded so forwarding can be asserted.
    """

    def __init__(
        self,
        name: str,
        *,
        result: Optional[ResultSet] = None,
        raises: Optional[Exception] = None,
    ) -> None:
        self.name = name
        self._result = result
        self._raises = raises
        self.calls: List[Tuple[str, Optional[Mapping[str, str]]]] = []

    def execute(
        self, *, query: str, sources: Optional[Mapping[str, str]] = None
    ) -> ResultSet:
        self.calls.append((query, sources))
        if self._raises is not None:
            raise self._raises
        if self._result is not None:
            return self._result
        return ResultSet.from_rows(
            engine=self.name,
            columns=["wkt", "n"],
            rows=[["POINT(0 0)", 1], ["POINT(1 1)", 2]],
        )


def _server_with(engine_name: str = "duckdb", **kwargs) -> Tuple[GeoQueryServer, _MockEngine]:
    engine = _MockEngine(engine_name, **kwargs)
    server = GeoQueryServer(engines={engine_name: engine})
    return server, engine


# --- Query execution (Req 8.6) --------------------------------------------


async def test_executes_query_against_configured_engine_and_returns_result_set() -> None:
    """Req 8.6: a valid query runs on the configured engine and returns its rows."""
    server, engine = _server_with("duckdb")
    result = await server.spatial_sql(
        query="SELECT ST_AsText(geom) AS wkt, COUNT(*) AS n FROM t GROUP BY 1",
        engine="duckdb",
    )
    assert isinstance(result, ResultSet)
    assert result.engine == "duckdb"
    assert result.columns == ["wkt", "n"]
    assert result.row_count == 2
    assert result.rows == [["POINT(0 0)", 1], ["POINT(1 1)", 2]]
    assert engine.calls == [
        ("SELECT ST_AsText(geom) AS wkt, COUNT(*) AS n FROM t GROUP BY 1", None)
    ]


async def test_default_engine_is_duckdb() -> None:
    """Req 8.6: omitting ``engine`` uses the default (DuckDB) engine."""
    assert DEFAULT_ENGINE == "duckdb"
    server, engine = _server_with("duckdb")
    result = await server.spatial_sql(query="SELECT 1")
    assert result.engine == "duckdb"
    assert engine.calls == [("SELECT 1", None)]


async def test_query_and_sources_are_forwarded_to_engine_unchanged() -> None:
    """Req 8.6: the SQL string and the ``sources`` mapping reach the engine
    verbatim - the server is a thin, faithful executor."""
    server, engine = _server_with("duckdb")
    query = "SELECT * FROM read_parquet('points') WHERE ST_Within(geom, bbox)"
    sources = {"points": "s3://amzn-s3-demo-bucket/points.parquet"}
    await server.spatial_sql(query=query, engine="duckdb", sources=sources)
    assert engine.calls == [(query, sources)]


@pytest.mark.parametrize("engine_name", list(SUPPORTED_ENGINES))
async def test_each_supported_engine_executes(engine_name: str) -> None:
    """Req 8.6: both supported engines accept a query and return a result set
    attributed to that engine when configured."""
    server, engine = _server_with(engine_name)
    result = await server.spatial_sql(query="SELECT 1", engine=engine_name)
    assert result.engine == engine_name
    assert engine.calls == [("SELECT 1", None)]


async def test_empty_query_result_is_valid() -> None:
    """Req 8.6: a query that matches nothing returns an empty ResultSet."""
    empty = ResultSet.from_rows(engine="duckdb", columns=["geom"], rows=[])
    server, _ = _server_with("duckdb", result=empty)
    result = await server.spatial_sql(
        query="SELECT geom FROM t WHERE 1=0", engine="duckdb"
    )
    assert result.row_count == 0
    assert result.rows == []


async def test_only_the_selected_engine_executes_when_several_are_configured() -> None:
    """Req 8.6: with multiple engines configured, only the selected one runs."""
    duck = _MockEngine("duckdb")
    athena = _MockEngine("athena")
    server = GeoQueryServer(engines={"duckdb": duck, "athena": athena})
    await server.spatial_sql(query="SELECT 1", engine="athena")
    assert athena.calls == [("SELECT 1", None)]
    assert duck.calls == []  # the unselected engine was never touched


# --- Input validation -> VALIDATION (Req 8.6) -----------------------------


@pytest.mark.parametrize("bad_query", ["", "   ", "\n"])
async def test_blank_query_is_validation_error(bad_query: str) -> None:
    server, engine = _server_with()
    with pytest.raises(ValidationError) as exc_info:
        await server.spatial_sql(query=bad_query, engine="duckdb")
    assert exc_info.value.category is ErrorCategory.VALIDATION
    assert engine.calls == []  # nothing executed


async def test_unknown_engine_is_validation_error() -> None:
    server, engine = _server_with()
    with pytest.raises(ValidationError) as exc_info:
        await server.spatial_sql(query="SELECT 1", engine="postgres")
    assert exc_info.value.category is ErrorCategory.VALIDATION
    assert engine.calls == []


@pytest.mark.parametrize(
    "bad_sources",
    [
        {"": "s3://amzn-s3-demo-bucket/x.parquet"},
        {"  ": "s3://amzn-s3-demo-bucket/x.parquet"},
        {"points": ""},
        {"points": "   "},
    ],
)
async def test_malformed_sources_is_validation_error(bad_sources) -> None:
    server, engine = _server_with()
    with pytest.raises(ValidationError) as exc_info:
        await server.spatial_sql(query="SELECT 1", engine="duckdb", sources=bad_sources)
    assert exc_info.value.category is ErrorCategory.VALIDATION
    assert engine.calls == []  # validation runs before execution


# --- Credential / configuration guard -------------------------------------


async def test_unconfigured_athena_denied_naming_mcp_json_key() -> None:
    """Athena with no configured credential -> AuthenticationError naming the key."""
    server = GeoQueryServer(engines={})  # nothing configured
    with pytest.raises(AuthenticationError) as exc_info:
        await server.spatial_sql(query="SELECT 1", engine="athena")
    err = exc_info.value
    assert err.category is ErrorCategory.AUTHENTICATION
    assert err.detail is not None
    assert err.detail["mcp_json_key"] == SUPPORTED_ENGINES["athena"].mcp_json_key
    assert err.detail["engine"] == "athena"


async def test_unconfigured_duckdb_reports_upstream_not_configured() -> None:
    """The open DuckDB engine, when its runtime is not wired, reports UPSTREAM
    (it carries no credential to authenticate)."""
    server = GeoQueryServer(engines={})
    with pytest.raises(UpstreamError) as exc_info:
        await server.spatial_sql(query="SELECT 1", engine="duckdb")
    assert exc_info.value.category is ErrorCategory.UPSTREAM


async def test_credential_guard_does_not_execute_the_query() -> None:
    """When the selected engine is unconfigured, no query is executed - even
    though another engine happens to be configured."""
    duck = _MockEngine("duckdb")
    server = GeoQueryServer(engines={"duckdb": duck})
    with pytest.raises(AuthenticationError):
        await server.spatial_sql(query="SELECT 1", engine="athena")
    assert duck.calls == []  # no fallback to a different engine


async def test_authentication_error_never_exposes_a_secret() -> None:
    """Req 3.8: the guard names the credential key but never echoes secrets."""
    server = GeoQueryServer(engines={})
    with pytest.raises(AuthenticationError) as exc_info:
        await server.spatial_sql(query="SELECT 1", engine="athena")
    err = exc_info.value
    # The mcp.json key is named; no secret value is present anywhere.
    assert SUPPORTED_ENGINES["athena"].mcp_json_key in str(err)
    assert "secret" not in str(err).lower()


# --- Startup credential guard (Req 16.5) ----------------------------------


def test_server_starts_without_any_engine_credentials() -> None:
    """Req 16.5: the Optional Athena credential never blocks startup, and the
    open DuckDB engine needs none."""
    server = GeoQueryServer()
    server.start(configured_keys=[])
    assert server.started is True


# --- Credential declaration (Req 16.1) ------------------------------------


def test_only_athena_declares_optional_credentials() -> None:
    server = GeoQueryServer()
    specs = server.required_credentials()
    # Athena declares the full AWS/Athena credential set; DuckDB declares none.
    keys = {s.mcp_json_key for s in specs}
    assert keys == {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "ATHENA_S3_STAGING_DIR",
    }
    assert all(
        s.classification is CredentialClassification.OPTIONAL for s in specs
    )
    # The primary deny-message key is part of the declared set.
    assert SUPPORTED_ENGINES["athena"].mcp_json_key in keys


# --- Catalog registration (Req 2.1, 11.3) ---------------------------------


def test_catalog_entries_cover_each_engine_at_its_openness_tier() -> None:
    server = GeoQueryServer()
    entries = {e.name: e for e in server.catalog_entries()}
    assert set(entries) == {"spatial_sql:duckdb", "spatial_sql:athena"}
    for entry in entries.values():
        assert entry.pillar == "B"
        assert entry.provider_server == "geo-query"
        assert 1 <= len(entry.capability_description) <= 500
    assert entries["spatial_sql:duckdb"].openness_tier is OpennessTier.OPEN
    assert entries["spatial_sql:athena"].openness_tier is OpennessTier.FREE_TIER


def test_catalog_marks_configured_engine_installed() -> None:
    server, _ = _server_with("duckdb")
    by_name = {e.name: e for e in server.catalog_entries()}
    configured = by_name["spatial_sql:duckdb"]
    other = by_name["spatial_sql:athena"]
    assert configured.installed is True and configured.install_command is None
    assert other.installed is False and other.install_command == INSTALL_COMMAND


# --- Error mapping onto the taxonomy (Req 11.2, 11.5) ---------------------


async def test_engine_geoerror_passes_through_unchanged() -> None:
    boom = AuthenticationError("rejected AWS credentials", source="athena")
    server = GeoQueryServer(engines={"athena": _MockEngine("athena", raises=boom)})
    with pytest.raises(AuthenticationError) as exc_info:
        await server.spatial_sql(query="SELECT 1", engine="athena")
    assert exc_info.value is boom


async def test_unmapped_driver_error_maps_to_upstream() -> None:
    server, _ = _server_with("duckdb", raises=RuntimeError("engine exploded"))
    with pytest.raises(UpstreamError) as exc_info:
        await server.spatial_sql(query="SELECT 1", engine="duckdb")
    err = exc_info.value
    assert err.category is ErrorCategory.UPSTREAM
    assert err.original is not None and "engine exploded" in err.original


async def test_transport_failure_maps_to_network() -> None:
    request = httpx.Request("POST", "https://athena.us-east-1.amazonaws.com")
    server, _ = _server_with(
        "athena", raises=httpx.ConnectError("refused", request=request)
    )
    with pytest.raises(NetworkError) as exc_info:
        await server.spatial_sql(query="SELECT 1", engine="athena")
    assert exc_info.value.category is ErrorCategory.NETWORK


# --- Misc invariants -------------------------------------------------------


def test_constructing_server_with_unknown_engine_key_raises() -> None:
    with pytest.raises(ValueError):
        GeoQueryServer(engines={"oracle": _MockEngine("oracle")})


def test_spatial_sql_is_registered_tool() -> None:
    server = GeoQueryServer()
    assert "spatial_sql" in server.tool_names()


def test_result_set_rejects_ragged_rows() -> None:
    """ResultSet keeps rows rectangular and row_count consistent."""
    with pytest.raises(Exception):
        ResultSet(engine="duckdb", columns=["a", "b"], rows=[[1]], row_count=1)
    with pytest.raises(Exception):
        ResultSet(engine="duckdb", columns=["a"], rows=[[1]], row_count=2)

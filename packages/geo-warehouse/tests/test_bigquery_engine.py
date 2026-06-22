"""Layer-1 tests for the concrete BigQuery warehouse engine (no driver, no account).

The :class:`~geo_warehouse.bigquery_engine.BigQueryEngine` is exercised against a
**fake BigQuery client** injected into the engine, so these tests verify the
request/response translation (query -> ``ResultSet``), the credential loading
(inline JSON vs. file path vs. malformed), the credential guard, and the
driver-error -> ``Error_Taxonomy`` mapping **without** installing
``google-cloud-bigquery`` or contacting Google. End-to-end verification against a
real BigQuery sandbox is a manual step documented in ``docs/testing-workflows.md``.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

import pytest

from geo_common.errors import (
    AuthenticationError,
    ErrorCategory,
    NotFoundError,
    UpstreamError,
    ValidationError,
)

from geo_warehouse import (
    BIGQUERY_CREDENTIALS_KEY,
    BIGQUERY_USE_ADC_KEY,
    BigQueryEngine,
    GeoWarehouseServer,
    default_warehouse_engines,
)
from geo_warehouse.models import ResultSet


@pytest.fixture(autouse=True)
def _clear_bigquery_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep warehouse auth env deterministic across tests (all engines' signals)."""
    for key in (
        BIGQUERY_CREDENTIALS_KEY,
        BIGQUERY_USE_ADC_KEY,
        "GOOGLE_APPLICATION_CREDENTIALS",
        "REDSHIFT_CONNECTION",
        "SNOWFLAKE_CONNECTION",
        "DATABRICKS_CONNECTION",
    ):
        monkeypatch.delenv(key, raising=False)


# --- Fake BigQuery client (stands in for google.cloud.bigquery.Client) -----


class _FakeField:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeRow:
    def __init__(self, values: List[Any]) -> None:
        self._values = list(values)

    def values(self) -> List[Any]:
        return list(self._values)


class _FakeResult:
    def __init__(self, columns: List[str], rows: List[List[Any]]) -> None:
        self.schema = [_FakeField(name) for name in columns]
        self._rows = [_FakeRow(row) for row in rows]

    def __iter__(self):
        return iter(self._rows)


class _FakeJob:
    def __init__(self, result: Any) -> None:
        self._result = result

    def result(self) -> Any:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _FakeClient:
    """Records queries and returns a canned result (or raises a canned error)."""

    def __init__(
        self,
        *,
        columns: Optional[List[str]] = None,
        rows: Optional[List[List[Any]]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self._columns = columns or []
        self._rows = rows or []
        self._error = error
        self.queries: List[str] = []

    def query(self, query: str) -> _FakeJob:
        self.queries.append(query)
        if self._error is not None:
            return _FakeJob(self._error)
        return _FakeJob(_FakeResult(self._columns, self._rows))


# Google-style API errors carrying a numeric ``code`` and a recognizable name.
class _ApiError(Exception):
    def __init__(self, message: str, code: Optional[int] = None) -> None:
        super().__init__(message)
        self.code = code


class Forbidden(_ApiError):
    pass


class NotFound(_ApiError):
    pass


class BadRequest(_ApiError):
    pass


# --- Query -> ResultSet translation ----------------------------------------


def test_execute_translates_rows_to_result_set() -> None:
    client = _FakeClient(
        columns=["geom", "n"],
        rows=[["POINT(0 0)", 1], ["POINT(1 1)", 2]],
    )
    engine = BigQueryEngine(client=client)
    result = engine.execute(query="SELECT geom, n FROM t", connection="my-project")

    assert isinstance(result, ResultSet)
    assert result.engine == "bigquery"
    assert result.columns == ["geom", "n"]
    assert result.rows == [["POINT(0 0)", 1], ["POINT(1 1)", 2]]
    assert result.row_count == 2
    # The query was passed through to the client unchanged.
    assert client.queries == ["SELECT geom, n FROM t"]


def test_execute_empty_result_is_valid() -> None:
    engine = BigQueryEngine(client=_FakeClient(columns=["geom"], rows=[]))
    result = engine.execute(query="SELECT geom FROM t WHERE 1=0", connection="p")
    assert result.row_count == 0
    assert result.rows == []
    assert result.columns == ["geom"]


def test_execute_coerces_non_json_cells_to_strings() -> None:
    from datetime import date

    engine = BigQueryEngine(
        client=_FakeClient(columns=["d"], rows=[[date(2024, 1, 2)]])
    )
    result = engine.execute(query="SELECT d FROM t", connection="p")
    assert result.rows == [["2024-01-02"]]


# --- Driver/API error -> Error_Taxonomy mapping ----------------------------


@pytest.mark.parametrize(
    ("error", "expected_category"),
    [
        (Forbidden("permission denied", code=403), ErrorCategory.AUTHENTICATION),
        (_ApiError("unauthorized", code=401), ErrorCategory.AUTHENTICATION),
        (NotFound("table not found", code=404), ErrorCategory.NOT_FOUND),
        (BadRequest("syntax error", code=400), ErrorCategory.VALIDATION),
        (RuntimeError("driver exploded"), ErrorCategory.UPSTREAM),
    ],
)
def test_execute_maps_driver_errors_onto_taxonomy(
    error: Exception, expected_category: ErrorCategory
) -> None:
    engine = BigQueryEngine(client=_FakeClient(error=error))
    with pytest.raises(Exception) as exc_info:
        engine.execute(query="SELECT 1", connection="p")
    assert exc_info.value.category is expected_category


def test_auth_error_names_credential_key_not_value() -> None:
    engine = BigQueryEngine(client=_FakeClient(error=Forbidden("nope", code=403)))
    with pytest.raises(AuthenticationError) as exc_info:
        engine.execute(query="SELECT 1", connection="p")
    err = exc_info.value
    assert err.detail is not None
    assert err.detail["mcp_json_key"] == BIGQUERY_CREDENTIALS_KEY


# --- Credential guard + loading (no driver required) -----------------------


def test_missing_credential_is_authentication_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(BIGQUERY_CREDENTIALS_KEY, raising=False)
    engine = BigQueryEngine()  # no injected client -> must build one
    with pytest.raises(AuthenticationError) as exc_info:
        engine.execute(query="SELECT 1", connection="p")
    err = exc_info.value
    assert err.detail is not None
    assert err.detail["mcp_json_key"] == BIGQUERY_CREDENTIALS_KEY


def test_load_credentials_info_parses_inline_json() -> None:
    engine = BigQueryEngine()
    info = engine._load_credentials_info('{"project_id": "p", "type": "service_account"}')
    assert info["project_id"] == "p"


def test_load_credentials_info_reads_a_file_path(tmp_path) -> None:
    key_file = tmp_path / "key.json"
    key_file.write_text(json.dumps({"project_id": "from-file"}), encoding="utf-8")
    engine = BigQueryEngine()
    info = engine._load_credentials_info(str(key_file))
    assert info["project_id"] == "from-file"


def test_load_credentials_info_rejects_malformed_json() -> None:
    engine = BigQueryEngine()
    with pytest.raises(AuthenticationError):
        engine._load_credentials_info("not json{")


def test_load_credentials_info_rejects_non_object_json() -> None:
    engine = BigQueryEngine()
    with pytest.raises(AuthenticationError):
        engine._load_credentials_info("[1, 2, 3]")


# --- Auto-wiring gate (default_warehouse_engines) --------------------------


def test_default_warehouse_engines_empty_without_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(BIGQUERY_CREDENTIALS_KEY, raising=False)
    assert default_warehouse_engines() == {}


def test_default_warehouse_engines_wires_bigquery_when_credential_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(BIGQUERY_CREDENTIALS_KEY, '{"project_id": "p"}')
    engines = default_warehouse_engines()
    assert set(engines) == {"bigquery"}
    assert isinstance(engines["bigquery"], BigQueryEngine)


def test_default_warehouse_engines_wires_bigquery_with_adc_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADC opt-in (no key) also wires the engine — for org-blocked SA keys."""
    monkeypatch.setenv(BIGQUERY_USE_ADC_KEY, "1")
    engines = default_warehouse_engines()
    assert set(engines) == {"bigquery"}


def test_default_warehouse_engines_wires_bigquery_with_google_app_creds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/some/adc.json")
    assert set(default_warehouse_engines()) == {"bigquery"}


def test_missing_credential_message_offers_adc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With neither a key nor ADC, the auth error points to both options."""
    engine = BigQueryEngine()
    with pytest.raises(AuthenticationError) as exc_info:
        engine.execute(query="SELECT 1", connection="p")
    message = str(exc_info.value).lower()
    assert "application-default" in message or BIGQUERY_USE_ADC_KEY.lower() in message


# --- End-to-end through the server with the fake client injected -----------


async def test_server_runs_bigquery_via_injected_engine() -> None:
    client = _FakeClient(columns=["n"], rows=[[1]])
    server = GeoWarehouseServer(engines={"bigquery": BigQueryEngine(client=client)})
    result = await server.warehouse_spatial_sql(
        query="SELECT 1 AS n", engine="bigquery", connection="my-project"
    )
    assert result.engine == "bigquery"
    assert result.rows == [[1]]
    assert client.queries == ["SELECT 1 AS n"]

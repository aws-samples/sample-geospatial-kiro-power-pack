"""Layer-1 tests for the concrete Athena query engine (no boto3, no account).

The :class:`~geo_query.athena_engine.AthenaEngine` is exercised against a **fake
Athena client** injected into the engine, so these tests verify the
start->poll->collect lifecycle, the header-row handling and NULL coercion in
result paging, the credential guard (missing ``ATHENA_S3_STAGING_DIR``), the
auto-wiring gate, and the botocore-error -> ``Error_Taxonomy`` mapping
**without** installing boto3 or contacting AWS. End-to-end verification against a
real Athena workgroup is a manual step documented in
``docs/testing-workflows.md``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from geo_common.errors import (
    AuthenticationError,
    ErrorCategory,
    NetworkError,
    NotFoundError,
    UpstreamError,
    ValidationError,
)

from geo_query import (
    ATHENA_S3_STAGING_DIR_KEY,
    AthenaEngine,
    GeoQueryServer,
    default_athena_engines,
)
from geo_query.athena_engine import AWS_ACCESS_KEY_ID_KEY
from geo_query.models import ResultSet


# --- Fake Athena boto3 client ----------------------------------------------


def _cell(value: Optional[str]) -> Dict[str, Any]:
    return {} if value is None else {"VarCharValue": value}


def _row(values: List[Optional[str]]) -> Dict[str, Any]:
    return {"Data": [_cell(v) for v in values]}


class _FakeAthenaClient:
    """A minimal in-memory stand-in for boto3's Athena client.

    Drives a single query through ``start_query_execution`` ->
    ``get_query_execution`` (returning ``states`` in order) ->
    ``get_query_results`` (returning ``pages``). ``start_error`` lets a test
    raise from the very first call to exercise error mapping.
    """

    def __init__(
        self,
        *,
        states: Optional[List[str]] = None,
        state_reason: Optional[str] = None,
        columns: Optional[List[str]] = None,
        data_rows: Optional[List[List[Optional[str]]]] = None,
        pages: Optional[List[Dict[str, Any]]] = None,
        start_error: Optional[Exception] = None,
    ) -> None:
        self._states = list(states or ["SUCCEEDED"])
        self._state_reason = state_reason
        self._columns = columns or []
        self._data_rows = data_rows or []
        self._pages = pages
        self._start_error = start_error
        self.started: List[Dict[str, Any]] = []
        self._poll_index = 0

    def start_query_execution(self, **kwargs: Any) -> Dict[str, Any]:
        if self._start_error is not None:
            raise self._start_error
        self.started.append(kwargs)
        return {"QueryExecutionId": "q-123"}

    def get_query_execution(self, *, QueryExecutionId: str) -> Dict[str, Any]:
        index = min(self._poll_index, len(self._states) - 1)
        state = self._states[index]
        self._poll_index += 1
        status: Dict[str, Any] = {"State": state}
        if self._state_reason is not None:
            status["StateChangeReason"] = self._state_reason
        return {"QueryExecution": {"Status": status}}

    def get_query_results(self, **kwargs: Any) -> Dict[str, Any]:
        if self._pages is not None:
            token = kwargs.get("NextToken")
            # Pages are pre-built dicts already shaped like Athena responses.
            if token is None:
                return self._pages[0]
            return self._pages[int(token)]
        # Single page: header row first, then data rows.
        header = _row(self._columns)
        rows = [header] + [_row(r) for r in self._data_rows]
        return {
            "ResultSet": {
                "ResultSetMetadata": {
                    "ColumnInfo": [{"Name": c, "Label": c} for c in self._columns]
                },
                "Rows": rows,
            }
        }


def _engine(client: _FakeAthenaClient, **kwargs: Any) -> AthenaEngine:
    # poll_interval 0 + injected no-op sleep so polling never really waits.
    return AthenaEngine(
        client=client,
        s3_staging_dir="s3://amzn-s3-demo-staging/prefix/",
        poll_interval_s=0,
        sleep=lambda _s: None,
        **kwargs,
    )


# --- start -> poll -> collect lifecycle ------------------------------------


def test_execute_returns_result_set_skipping_header_row() -> None:
    client = _FakeAthenaClient(
        states=["SUCCEEDED"],
        columns=["geom", "n"],
        data_rows=[["POINT(0 0)", "1"], ["POINT(1 1)", "2"]],
    )
    result = _engine(client).execute(query="SELECT geom, n FROM t")
    assert isinstance(result, ResultSet)
    assert result.engine == "athena"
    assert result.columns == ["geom", "n"]
    assert result.rows == [["POINT(0 0)", "1"], ["POINT(1 1)", "2"]]
    assert result.row_count == 2
    # The staging output location was passed to start_query_execution.
    assert client.started[0]["ResultConfiguration"]["OutputLocation"] == "s3://amzn-s3-demo-staging/prefix/"


def test_execute_sets_database_context_when_provided() -> None:
    client = _FakeAthenaClient(columns=["n"], data_rows=[["1"]])
    _engine(client, database="my_db").execute(query="SELECT 1 n")
    assert client.started[0]["QueryExecutionContext"] == {"Database": "my_db"}


def test_execute_omits_database_context_when_absent() -> None:
    client = _FakeAthenaClient(columns=["n"], data_rows=[["1"]])
    _engine(client).execute(query="SELECT 1 n")
    assert "QueryExecutionContext" not in client.started[0]


def test_execute_polls_until_succeeded() -> None:
    client = _FakeAthenaClient(
        states=["QUEUED", "RUNNING", "SUCCEEDED"],
        columns=["n"],
        data_rows=[["1"]],
    )
    result = _engine(client).execute(query="SELECT 1 n")
    assert result.rows == [["1"]]


def test_execute_coerces_null_cells_to_none() -> None:
    client = _FakeAthenaClient(columns=["a", "b"], data_rows=[["x", None]])
    result = _engine(client).execute(query="SELECT a, b FROM t")
    assert result.rows == [["x", None]]


def test_execute_pages_through_results() -> None:
    page0 = {
        "ResultSet": {
            "ResultSetMetadata": {"ColumnInfo": [{"Name": "n", "Label": "n"}]},
            "Rows": [_row(["n"]), _row(["1"])],  # header + 1 data row
        },
        "NextToken": "1",
    }
    page1 = {
        "ResultSet": {
            "ResultSetMetadata": {"ColumnInfo": [{"Name": "n", "Label": "n"}]},
            "Rows": [_row(["2"]), _row(["3"])],  # no header on later pages
        }
    }
    client = _FakeAthenaClient(pages=[page0, page1])
    result = _engine(client).execute(query="SELECT n FROM t")
    assert result.columns == ["n"]
    assert result.rows == [["1"], ["2"], ["3"]]


# --- terminal failure states ----------------------------------------------


def test_failed_query_with_syntax_reason_is_validation_error() -> None:
    client = _FakeAthenaClient(
        states=["FAILED"], state_reason="line 1:8: mismatched input 'FROM'"
    )
    with pytest.raises(ValidationError) as exc_info:
        _engine(client).execute(query="SELECT FROM t")
    assert exc_info.value.category is ErrorCategory.VALIDATION


def test_failed_query_with_missing_table_is_not_found() -> None:
    client = _FakeAthenaClient(
        states=["FAILED"], state_reason="Table awsdatacatalog.db.t does not exist"
    )
    with pytest.raises(NotFoundError):
        _engine(client).execute(query="SELECT * FROM t")


def test_failed_query_other_reason_is_upstream() -> None:
    client = _FakeAthenaClient(states=["FAILED"], state_reason="Internal error")
    with pytest.raises(UpstreamError):
        _engine(client).execute(query="SELECT 1")


def test_poll_timeout_is_network_error() -> None:
    client = _FakeAthenaClient(states=["RUNNING"])  # never terminal
    engine = AthenaEngine(
        client=client,
        s3_staging_dir="s3://amzn-s3-demo-staging/",
        poll_interval_s=0,
        max_polls=3,
        sleep=lambda _s: None,
    )
    with pytest.raises(NetworkError):
        engine.execute(query="SELECT 1")


# --- botocore error mapping ------------------------------------------------


class _ClientError(Exception):
    def __init__(self, code: str, http_status: int, message: str = "boom") -> None:
        super().__init__(message)
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": http_status},
        }


class NoCredentialsError(Exception):
    pass


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_ClientError("AccessDeniedException", 403), ErrorCategory.AUTHENTICATION),
        (NoCredentialsError("unable to locate credentials"), ErrorCategory.AUTHENTICATION),
        (_ClientError("InvalidRequestException", 400), ErrorCategory.VALIDATION),
        (_ClientError("ResourceNotFoundException", 404), ErrorCategory.NOT_FOUND),
        (_ClientError("ThrottlingException", 500), ErrorCategory.UPSTREAM),
    ],
)
def test_start_error_maps_onto_taxonomy(
    error: Exception, expected: ErrorCategory
) -> None:
    client = _FakeAthenaClient(start_error=error)
    with pytest.raises(Exception) as exc_info:
        _engine(client).execute(query="SELECT 1")
    assert exc_info.value.category is expected


def test_auth_error_names_aws_key_not_value() -> None:
    client = _FakeAthenaClient(start_error=_ClientError("AccessDenied", 403))
    with pytest.raises(AuthenticationError) as exc_info:
        _engine(client).execute(query="SELECT 1")
    err = exc_info.value
    assert err.detail is not None
    assert err.detail["mcp_json_key"] == AWS_ACCESS_KEY_ID_KEY


# --- credential guard (no staging dir) -------------------------------------


def test_missing_staging_dir_is_authentication_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ATHENA_S3_STAGING_DIR_KEY, raising=False)
    client = _FakeAthenaClient(columns=["n"], data_rows=[["1"]])
    engine = AthenaEngine(client=client, poll_interval_s=0, sleep=lambda _s: None)
    with pytest.raises(AuthenticationError) as exc_info:
        engine.execute(query="SELECT 1")
    err = exc_info.value
    assert err.detail is not None
    assert err.detail["mcp_json_key"] == ATHENA_S3_STAGING_DIR_KEY


# --- auto-wiring gate ------------------------------------------------------


def test_default_athena_engines_empty_without_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(AWS_ACCESS_KEY_ID_KEY, raising=False)
    assert default_athena_engines() == {}


def test_default_athena_engines_wires_athena_when_credential_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(AWS_ACCESS_KEY_ID_KEY, "AKIA-test")
    engines = default_athena_engines()
    assert set(engines) == {"athena"}
    assert isinstance(engines["athena"], AthenaEngine)


# --- end-to-end through the server -----------------------------------------


async def test_server_runs_athena_via_injected_engine() -> None:
    client = _FakeAthenaClient(columns=["n"], data_rows=[["1"]])
    server = GeoQueryServer(engines={"athena": _engine(client)})
    result = await server.spatial_sql(query="SELECT 1 AS n", engine="athena")
    assert result.engine == "athena"
    assert result.rows == [["1"]]

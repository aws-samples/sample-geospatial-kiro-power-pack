"""The Amazon Athena query engine for ``geo-query`` (credentialed, AWS).

:class:`AthenaEngine` is a concrete :class:`~geo_query.engine.QueryEngine` that
runs ad-hoc spatial SQL over data in S3 via Amazon Athena and returns a uniform
:class:`~geo_query.models.ResultSet`. It is the credentialed counterpart to the
open in-process DuckDB engine.

Athena (boto3) is an **optional** dependency (the ``[athena]`` extra). It is
imported lazily so the rest of ``geo-query`` - and its test suite, which injects
a fake Athena client - never requires boto3 to be installed.

Credential surface (the single ``mcp.json`` surface, the environment): boto3
reads the standard ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` /
``AWS_REGION`` keys, and Athena additionally needs ``ATHENA_S3_STAGING_DIR`` (the
``s3://`` location where query results are written). An optional
``ATHENA_DATABASE`` sets the default Glue database for unqualified table names.
:func:`default_athena_engines` wires this engine when ``AWS_ACCESS_KEY_ID`` is
configured, so the server's ``main()`` offers Athena once AWS is set up;
otherwise the engine is not configured and the server denies ``spatial_sql``
with engine ``athena`` via an ``AuthenticationError`` naming the key.

Note: unlike DuckDB, Athena reads tables from the AWS Glue Data Catalog, not
from ad-hoc file paths, so the ``sources`` mapping is **not** used by this
engine (queries reference catalog ``database.table`` names directly).
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Mapping, Optional

from geo_common.errors import (
    AuthenticationError,
    GeoError,
    NetworkError,
    NotFoundError,
    UpstreamError,
    ValidationError,
)

from geo_query.engine import QueryEngine
from geo_query.models import ResultSet

__all__ = [
    "AthenaEngine",
    "default_athena_engines",
    "ATHENA_S3_STAGING_DIR_KEY",
    "AWS_ACCESS_KEY_ID_KEY",
]

#: ``mcp.json`` key for the S3 location Athena writes query results to.
ATHENA_S3_STAGING_DIR_KEY = "ATHENA_S3_STAGING_DIR"

#: Primary AWS credential key (named in the credential-deny message).
AWS_ACCESS_KEY_ID_KEY = "AWS_ACCESS_KEY_ID"

#: Optional ``mcp.json`` key naming the default Glue database for the session.
ATHENA_DATABASE_KEY = "ATHENA_DATABASE"

#: Terminal Athena query states.
_SUCCEEDED = "SUCCEEDED"
_FAILED = "FAILED"
_CANCELLED = "CANCELLED"


class AthenaEngine(QueryEngine):
    """Execute spatial SQL over data in S3 via Amazon Athena.

    On each query the engine starts an Athena execution (writing results to the
    configured S3 staging dir), polls until the query reaches a terminal state,
    and returns the column names + rows as a :class:`ResultSet` attributed to
    ``athena``. A boto3 Athena client may be injected (``client=...``) so the
    engine is exercisable without boto3 or a real account; otherwise a client is
    built lazily from the environment credentials.
    """

    name = "athena"

    def __init__(
        self,
        *,
        client: Optional[Any] = None,
        s3_staging_dir: Optional[str] = None,
        database: Optional[str] = None,
        region: Optional[str] = None,
        poll_interval_s: float = 1.0,
        max_polls: int = 600,
        sleep: Any = time.sleep,
    ) -> None:
        self._client = client
        self._s3_staging_dir = s3_staging_dir
        self._database = database
        self._region = region
        self._poll_interval_s = float(poll_interval_s)
        self._max_polls = int(max_polls)
        self._sleep = sleep

    def execute(
        self, *, query: str, sources: Optional[Mapping[str, str]] = None
    ) -> ResultSet:
        client = self._client or self._build_client()
        staging = self._s3_staging_dir or os.environ.get(ATHENA_S3_STAGING_DIR_KEY)
        if not staging or not str(staging).strip():
            raise AuthenticationError(
                "Amazon Athena is not fully configured: set %s in mcp.json (the "
                "S3 location for query results)" % ATHENA_S3_STAGING_DIR_KEY,
                source=self.name,
                detail={"engine": self.name, "mcp_json_key": ATHENA_S3_STAGING_DIR_KEY},
            )
        database = self._database or os.environ.get(ATHENA_DATABASE_KEY)

        try:
            query_id = self._start(client, query=query, staging=staging, database=database)
            self._wait_for_completion(client, query_id=query_id)
            return self._collect_results(client, query_id=query_id)
        except GeoError:
            raise
        except Exception as exc:  # noqa: BLE001 - boto3/botocore error
            raise self._classify(exc) from exc

    def _build_client(self) -> Any:
        """Build a boto3 Athena client, or raise an actionable taxonomy error.

        A missing driver raises an ``UpstreamError`` pointing at the
        ``geo-query[athena]`` extra. boto3 resolves AWS credentials from the
        environment itself; absent credentials surface as an
        ``AuthenticationError`` at call time (mapped in :meth:`_classify`).
        """
        try:
            import boto3  # noqa: PLC0415 - optional dependency, imported lazily
        except ImportError as exc:  # pragma: no cover - exercised only w/o boto3
            raise UpstreamError(
                "the Athena engine requires the 'boto3' package; install the "
                "geo-query[athena] extra to enable it",
                source=self.name,
                original=str(exc),
            ) from exc
        region = (
            self._region
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
        )
        return boto3.client("athena", region_name=region)

    def _start(
        self, client: Any, *, query: str, staging: str, database: Optional[str]
    ) -> str:
        """Start an Athena query execution and return its QueryExecutionId."""
        kwargs: Dict[str, Any] = {
            "QueryString": query,
            "ResultConfiguration": {"OutputLocation": staging},
        }
        if database:
            kwargs["QueryExecutionContext"] = {"Database": database}
        response = client.start_query_execution(**kwargs)
        query_id = response.get("QueryExecutionId") if isinstance(response, Mapping) else None
        if not query_id:
            raise UpstreamError(
                "Athena did not return a QueryExecutionId",
                source=self.name,
                detail={"engine": self.name},
            )
        return str(query_id)

    def _wait_for_completion(self, client: Any, *, query_id: str) -> None:
        """Poll until the query reaches a terminal state (Req-style bounded wait).

        ``SUCCEEDED`` returns; ``FAILED``/``CANCELLED`` raise an
        ``Error_Taxonomy`` error carrying Athena's state-change reason (a syntax
        failure -> validation, otherwise upstream). Exceeding ``max_polls``
        raises a ``NetworkError`` rather than blocking forever.
        """
        for _ in range(self._max_polls):
            response = client.get_query_execution(QueryExecutionId=query_id)
            execution = response.get("QueryExecution", {}) if isinstance(response, Mapping) else {}
            status = execution.get("Status", {}) if isinstance(execution, Mapping) else {}
            state = status.get("State")
            if state == _SUCCEEDED:
                return
            if state in (_FAILED, _CANCELLED):
                reason = status.get("StateChangeReason") or "Athena query %s" % state.lower()
                raise self._terminal_error(state=state, reason=str(reason))
            self._sleep(self._poll_interval_s)
        raise NetworkError(
            "Athena query %r did not complete within %d polls"
            % (query_id, self._max_polls),
            source=self.name,
            detail={"engine": self.name, "query_id": query_id},
        )

    def _collect_results(self, client: Any, *, query_id: str) -> ResultSet:
        """Page through ``get_query_results`` into a :class:`ResultSet`.

        The first row of the first page is the column header (Athena's
        convention for SELECT results) and is skipped; ``NULL`` cells (a missing
        ``VarCharValue``) become ``None``.
        """
        columns: List[str] = []
        rows: List[List[Any]] = []
        next_token: Optional[str] = None
        first_page = True
        while True:
            kwargs: Dict[str, Any] = {"QueryExecutionId": query_id}
            if next_token:
                kwargs["NextToken"] = next_token
            response = client.get_query_results(**kwargs)
            result_set = response.get("ResultSet", {}) if isinstance(response, Mapping) else {}
            if not columns:
                meta = result_set.get("ResultSetMetadata", {})
                column_info = meta.get("ColumnInfo", []) if isinstance(meta, Mapping) else []
                columns = [
                    str(ci.get("Label") or ci.get("Name") or "")
                    for ci in column_info
                ]
            page_rows = result_set.get("Rows", []) or []
            start = 1 if first_page else 0
            for row in page_rows[start:]:
                data = row.get("Data", []) if isinstance(row, Mapping) else []
                rows.append([_cell_value(cell) for cell in data])
            first_page = False
            next_token = response.get("NextToken") if isinstance(response, Mapping) else None
            if not next_token:
                break
        return ResultSet.from_rows(engine=self.name, columns=columns, rows=rows)

    def _terminal_error(self, *, state: str, reason: str) -> GeoError:
        """Map a FAILED/CANCELLED Athena query onto the taxonomy."""
        lowered = reason.lower()
        if any(token in lowered for token in ("syntax", "parse", "mismatched", "cannot resolve")):
            return ValidationError(
                "Athena rejected the query: %s" % reason,
                source=self.name,
                detail={"engine": self.name, "state": state},
            )
        if "not found" in lowered or "does not exist" in lowered:
            return NotFoundError(
                "Athena could not find a referenced resource: %s" % reason,
                source=self.name,
                detail={"engine": self.name, "state": state},
            )
        return UpstreamError(
            "Athena query %s: %s" % (state.lower(), reason),
            source=self.name,
            detail={"engine": self.name, "state": state},
            original=reason,
        )

    def _classify(self, exc: Exception) -> GeoError:
        """Map a boto3/botocore failure onto the ``Error_Taxonomy``.

        Inspects the botocore error code / HTTP status without importing
        botocore: missing or rejected credentials -> authentication (naming the
        AWS key), a malformed request -> validation, a missing entity ->
        not-found, everything else -> upstream with the original message.
        """
        name = type(exc).__name__
        message = str(exc)
        code: Optional[str] = None
        http_status: Optional[int] = None
        response = getattr(exc, "response", None)
        if isinstance(response, Mapping):
            error = response.get("Error", {})
            if isinstance(error, Mapping):
                code = error.get("Code")
            meta = response.get("ResponseMetadata", {})
            if isinstance(meta, Mapping):
                http_status = meta.get("HTTPStatusCode")

        auth_codes = {
            "AccessDenied",
            "AccessDeniedException",
            "UnrecognizedClientException",
            "InvalidSignatureException",
            "AuthFailure",
            "ExpiredTokenException",
        }
        if (
            name in ("NoCredentialsError", "PartialCredentialsError")
            or code in auth_codes
            or http_status in (401, 403)
        ):
            return AuthenticationError(
                "Amazon Athena rejected the AWS credentials configured in mcp.json",
                source=self.name,
                detail={"engine": self.name, "mcp_json_key": AWS_ACCESS_KEY_ID_KEY},
                original=message,
            )
        if code in ("InvalidRequestException",) or http_status == 400:
            return ValidationError(
                "Athena rejected the request: %s" % message,
                source=self.name,
                detail={"engine": self.name},
            )
        if code in ("ResourceNotFoundException", "EntityNotFoundException") or http_status == 404:
            return NotFoundError(
                "Athena could not find a referenced resource: %s" % message,
                source=self.name,
                detail={"engine": self.name},
            )
        return UpstreamError(
            "Athena query failed: %s" % message,
            source=self.name,
            detail={"engine": self.name},
            original=message,
        )


def _cell_value(cell: Any) -> Any:
    """Extract a single Athena result cell value (missing VarCharValue -> None)."""
    if isinstance(cell, Mapping):
        return cell.get("VarCharValue")
    return None


def default_athena_engines() -> "Dict[str, QueryEngine]":
    """Return the auto-wired Athena engine set used by the server's ``main()``.

    Gates on **credential presence**: the Athena engine is wired only when
    :data:`AWS_ACCESS_KEY_ID_KEY` is configured in the environment. Absent the
    credential the engine is not configured, so the server denies ``spatial_sql``
    with engine ``athena`` via an ``AuthenticationError`` naming the key (the
    documented healthy refusal). The boto3 driver is imported lazily at query
    time, so a configured-but-driver-missing deployment surfaces an actionable
    ``UpstreamError`` pointing at the ``geo-query[athena]`` extra.
    """
    if not os.environ.get(AWS_ACCESS_KEY_ID_KEY):
        return {}
    return {"athena": AthenaEngine()}

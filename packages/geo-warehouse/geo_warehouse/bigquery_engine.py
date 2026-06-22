"""The BigQuery warehouse engine for ``geo-warehouse`` (credentialed, License-Needed).

:class:`BigQueryEngine` is a concrete :class:`~geo_warehouse.engine.WarehouseEngine`
that runs warehouse-scale spatial SQL (BigQuery GIS) against Google BigQuery and
returns a uniform :class:`~geo_warehouse.models.ResultSet`. It was the first
shipped concrete warehouse engine; Redshift, Snowflake, and Databricks are now
concrete engines too.

BigQuery is an **optional** dependency (the ``[bigquery]`` extra:
``google-cloud-bigquery``). It is imported lazily so the rest of
``geo-warehouse`` - and its test suite, which injects a fake client or a mock
engine - never requires the driver to be installed.

Credential surface (the single ``mcp.json`` surface, the environment):
``BIGQUERY_CREDENTIALS`` holds either the path to a service-account JSON key
file *or* the service-account JSON itself. The ``connection`` argument passed to
``warehouse_spatial_sql`` is the **GCP project id** the query runs in (billing
project). Alternatively, set ``BIGQUERY_USE_ADC=1`` (or the standard
``GOOGLE_APPLICATION_CREDENTIALS``) to authenticate with **Application Default
Credentials** - e.g. ``gcloud auth application-default login`` - which uses an
ambient user login instead of a downloadable key, the secure path when an
organization blocks service-account key creation
(``iam.disableServiceAccountKeyCreation``). :func:`default_warehouse_engines`
wires this engine when ``BIGQUERY_CREDENTIALS`` is configured *or* ADC is
requested, so the server's ``main()`` runs BigQuery once either is set and the
extra is installed; absent both, the engine is not configured and the server
denies the call with an ``AuthenticationError`` naming the options.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from geo_common.errors import (
    AuthenticationError,
    GeoError,
    NotFoundError,
    UpstreamError,
    ValidationError,
)

from geo_warehouse.engine import WarehouseEngine
from geo_warehouse.models import ResultSet

__all__ = [
    "BigQueryEngine",
    "bigquery_engine_if_configured",
    "BIGQUERY_USE_ADC_KEY",
]

#: The ``mcp.json`` key holding the BigQuery service-account credential (path or
#: inline JSON). Mirrors ``bundle-manifest.json`` and the ``EngineSpec``.
BIGQUERY_CREDENTIALS_KEY = "BIGQUERY_CREDENTIALS"

#: Opt-in flag to authenticate via Application Default Credentials (ADC) instead
#: of a service-account key - e.g. ``gcloud auth application-default login``.
#: Set this (truthy) when your organization blocks service-account key creation
#: (``iam.disableServiceAccountKeyCreation``). The standard
#: ``GOOGLE_APPLICATION_CREDENTIALS`` env also implies ADC.
BIGQUERY_USE_ADC_KEY = "BIGQUERY_USE_ADC"


def _adc_requested() -> bool:
    """Return True when Application Default Credentials should be used.

    True if ``GOOGLE_APPLICATION_CREDENTIALS`` is set (the standard ADC env) or
    ``BIGQUERY_USE_ADC`` is truthy. ADC uses ambient credentials (a user login
    from ``gcloud auth application-default login`` or an attached service
    account) rather than a downloadable key file.
    """
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return True
    return os.environ.get(BIGQUERY_USE_ADC_KEY, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class BigQueryEngine(WarehouseEngine):
    """Execute spatial SQL against Google BigQuery (BigQuery GIS).

    On each query a BigQuery client is built for the ``connection`` project
    (using the service-account credential from :data:`BIGQUERY_CREDENTIALS_KEY`),
    the query is run, and the column names + rows are returned as a
    :class:`ResultSet` attributed to ``bigquery``. A client may be injected
    (``client=...``) so the engine is exercisable without the driver or a real
    account; otherwise a client is built lazily from the environment credential.
    """

    name = "bigquery"

    def __init__(
        self,
        *,
        client: Optional[Any] = None,
        credentials_env: str = BIGQUERY_CREDENTIALS_KEY,
    ) -> None:
        self._client = client
        self._credentials_env = credentials_env

    def execute(self, *, query: str, connection: str) -> ResultSet:
        client = self._client or self._build_client(project=connection)
        try:
            job = client.query(query)
            result = job.result()
            schema = list(getattr(result, "schema", []) or [])
            columns = [field.name for field in schema]
            rows: List[List[Any]] = []
            for row in result:
                rows.append([_jsonable(value) for value in _row_values(row, columns)])
        except GeoError:
            raise
        except Exception as exc:  # noqa: BLE001 - BigQuery driver/API error
            raise self._classify(exc) from exc
        return ResultSet.from_rows(engine=self.name, columns=columns, rows=rows)

    def _build_client(self, *, project: str) -> Any:
        """Build a BigQuery client from a key or Application Default Credentials.

        Resolution order: a configured service-account key in
        :data:`BIGQUERY_CREDENTIALS_KEY` is used when present; otherwise, if ADC
        is requested (:func:`_adc_requested`), ambient Application Default
        Credentials are used (e.g. from ``gcloud auth application-default
        login``) with no key file - the secure path when an organization blocks
        service-account key creation. When neither is configured the call raises
        an ``AuthenticationError`` naming both options (before importing the
        driver). A missing driver raises an ``UpstreamError`` pointing at the
        ``geo-warehouse[bigquery]`` extra.
        """
        raw = os.environ.get(self._credentials_env)
        has_key = bool(raw and raw.strip())
        use_adc = _adc_requested()
        if not has_key and not use_adc:
            raise AuthenticationError(
                "Google BigQuery is not configured: set %s in mcp.json to a "
                "service-account JSON (path or inline), or use Application "
                "Default Credentials by running `gcloud auth application-default "
                "login` and setting %s=1 (recommended when your org blocks "
                "service-account keys)"
                % (self._credentials_env, BIGQUERY_USE_ADC_KEY),
                source=self.name,
                detail={"engine": self.name, "mcp_json_key": self._credentials_env},
            )

        try:
            from google.cloud import bigquery  # noqa: PLC0415 - optional dep
            from google.oauth2 import service_account  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised only w/o driver
            raise UpstreamError(
                "the BigQuery engine requires the 'google-cloud-bigquery' "
                "package; install the geo-warehouse[bigquery] extra to enable it",
                source=self.name,
                original=str(exc),
            ) from exc

        if has_key:
            info = self._load_credentials_info(raw)
            try:
                credentials = service_account.Credentials.from_service_account_info(info)
            except Exception as exc:  # noqa: BLE001 - malformed service-account JSON
                raise AuthenticationError(
                    "%s does not contain a valid BigQuery service-account credential"
                    % self._credentials_env,
                    source=self.name,
                    detail={
                        "engine": self.name,
                        "mcp_json_key": self._credentials_env,
                    },
                    original=str(exc),
                ) from exc
            return bigquery.Client(
                project=(project or info.get("project_id")),
                credentials=credentials,
            )

        # Application Default Credentials path (no key file).
        try:
            return bigquery.Client(project=(project or None))
        except Exception as exc:  # noqa: BLE001 - google.auth DefaultCredentialsError etc.
            raise AuthenticationError(
                "Google BigQuery could not load Application Default Credentials; "
                "run `gcloud auth application-default login` (and, if needed, "
                "`gcloud auth application-default set-quota-project <project>`), "
                "or set %s to a service-account JSON" % self._credentials_env,
                source=self.name,
                detail={"engine": self.name, "mcp_json_key": self._credentials_env},
                original=str(exc),
            ) from exc

    def _load_credentials_info(self, raw: str) -> Dict[str, Any]:
        """Parse ``BIGQUERY_CREDENTIALS`` as inline JSON or a path to a JSON file."""
        text = raw.strip()
        if not text.startswith("{") and os.path.isfile(text):
            try:
                with open(text, "r", encoding="utf-8") as handle:
                    text = handle.read()
            except OSError as exc:
                raise AuthenticationError(
                    "could not read the BigQuery credential file named by %s"
                    % self._credentials_env,
                    source=self.name,
                    detail={
                        "engine": self.name,
                        "mcp_json_key": self._credentials_env,
                    },
                    original=str(exc),
                ) from exc
        try:
            info = json.loads(text)
        except ValueError as exc:
            raise AuthenticationError(
                "%s is not valid JSON (expected a service-account key or a path "
                "to one)" % self._credentials_env,
                source=self.name,
                detail={"engine": self.name, "mcp_json_key": self._credentials_env},
                original=str(exc),
            ) from exc
        if not isinstance(info, dict):
            raise AuthenticationError(
                "%s must be a service-account JSON object" % self._credentials_env,
                source=self.name,
                detail={"engine": self.name, "mcp_json_key": self._credentials_env},
            )
        return info

    def _classify(self, exc: Exception) -> GeoError:
        """Map a BigQuery driver/API failure onto the ``Error_Taxonomy``.

        Uses the exception's HTTP-ish ``code`` and class name (without importing
        the optional ``google.api_core`` types): 401/403 -> authentication
        (naming the credential), 404 -> not-found, 400 -> validation (a bad
        query), everything else -> upstream with the driver's own message
        retained. Messages are the driver's own and secret-free.
        """
        message = str(exc)
        code = getattr(exc, "code", None)
        name = type(exc).__name__

        if code in (401, 403) or name in ("Unauthorized", "Forbidden"):
            return AuthenticationError(
                "Google BigQuery rejected the credential configured in %s"
                % self._credentials_env,
                source=self.name,
                detail={"engine": self.name, "mcp_json_key": self._credentials_env},
                original=message,
            )
        if code == 404 or name == "NotFound":
            return NotFoundError(
                "BigQuery could not find a referenced resource: %s" % message,
                source=self.name,
                detail={"engine": self.name},
            )
        if code == 400 or name in ("BadRequest", "BadQuery"):
            return ValidationError(
                "BigQuery rejected the query: %s" % message,
                source=self.name,
                detail={"engine": self.name},
            )
        return UpstreamError(
            "BigQuery query failed: %s" % message,
            source=self.name,
            detail={"engine": self.name},
            original=message,
        )


def _row_values(row: Any, columns: List[str]) -> List[Any]:
    """Extract a BigQuery row's values in column order (tolerating row shapes)."""
    values = getattr(row, "values", None)
    if callable(values):
        try:
            return list(row.values())
        except Exception:  # noqa: BLE001 - fall back to per-column access
            pass
    return [_row_get(row, column) for column in columns]


def _row_get(row: Any, column: str) -> Any:
    """Best-effort single-cell access for a BigQuery row."""
    try:
        return row[column]
    except Exception:  # noqa: BLE001
        getter = getattr(row, "get", None)
        if callable(getter):
            return getter(column)
        return None


def _jsonable(value: Any) -> Any:
    """Coerce a BigQuery cell to a JSON-serializable value (best effort)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def bigquery_engine_if_configured() -> "Optional[WarehouseEngine]":
    """Return a :class:`BigQueryEngine` when BigQuery is configured, else ``None``.

    "Configured" means **either** a service-account credential
    (:data:`BIGQUERY_CREDENTIALS_KEY`) is set **or** Application Default
    Credentials are requested (:func:`_adc_requested` - e.g. ``BIGQUERY_USE_ADC=1``
    after ``gcloud auth application-default login``, the secure path when an org
    blocks service-account keys). The driver is imported lazily at query time.
    """
    raw = os.environ.get(BIGQUERY_CREDENTIALS_KEY)
    if (raw and raw.strip()) or _adc_requested():
        return BigQueryEngine()
    return None

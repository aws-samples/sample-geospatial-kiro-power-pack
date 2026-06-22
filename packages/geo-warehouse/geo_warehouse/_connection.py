"""Shared connection + error helpers for the concrete warehouse engines.

The proprietary warehouse engines (BigQuery, Redshift, Snowflake, Databricks)
follow one **consistent, security-first** connection convention:

* The engine's ``*_CONNECTION`` ``mcp.json`` key holds the connection
  parameters as a **JSON object** *or a path to a JSON file* (so the secret can
  live in a ``0600`` file rather than inline in ``mcp.json``). It is the only
  secret surface and is never echoed back.
* The per-call ``connection`` argument is the **non-secret target selector**
  (the database/schema), matching how BigQuery takes the project id - it never
  carries secrets, so it is safe even if a caller logs tool arguments.

This module centralizes the parsing (:func:`load_connection_config`), the
JSON-safe cell coercion (:func:`jsonable`), and the failure -> ``Error_Taxonomy``
mapping (:func:`classify_warehouse_error`) so every engine behaves identically
and never leaks a credential value into an error message.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from geo_common.errors import (
    AuthenticationError,
    GeoError,
    NotFoundError,
    UpstreamError,
    ValidationError,
)

__all__ = ["load_connection_config", "jsonable", "classify_warehouse_error"]


def load_connection_config(raw: str, *, env_key: str, source: str) -> Dict[str, Any]:
    """Parse a ``*_CONNECTION`` value as inline JSON or a path to a JSON file.

    Returns the connection-parameter object. A value that cannot be read or is
    not a JSON object raises an :class:`~geo_common.errors.AuthenticationError`
    naming ``env_key`` (never the value), so a malformed credential surfaces as
    an authentication problem rather than leaking content.
    """
    text = raw.strip()
    if not text.startswith("{") and os.path.isfile(text):
        try:
            with open(text, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            raise AuthenticationError(
                "could not read the connection file named by %s" % env_key,
                source=source,
                detail={"engine": source, "mcp_json_key": env_key},
                original=str(exc),
            ) from exc
    try:
        config = json.loads(text)
    except ValueError as exc:
        raise AuthenticationError(
            "%s is not valid JSON (expected a connection object or a path to "
            "one)" % env_key,
            source=source,
            detail={"engine": source, "mcp_json_key": env_key},
            original=str(exc),
        ) from exc
    if not isinstance(config, dict):
        raise AuthenticationError(
            "%s must be a JSON object of connection parameters" % env_key,
            source=source,
            detail={"engine": source, "mcp_json_key": env_key},
        )
    return config


def jsonable(value: Any) -> Any:
    """Coerce a database cell to a JSON-serializable value (best effort)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


# Substring patterns (lower-cased) used to classify a driver/DBAPI error. Order
# matters: a missing object often co-occurs with "not authorized", so not-found
# is checked before authentication to avoid mislabeling it as an auth failure.
_VALIDATION_PATTERNS = (
    "syntax error",
    "parse error",
    "invalid identifier",
    "select list must not be empty",
)
_NOT_FOUND_PATTERNS = (
    "does not exist",
    "not found",
    "undefined table",
    "undefined column",
    "unknown database",
    "unknown table",
    "unknown column",
    "cannot be found",
    "cannot be resolved",
    "unresolved_column",
    "no such",
)
_AUTH_PATTERNS = (
    "authentication",
    "password",
    "credential",
    "not authorized",
    "access denied",
    "permission denied",
    "invalid username",
    "incorrect username",
    "403",
    "401",
)


def classify_warehouse_error(
    exc: Exception, *, source: str, auth_key: str
) -> GeoError:
    """Map a driver/DBAPI failure onto the ``Error_Taxonomy`` (consistent across engines).

    An already-classified :class:`~geo_common.errors.GeoError` passes through.
    Otherwise the (secret-free) message is matched against known patterns:
    syntax -> ``VALIDATION``, missing object -> ``NOT_FOUND``, auth/permission ->
    ``AUTHENTICATION`` (naming ``auth_key``), anything else -> ``UPSTREAM``. The
    engine's own message is retained; credential values never appear here because
    drivers do not echo them into exception text.
    """
    if isinstance(exc, GeoError):
        return exc
    message = str(exc).strip() or type(exc).__name__
    lowered = message.lower()

    if any(p in lowered for p in _VALIDATION_PATTERNS):
        return ValidationError(
            "%s rejected the query: %s" % (source, message),
            source=source,
            detail={"engine": source},
        )
    if any(p in lowered for p in _NOT_FOUND_PATTERNS):
        return NotFoundError(
            "%s could not find a referenced object: %s" % (source, message),
            source=source,
            detail={"engine": source},
        )
    if any(p in lowered for p in _AUTH_PATTERNS):
        return AuthenticationError(
            "%s rejected the credentials configured in %s" % (source, auth_key),
            source=source,
            detail={"engine": source, "mcp_json_key": auth_key},
            original=message,
        )
    return UpstreamError(
        "%s query failed: %s" % (source, message),
        source=source,
        detail={"engine": source},
        original=message,
    )

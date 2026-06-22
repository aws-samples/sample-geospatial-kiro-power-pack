"""Sentinel Hub TPDI connector for ``geo-commercial-imagery`` — **Maxar only**.

Sentinel Hub's Third-Party Data Import (TPDI) API is being wound down for Planet
data (Planet migrated to its own Orders/Subscriptions APIs - see
:mod:`geo_commercial_imagery.planet`), and Airbus is no longer offered through
TPDI. Per Sentinel Hub / Planet guidance, **Maxar WorldView remains on TPDI**, so
this connector now serves Maxar only:

* :func:`search_commercial` - ``POST /api/v1/dataimport/search`` to discover
  Maxar scenes for a bbox + time range, returned as a list of
  :class:`~geo_commercial_imagery.models.StacItem` (ids from ``catalogID``).
* :func:`order_scene` - ``POST /api/v1/dataimport/orders`` to create an order
  for one or more product ids (optionally followed by
  ``POST /api/v1/dataimport/orders/{id}/confirm``), returned as an
  :class:`~geo_commercial_imagery.models.OrderConfirmation`.

Sentinel Hub authenticates with an OAuth2 *client-credentials* flow: the
``SENTINELHUB_CLIENT_ID`` / ``SENTINELHUB_CLIENT_SECRET`` pair is exchanged for a
bearer token, which authorizes the TPDI calls. Both halves of the credential are
classified License-Needed (``bundle-manifest.json``).

Two cross-cutting rules apply to every call (Requirement 7.8):

* **Credential guard.** Each capability requires *both* Sentinel Hub
  credentials. When either is absent the call raises an
  :class:`~geo_common.errors.AuthenticationError` *naming* the missing
  ``mcp.json`` key(s) **before any network I/O**.
* **Invalid-credential mapping.** Present-but-rejected credentials map onto an
  :class:`~geo_common.errors.AuthenticationError` naming the credential, with
  **no items / no order** returned.

Ordering safety: creating a TPDI order does **not** spend account quota;
*confirming* it does. :func:`order_scene` defaults to ``confirm=False``.
Sentinel Hub's own error message is surfaced on any upstream rejection so a 400
explains *why* it was refused rather than only "HTTP 400".
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlencode

import httpx

from geo_common.errors import (
    AuthenticationError,
    GeoError,
    NotFoundError,
    UpstreamError,
    ValidationError,
)
from geo_common.http import HttpClient

from geo_commercial_imagery.models import BBox, OrderConfirmation, StacItem

__all__ = [
    "SENTINELHUB_CLIENT_ID_KEY",
    "SENTINELHUB_CLIENT_SECRET_KEY",
    "CREDENTIAL_KEYS",
    "PROVIDER",
    "DEFAULT_MAXAR_PRODUCT_BANDS",
    "DEFAULT_TOKEN_URL",
    "DEFAULT_SEARCH_URL",
    "DEFAULT_ORDERS_URL",
    "require_credentials",
    "fetch_token",
    "search_commercial",
    "order_scene",
]

#: The ``mcp.json`` keys holding the Sentinel Hub OAuth credential. Both are
#: named in the authentication error raised when either is missing/invalid.
SENTINELHUB_CLIENT_ID_KEY = "SENTINELHUB_CLIENT_ID"
SENTINELHUB_CLIENT_SECRET_KEY = "SENTINELHUB_CLIENT_SECRET"

#: Both credential keys, in declaration order.
CREDENTIAL_KEYS: Tuple[str, str] = (
    SENTINELHUB_CLIENT_ID_KEY,
    SENTINELHUB_CLIENT_SECRET_KEY,
)

#: The secret-free source identifier used on errors raised from this module.
_SOURCE = "geo-commercial-imagery"

#: The single TPDI provider this connector still serves.
PROVIDER = "MAXAR"

#: The default Maxar ``productBands`` when none is supplied.
DEFAULT_MAXAR_PRODUCT_BANDS = "4BB"

#: Maxar TPDI names the search-result id ``catalogID``.
_MAXAR_ID_FIELD = "catalogID"

#: Default Sentinel Hub OAuth2 token endpoint (client-credentials flow).
DEFAULT_TOKEN_URL = "https://services.sentinel-hub.com/oauth/token"

#: Default Sentinel Hub TPDI search endpoint.
DEFAULT_SEARCH_URL = "https://services.sentinel-hub.com/api/v1/dataimport/search"

#: Default Sentinel Hub TPDI orders endpoint.
DEFAULT_ORDERS_URL = "https://services.sentinel-hub.com/api/v1/dataimport/orders"

#: The maximum number of scenes returned per ``search_commercial`` response.
MAX_ITEMS = 1000


# ---------------------------------------------------------------------------
# Credential guard (Requirement 7.8)
# ---------------------------------------------------------------------------


def require_credentials(
    client_id: Optional[str],
    client_secret: Optional[str],
    *,
    request: str,
) -> Tuple[str, str]:
    """Return the credential pair, or raise an auth error naming what is missing."""
    missing = [
        key
        for key, value in (
            (SENTINELHUB_CLIENT_ID_KEY, client_id),
            (SENTINELHUB_CLIENT_SECRET_KEY, client_secret),
        )
        if not (value and str(value).strip())
    ]
    if missing:
        raise AuthenticationError(
            "geo-commercial-imagery requires configured Sentinel Hub "
            "credentials for Maxar: the %s credential(s) are missing; configure "
            "them in mcp.json" % ", ".join(repr(k) for k in missing),
            source=_SOURCE,
            detail={"missing_credentials": missing, "request": request},
        )
    return str(client_id), str(client_secret)


def _auth_error(
    request: str,
    *,
    original: Optional[str] = None,
    message: Optional[str] = None,
) -> AuthenticationError:
    """Build an auth error naming the Sentinel Hub credentials (Req 7.8)."""
    suffix = (": %s" % message) if message else ""
    return AuthenticationError(
        "Sentinel Hub rejected request %r: the %s credential(s) are missing or "
        "invalid%s" % (request, " / ".join(CREDENTIAL_KEYS), suffix),
        source=_SOURCE,
        detail={
            "request": request,
            "invalid_credentials": list(CREDENTIAL_KEYS),
        },
        original=original,
    )


# ---------------------------------------------------------------------------
# OAuth2 client-credentials token exchange
# ---------------------------------------------------------------------------


async def fetch_token(
    *,
    client_id: Optional[str],
    client_secret: Optional[str],
    http: HttpClient,
    token_url: str = DEFAULT_TOKEN_URL,
    request: str = "oauth_token",
) -> str:
    """Exchange the credential pair for a Sentinel Hub bearer token (Req 7.8)."""
    cid, secret = require_credentials(client_id, client_secret, request=request)

    form = urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": cid,
            "client_secret": secret,
        }
    ).encode("ascii")
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    try:
        response = await http.post(token_url, content=form, headers=headers)
    except AuthenticationError as exc:
        raise _auth_error(request, original=exc.original)
    except GeoError as exc:
        raise _identify_request(exc, request=request)

    payload = _decode_json(response, request=request)
    if response.status_code >= 400:
        message = _oauth_error_message(payload)
        raise _auth_error(request, original=message, message=message)

    token = payload.get("access_token") if isinstance(payload, Mapping) else None
    if not token or not isinstance(token, str):
        raise _auth_error(
            request, message="token endpoint returned no access_token"
        )
    return token


def _oauth_error_message(payload: Any) -> Optional[str]:
    """Extract a secret-free OAuth error description, if present."""
    if not isinstance(payload, Mapping):
        return None
    for key in ("error_description", "error", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


# ---------------------------------------------------------------------------
# Capability 1: search_commercial (Maxar)
# ---------------------------------------------------------------------------


async def search_commercial(
    *,
    bbox: Sequence[float],
    datetime_range: Sequence[str],
    product_bands: Optional[str] = None,
    max_cloud_coverage: Optional[float] = None,
    http: HttpClient,
    client_id: Optional[str],
    client_secret: Optional[str],
    limit: int = MAX_ITEMS,
    token_url: str = DEFAULT_TOKEN_URL,
    search_url: str = DEFAULT_SEARCH_URL,
) -> List[StacItem]:
    """Search Sentinel Hub TPDI for Maxar scenes matching a query (Req 2.1, 7.8).

    ``bbox`` is ``(west, south, east, north)`` in EPSG:4326; ``datetime_range``
    is an ISO-8601 ``(start, end)`` pair; ``product_bands`` selects the Maxar
    band product (default :data:`DEFAULT_MAXAR_PRODUCT_BANDS`);
    ``max_cloud_coverage`` (0-100) optionally caps cloud cover. All inputs are
    validated before any network call.
    """
    request = "search_commercial"
    validated_bbox = _validate_bbox(bbox)
    start, end = _validate_datetime_range(datetime_range)
    effective_limit = _validate_and_cap_limit(limit)
    cloud = _validate_max_cloud_coverage(max_cloud_coverage)
    bands = _validate_product_bands(product_bands)

    token = await fetch_token(
        client_id=client_id,
        client_secret=client_secret,
        http=http,
        token_url=token_url,
        request=request,
    )

    data_filter: Dict[str, Any] = {"timeRange": {"from": start, "to": end}}
    if cloud is not None:
        data_filter["maxCloudCoverage"] = _cloud_number(cloud)
    data_obj: Dict[str, Any] = {"productBands": bands, "dataFilter": data_filter}

    body: Dict[str, Any] = {
        "provider": PROVIDER,
        "bounds": {"geometry": _bbox_to_polygon(validated_bbox)},
        "data": [data_obj],
    }

    response = await _send_json(
        http, search_url, body=body, token=token, request=request
    )
    payload = _decode_json(response, request=request)
    _raise_for_error(response, payload, request=request)

    features = payload.get("features", []) if isinstance(payload, Mapping) else []
    if not isinstance(features, list):
        features = []

    items: List[StacItem] = []
    for feature in features:
        if len(items) >= effective_limit:
            break
        item = _feature_to_item(feature, query_bbox=validated_bbox)
        if item is not None:
            items.append(item)
    return items


# ---------------------------------------------------------------------------
# Capability 2: order_scene (Maxar)
# ---------------------------------------------------------------------------


async def order_scene(
    *,
    scene_ids: Sequence[str],
    bbox: Sequence[float],
    product_bands: Optional[str] = None,
    name: Optional[str] = None,
    confirm: bool = False,
    http: HttpClient,
    client_id: Optional[str],
    client_secret: Optional[str],
    token_url: str = DEFAULT_TOKEN_URL,
    orders_url: str = DEFAULT_ORDERS_URL,
) -> OrderConfirmation:
    """Create (and optionally confirm) a TPDI order for Maxar scenes (Req 2.1, 7.8).

    ``scene_ids`` are the ``catalogID`` values from :func:`search_commercial`;
    ``bbox`` is the order's area (TPDI requires ``bounds``). Creating an order
    does **not** spend quota; confirming it does, so ``confirm`` defaults to
    ``False`` (create-only).
    """
    request = "order_scene"
    ids = _validate_scene_ids(scene_ids)
    validated_bbox = _validate_bbox(bbox)
    bands = _validate_product_bands(product_bands)
    order_name = _resolve_order_name(name, scene_ids=ids)

    token = await fetch_token(
        client_id=client_id,
        client_secret=client_secret,
        http=http,
        token_url=token_url,
        request=request,
    )

    data_obj: Dict[str, Any] = {"productBands": bands, "selectedImages": ids}
    body: Dict[str, Any] = {
        "name": order_name,
        "input": {
            "provider": PROVIDER,
            "bounds": {"geometry": _bbox_to_polygon(validated_bbox)},
            "data": [data_obj],
        },
    }

    response = await _send_json(
        http, orders_url, body=body, token=token, request=request
    )
    payload = _decode_json(response, request=request)
    _raise_for_error(response, payload, request=request)

    order_id, status, sqkm, collection_id, created = _extract_order_fields(
        payload, request=request
    )

    confirmed = False
    if confirm:
        confirm_status = await _confirm_order(
            http, orders_url, order_id=order_id, token=token, request=request
        )
        confirmed = True
        if confirm_status:
            status = confirm_status

    return OrderConfirmation(
        order_id=order_id,
        provider=PROVIDER,
        scene_ids=ids,
        status=status,
        confirmed=confirmed,
        sqkm=sqkm,
        collection=collection_id,
        created=created,
    )


async def _confirm_order(
    http: HttpClient,
    orders_url: str,
    *,
    order_id: str,
    token: str,
    request: str,
) -> Optional[str]:
    """Confirm a created TPDI order, returning its post-confirm status if given."""
    confirm_url = "%s/%s/confirm" % (orders_url.rstrip("/"), order_id)
    response = await _send_json(
        http, confirm_url, body=None, token=token, request=request
    )
    try:
        payload = response.json()
    except ValueError:
        payload = None
    _raise_for_error(response, payload, request=request)
    if isinstance(payload, Mapping):
        status = payload.get("status")
        if isinstance(status, str) and status:
            return status
    return None


# ---------------------------------------------------------------------------
# Sentinel Hub error mapping (Requirements 7.8, 11.5)
# ---------------------------------------------------------------------------


def _raise_for_error(
    response: httpx.Response,
    payload: Any,
    *,
    request: str,
) -> None:
    """Map a Sentinel Hub failure onto the ``Error_Taxonomy`` (Req 7.8, 11.5).

    Surfaces Sentinel Hub's own error **message** (parsed from the body) in the
    taxonomy error so a malformed-query rejection says *why* it was refused
    rather than only "HTTP 400".
    """
    status = response.status_code
    http_detail = _status_detail(response)
    body_message = _body_message(payload)
    original = "; ".join(part for part in (http_detail, body_message) if part)
    suffix = (": %s" % body_message) if body_message else ""

    if status == 401:
        raise _auth_error(request, original=original)
    if status == 404:
        raise NotFoundError(
            "Sentinel Hub request %r was not found (HTTP 404)%s" % (request, suffix),
            source=_SOURCE,
            detail={"request": request, "status_code": status, "sh_message": body_message},
            original=original,
        )
    if status in (400, 422):
        raise ValidationError(
            "Sentinel Hub rejected request %r as invalid (HTTP %d)%s"
            % (request, status, suffix),
            source=_SOURCE,
            detail={"request": request, "status_code": status, "sh_message": body_message},
            original=original,
        )
    if status >= 400:
        raise UpstreamError(
            "Sentinel Hub returned HTTP %d for request %r%s" % (status, request, suffix),
            source=_SOURCE,
            detail={"request": request, "status_code": status, "sh_message": body_message},
            original=original,
        )

    if not isinstance(payload, Mapping):
        return
    error = payload.get("error")
    if not isinstance(error, Mapping):
        return
    message = str(error.get("message") or "Sentinel Hub request failed")
    code = error.get("status") or error.get("code")
    if code == 401:
        raise _auth_error(request, message=message)
    if code == 404:
        raise NotFoundError(
            "Sentinel Hub request %r was not found: %s" % (request, message),
            source=_SOURCE,
            detail={"request": request, "sh_code": code},
        )
    if code in (400, 422):
        raise ValidationError(
            "Sentinel Hub rejected request %r as invalid: %s" % (request, message),
            source=_SOURCE,
            detail={"request": request, "sh_code": code},
        )
    raise UpstreamError(
        "Sentinel Hub request %r failed: %s" % (request, message),
        source=_SOURCE,
        detail={"request": request, "sh_code": code},
    )


def _body_message(payload: Any) -> Optional[str]:
    """Extract a secret-free human message from a Sentinel Hub error body."""
    if not isinstance(payload, Mapping):
        return None
    error = payload.get("error")
    parts: List[str] = []
    if isinstance(error, Mapping):
        base = error.get("message")
        if isinstance(base, str) and base:
            parts.append(base)
        nested = error.get("errors")
        if isinstance(nested, list):
            for item in nested:
                if isinstance(item, Mapping):
                    text = (
                        item.get("message")
                        or item.get("violation")
                        or item.get("description")
                    )
                    if isinstance(text, str) and text:
                        parts.append(text)
                elif isinstance(item, str) and item:
                    parts.append(item)
        if not parts:
            reason = error.get("reason") or error.get("code")
            if isinstance(reason, str) and reason:
                parts.append(reason)
    elif isinstance(error, str) and error:
        parts.append(error)
    if not parts:
        for key in ("message", "description", "error_description"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
                break
    return " | ".join(parts) if parts else None


def _identify_request(exc: GeoError, *, request: str) -> GeoError:
    """Re-tag a taxonomy error so it identifies the failed request (Req 7.8)."""
    if isinstance(exc, AuthenticationError):
        return _auth_error(request, original=exc.original)
    detail = dict(exc.detail or {})
    detail.setdefault("request", request)
    return type(exc)(
        exc.message,
        source=_SOURCE,
        detail=detail,
        original=exc.original,
        retry_after=exc.retry_after,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _send_json(
    http: HttpClient,
    url: str,
    *,
    body: Optional[Mapping[str, Any]],
    token: str,
    request: str,
) -> httpx.Response:
    """POST ``body`` as JSON with the bearer token, normalizing failures."""
    headers = {"Authorization": "Bearer %s" % token}
    try:
        if body is None:
            return await http.post(url, headers=headers)
        return await http.post(url, json=dict(body), headers=headers)
    except GeoError as exc:
        raise _identify_request(exc, request=request)
    except Exception as exc:  # pragma: no cover - defensive catch-all
        raise UpstreamError(
            "Sentinel Hub request %r failed unexpectedly" % request,
            source=_SOURCE,
            detail={"request": request},
            original=str(exc),
        )


def _decode_json(response: httpx.Response, *, request: str) -> Any:
    """Parse a response body as JSON, or raise an upstream error (Req 11.5)."""
    try:
        return response.json()
    except ValueError as exc:
        raise UpstreamError(
            "Sentinel Hub returned a non-JSON response for request %r" % request,
            source=_SOURCE,
            detail={"request": request},
            original=str(exc),
        )


def _status_detail(response: httpx.Response) -> str:
    """A short, secret-free description of an HTTP response."""
    reason = response.reason_phrase or ""
    return ("HTTP %d %s" % (response.status_code, reason)).strip()


def _resolve_order_name(name: Optional[str], *, scene_ids: Sequence[str]) -> str:
    """Resolve the human-readable order name, validating an explicit one."""
    if name is None:
        return "geo-commercial-imagery Maxar order (%d item(s))" % len(scene_ids)
    if not isinstance(name, str) or not name.strip():
        raise ValidationError(
            "name must be a non-empty string when provided",
            source=_SOURCE,
            detail={"parameter": "name"},
        )
    return name.strip()


def _extract_order_fields(
    payload: Any, *, request: str
) -> Tuple[str, str, Optional[float], Optional[str], Optional[str]]:
    """Pull ``(order_id, status, sqkm, collectionId, created)`` from an order body."""
    if not isinstance(payload, Mapping) or not payload:
        raise UpstreamError(
            "Sentinel Hub returned an empty order response for %r" % request,
            source=_SOURCE,
            detail={"request": request},
        )
    order_id = payload.get("id") or payload.get("orderId")
    if not order_id:
        raise UpstreamError(
            "Sentinel Hub order response omitted an order id for %r" % request,
            source=_SOURCE,
            detail={"request": request},
        )
    status = payload.get("status") or "CREATED"
    sqkm = payload.get("sqkm")
    sqkm_value = (
        float(sqkm)
        if isinstance(sqkm, (int, float)) and not isinstance(sqkm, bool)
        else None
    )
    collection_id = _opt_str(payload.get("collectionId"))
    created = _opt_str(payload.get("created"))
    return str(order_id), str(status), sqkm_value, collection_id, created


def _opt_str(value: Any) -> Optional[str]:
    """Return ``str(value)`` for a non-empty value, else ``None``."""
    if value is None:
        return None
    text = str(value)
    return text if text else None


# ---------------------------------------------------------------------------
# Validation helpers (shape-compatible with geo-stac)
# ---------------------------------------------------------------------------


def _validate_product_bands(product_bands: Optional[str]) -> str:
    """Validate the Maxar ``productBands`` selector (defaulting when absent)."""
    value = product_bands if product_bands is not None else DEFAULT_MAXAR_PRODUCT_BANDS
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            "product_bands must be a non-empty Maxar productBands (e.g. '4BB')",
            source=_SOURCE,
            detail={"parameter": "product_bands"},
        )
    return value.strip()


def _validate_scene_ids(scene_ids: Sequence[str]) -> List[str]:
    """Validate ``scene_ids`` as a non-empty list of product-id strings."""
    if isinstance(scene_ids, (str, bytes)) or not isinstance(scene_ids, Sequence):
        raise ValidationError(
            "scene_ids must be a non-empty list of product id strings",
            source=_SOURCE,
            detail={"parameter": "scene_ids"},
        )
    ids: List[str] = []
    for value in scene_ids:
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(
                "each scene_id must be a non-empty string",
                source=_SOURCE,
                detail={"parameter": "scene_ids"},
            )
        ids.append(value.strip())
    if not ids:
        raise ValidationError(
            "scene_ids must contain at least one product id",
            source=_SOURCE,
            detail={"parameter": "scene_ids"},
        )
    return ids


def _validate_max_cloud_coverage(value: Any) -> Optional[float]:
    """Validate optional ``max_cloud_coverage`` as a percentage in ``[0, 100]``."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            "max_cloud_coverage must be a number between 0 and 100",
            source=_SOURCE,
            detail={"parameter": "max_cloud_coverage"},
        )
    fvalue = float(value)
    if not math.isfinite(fvalue) or not (0.0 <= fvalue <= 100.0):
        raise ValidationError(
            "max_cloud_coverage must be between 0 and 100",
            source=_SOURCE,
            detail={"parameter": "max_cloud_coverage"},
        )
    return fvalue


def _validate_bbox(bbox: Sequence[float]) -> BBox:
    """Validate a ``(west, south, east, north)`` bbox before any network call."""
    if isinstance(bbox, (str, bytes)) or not isinstance(bbox, Sequence):
        raise ValidationError(
            "bbox must be a sequence of four numbers (west, south, east, north)",
            source=_SOURCE,
            detail={"parameter": "bbox"},
        )
    if len(bbox) != 4:
        raise ValidationError(
            "bbox must contain exactly four values (west, south, east, north), "
            "got %d" % len(bbox),
            source=_SOURCE,
            detail={"parameter": "bbox"},
        )

    coords: List[float] = []
    for name, value in zip(("west", "south", "east", "north"), bbox):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(
                "bbox component %r must be a number" % name,
                source=_SOURCE,
                detail={"parameter": "bbox"},
            )
        fvalue = float(value)
        if not math.isfinite(fvalue):
            raise ValidationError(
                "bbox component %r must be finite" % name,
                source=_SOURCE,
                detail={"parameter": "bbox"},
            )
        coords.append(fvalue)

    west, south, east, north = coords
    if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
        raise ValidationError(
            "bbox longitudes must be within [-180, 180]",
            source=_SOURCE,
            detail={"parameter": "bbox"},
        )
    if not (-90.0 <= south <= 90.0 and -90.0 <= north <= 90.0):
        raise ValidationError(
            "bbox latitudes must be within [-90, 90]",
            source=_SOURCE,
            detail={"parameter": "bbox"},
        )
    if west > east:
        raise ValidationError(
            "bbox west must not be greater than east",
            source=_SOURCE,
            detail={"parameter": "bbox"},
        )
    if south > north:
        raise ValidationError(
            "bbox south must not be greater than north",
            source=_SOURCE,
            detail={"parameter": "bbox"},
        )
    return west, south, east, north


def _validate_datetime_range(datetime_range: Sequence[str]) -> Tuple[str, str]:
    """Validate a ``(start, end)`` ISO-8601 range, returning RFC3339 timestamps."""
    if (
        isinstance(datetime_range, (str, bytes))
        or not isinstance(datetime_range, Sequence)
        or len(datetime_range) != 2
    ):
        raise ValidationError(
            "datetime_range must be a (start, end) pair of ISO-8601 timestamps",
            source=_SOURCE,
            detail={"parameter": "datetime_range"},
        )
    start_raw, end_raw = datetime_range
    start_dt = _parse_iso(start_raw, field="start")
    end_dt = _parse_iso(end_raw, field="end")
    if start_dt > end_dt:
        raise ValidationError(
            "datetime_range start must not be later than end",
            source=_SOURCE,
            detail={"parameter": "datetime_range"},
        )
    return (
        _to_rfc3339(start_raw, start_dt, is_end=False),
        _to_rfc3339(end_raw, end_dt, is_end=True),
    )


def _to_rfc3339(raw: Any, dt: datetime, *, is_end: bool) -> str:
    """Normalize a parsed timestamp to full UTC RFC3339 (``...Z``) for TPDI."""
    text = raw.strip() if isinstance(raw, str) else str(raw)
    date_only = "T" not in text and " " not in text
    if date_only and is_end:
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=999000)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _parse_iso(value: Any, *, field: str) -> datetime:
    """Parse an ISO-8601 timestamp, tolerating a trailing ``Z`` (UTC)."""
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            "datetime_range %s must be a non-empty ISO-8601 string" % field,
            source=_SOURCE,
            detail={"parameter": "datetime_range"},
        )
    text = value.strip()
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValidationError(
            "datetime_range %s is not a valid ISO-8601 timestamp: %r"
            % (field, text),
            source=_SOURCE,
            detail={"parameter": "datetime_range"},
        ) from exc


def _validate_and_cap_limit(limit: int) -> int:
    """Validate ``limit`` and clamp it to :data:`MAX_ITEMS`."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValidationError(
            "limit must be an integer",
            source=_SOURCE,
            detail={"parameter": "limit"},
        )
    if limit < 1:
        raise ValidationError(
            "limit must be a positive integer",
            source=_SOURCE,
            detail={"parameter": "limit"},
        )
    return min(limit, MAX_ITEMS)


def _bbox_to_polygon(bbox: BBox) -> Dict[str, Any]:
    """Build a GeoJSON ``Polygon`` (lon/lat) from a bbox for TPDI ``bounds``."""
    west, south, east, north = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south],
            ]
        ],
    }


def _cloud_number(value: float) -> Any:
    """Render ``maxCloudCoverage`` as an int when whole (matching TPDI examples)."""
    if float(value).is_integer():
        return int(value)
    return value


def _feature_to_item(feature: Any, *, query_bbox: BBox) -> Optional[StacItem]:
    """Convert a Maxar TPDI search feature into a :class:`StacItem`.

    The id is read from ``catalogID`` (Maxar's field), falling back to ``id``.
    Returns ``None`` for a structurally unusable feature.
    """
    if not isinstance(feature, Mapping):
        return None
    item_id = feature.get(_MAXAR_ID_FIELD) or feature.get("id")
    if not isinstance(item_id, str) or not item_id:
        return None

    properties = feature.get("properties")
    if not isinstance(properties, Mapping):
        properties = {}

    assets = feature.get("assets")
    if not isinstance(assets, Mapping):
        assets = {}

    collection = feature.get("collection")
    if not isinstance(collection, str) or not collection:
        collection = PROVIDER

    return StacItem(
        id=item_id,
        bbox=_extract_bbox(feature.get("bbox"), fallback=query_bbox),
        datetime=_extract_datetime(properties),
        collection=collection,
        assets=dict(assets),
        properties=dict(properties),
    )


def _extract_datetime(properties: Mapping[str, Any]) -> str:
    """Pick the item's temporal metadata from STAC ``properties``."""
    for key in ("datetime", "start_datetime", "end_datetime", "acquisitionDate"):
        value = properties.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _extract_bbox(raw: Any, *, fallback: BBox) -> BBox:
    """Normalize a STAC bbox (4- or 6-element) to ``(w, s, e, n)``."""
    if isinstance(raw, (list, tuple)):
        nums = [
            float(v)
            for v in raw
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if len(raw) == 4 and len(nums) == 4:
            return nums[0], nums[1], nums[2], nums[3]
        if len(raw) == 6 and len(nums) == 6:
            return nums[0], nums[1], nums[3], nums[4]
    return fallback

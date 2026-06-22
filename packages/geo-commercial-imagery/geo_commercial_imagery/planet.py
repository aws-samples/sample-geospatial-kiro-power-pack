"""Planet-native connector for ``geo-commercial-imagery`` (Data + Orders APIs).

Sentinel Hub's TPDI is deprecated for Planet data (Planet migrated to its own
platform), so Planet discovery and ordering go through **Planet's first-party
APIs** here, not through Sentinel Hub:

* :func:`search_commercial` - ``POST https://api.planet.com/data/v1/quick-search``
  (the **Data API**) to discover Planet scenes for a bbox + time range, returned
  as a list of :class:`~geo_commercial_imagery.models.StacItem`.
* :func:`order_scene` - ``POST https://api.planet.com/compute/ops/orders/v2``
  (the **Orders API**) to place an order for one or more item ids.

Auth is a single ``mcp.json`` key, ``PL_API_KEY`` (a Planet API key), sent as
HTTP Basic auth with the key as the username (Planet's documented scheme). It is
classified License-Needed and enforced per-invocation: a missing key raises an
:class:`~geo_common.errors.AuthenticationError` naming it before any I/O.

Ordering safety differs from Sentinel Hub TPDI: a Planet order has **no draft
state** - ``POST`` to the Orders API *places* the order immediately (it consumes
quota). To preserve the uniform "explicit opt-in to spend" contract,
:func:`order_scene` refuses unless ``confirm=True`` is passed (there is no
dry-run to return), and only then issues the order.

The generic input validators (bbox, time range, scene ids, cloud) are shared
with the Sentinel Hub connector; only the request construction and response
parsing are Planet-specific.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Mapping, Optional, Sequence

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
from geo_commercial_imagery.sentinelhub import (
    _bbox_to_polygon,
    _validate_and_cap_limit,
    _validate_bbox,
    _validate_datetime_range,
    _validate_max_cloud_coverage,
    _validate_scene_ids,
)

__all__ = [
    "PLANET_API_KEY_KEY",
    "PROVIDER",
    "DEFAULT_PLANET_ITEM_TYPE",
    "DEFAULT_PLANET_PRODUCT_BUNDLE",
    "DEFAULT_DATA_SEARCH_URL",
    "DEFAULT_ORDERS_URL",
    "require_api_key",
    "search_commercial",
    "order_scene",
]

#: The ``mcp.json`` key holding the Planet API key.
PLANET_API_KEY_KEY = "PL_API_KEY"

#: The provider this connector serves.
PROVIDER = "PLANET"

#: The secret-free source identifier used on errors raised from this module.
_SOURCE = "geo-commercial-imagery"

#: Default Planet item type (the dominant PlanetScope scene type).
DEFAULT_PLANET_ITEM_TYPE = "PSScene"

#: Default Planet product bundle used when ordering.
DEFAULT_PLANET_PRODUCT_BUNDLE = "analytic_udm2"

#: Planet Data API quick-search endpoint.
DEFAULT_DATA_SEARCH_URL = "https://api.planet.com/data/v1/quick-search"

#: Planet Orders API v2 endpoint.
DEFAULT_ORDERS_URL = "https://api.planet.com/compute/ops/orders/v2"

#: Max pages of search results to follow.
_MAX_PAGES = 20

#: Default cap on returned scenes.
MAX_ITEMS = 1000


# ---------------------------------------------------------------------------
# Credential guard
# ---------------------------------------------------------------------------


def require_api_key(api_key: Optional[str], *, request: str) -> str:
    """Return the Planet API key, or raise an auth error naming the key."""
    if not (api_key and str(api_key).strip()):
        raise AuthenticationError(
            "geo-commercial-imagery requires a configured Planet API key for "
            "Planet data: %s is missing; configure it in mcp.json"
            % PLANET_API_KEY_KEY,
            source=_SOURCE,
            detail={"missing_credentials": [PLANET_API_KEY_KEY], "request": request},
        )
    return str(api_key).strip()


def _auth_headers(api_key: str) -> Dict[str, str]:
    """Build the HTTP Basic auth header (Planet API key as username)."""
    token = base64.b64encode(("%s:" % api_key).encode("ascii")).decode("ascii")
    return {"Authorization": "Basic %s" % token}


def _auth_error(request: str, *, original: Optional[str] = None) -> AuthenticationError:
    """Build an auth error naming the Planet API key (rejected credential)."""
    return AuthenticationError(
        "Planet rejected request %r: %s is missing or invalid"
        % (request, PLANET_API_KEY_KEY),
        source=_SOURCE,
        detail={"request": request, "invalid_credentials": [PLANET_API_KEY_KEY]},
        original=original,
    )


def _validate_item_type(item_type: Optional[str]) -> str:
    """Validate the Planet ``item_type`` (defaulting when absent)."""
    value = item_type if item_type is not None else DEFAULT_PLANET_ITEM_TYPE
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            "item_type must be a non-empty Planet itemType (e.g. 'PSScene')",
            source=_SOURCE,
            detail={"parameter": "item_type"},
        )
    return value.strip()


def _validate_product_bundle(product_bundle: Optional[str]) -> str:
    """Validate the Planet ``product_bundle`` (defaulting when absent)."""
    value = (
        product_bundle if product_bundle is not None else DEFAULT_PLANET_PRODUCT_BUNDLE
    )
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            "product_bundle must be a non-empty Planet product bundle "
            "(e.g. 'analytic_udm2')",
            source=_SOURCE,
            detail={"parameter": "product_bundle"},
        )
    return value.strip()


# ---------------------------------------------------------------------------
# Capability 1: search_commercial (Planet Data API)
# ---------------------------------------------------------------------------


async def search_commercial(
    *,
    bbox: Sequence[float],
    datetime_range: Sequence[str],
    item_type: Optional[str] = None,
    max_cloud_coverage: Optional[float] = None,
    http: HttpClient,
    api_key: Optional[str],
    limit: int = MAX_ITEMS,
    search_url: str = DEFAULT_DATA_SEARCH_URL,
) -> List[StacItem]:
    """Search Planet's Data API for scenes matching a query (Req 2.1, 7.8).

    ``bbox`` is ``(w, s, e, n)`` in EPSG:4326; ``datetime_range`` is an ISO-8601
    ``(start, end)`` pair (filtered on Planet's ``acquired`` field);
    ``item_type`` selects the Planet item type (default ``PSScene``);
    ``max_cloud_coverage`` (0-100) is converted to Planet's 0-1 ``cloud_cover``
    range. All inputs are validated before any network call.
    """
    request = "search_commercial"
    validated_bbox = _validate_bbox(bbox)
    start, end = _validate_datetime_range(datetime_range)
    effective_limit = _validate_and_cap_limit(limit)
    cloud = _validate_max_cloud_coverage(max_cloud_coverage)
    item = _validate_item_type(item_type)
    key = require_api_key(api_key, request=request)
    headers = _auth_headers(key)

    filter_config: List[Dict[str, Any]] = [
        {
            "type": "GeometryFilter",
            "field_name": "geometry",
            "config": _bbox_to_polygon(validated_bbox),
        },
        {
            "type": "DateRangeFilter",
            "field_name": "acquired",
            "config": {"gte": start, "lte": end},
        },
    ]
    if cloud is not None:
        filter_config.append(
            {
                "type": "RangeFilter",
                "field_name": "cloud_cover",
                "config": {"lte": cloud / 100.0},
            }
        )
    body: Dict[str, Any] = {
        "item_types": [item],
        "filter": {"type": "AndFilter", "config": filter_config},
    }

    items: List[StacItem] = []
    next_url: Optional[str] = None
    for _page in range(_MAX_PAGES):
        if next_url is None:
            response = await _post(http, search_url, body=body, headers=headers, request=request)
        else:
            response = await _get(http, next_url, headers=headers, request=request)
        payload = _decode_json(response, request=request)
        _raise_for_error(response, payload, request=request)

        features = payload.get("features", []) if isinstance(payload, Mapping) else []
        if not isinstance(features, list):
            features = []
        for feature in features:
            if len(items) >= effective_limit:
                return items
            converted = _feature_to_item(feature, query_bbox=validated_bbox, item_type=item)
            if converted is not None:
                items.append(converted)
        next_url = _next_link(payload)
        if not next_url or len(items) >= effective_limit:
            break
    return items


# ---------------------------------------------------------------------------
# Capability 2: order_scene (Planet Orders API)
# ---------------------------------------------------------------------------


async def order_scene(
    *,
    scene_ids: Sequence[str],
    item_type: Optional[str] = None,
    product_bundle: Optional[str] = None,
    name: Optional[str] = None,
    confirm: bool = False,
    http: HttpClient,
    api_key: Optional[str],
    orders_url: str = DEFAULT_ORDERS_URL,
) -> OrderConfirmation:
    """Place a Planet order for one or more item ids (Req 2.1, 7.8).

    Unlike Sentinel Hub TPDI, a Planet order has **no draft state**: issuing the
    order *places* it and consumes quota. To keep the uniform safety contract,
    this refuses unless ``confirm=True`` is passed (there is no dry-run preview to
    return), and only then posts the order. ``item_type`` (default ``PSScene``)
    and ``product_bundle`` (default ``analytic_udm2``) select the product;
    ``scene_ids`` are the item ids from :func:`search_commercial`.
    """
    request = "order_scene"
    ids = _validate_scene_ids(scene_ids)
    item = _validate_item_type(item_type)
    bundle = _validate_product_bundle(product_bundle)
    order_name = _resolve_order_name(name, scene_ids=ids)
    key = require_api_key(api_key, request=request)

    if not confirm:
        raise ValidationError(
            "Planet orders are placed immediately and consume quota (there is no "
            "draft/confirm step like Sentinel Hub); pass confirm=true to place "
            "the order",
            source=_SOURCE,
            detail={"parameter": "confirm", "provider": PROVIDER},
        )

    headers = _auth_headers(key)
    body: Dict[str, Any] = {
        "name": order_name,
        "products": [
            {"item_ids": ids, "item_type": item, "product_bundle": bundle}
        ],
    }

    response = await _post(http, orders_url, body=body, headers=headers, request=request)
    payload = _decode_json(response, request=request)
    _raise_for_error(response, payload, request=request)

    if not isinstance(payload, Mapping) or not payload:
        raise UpstreamError(
            "Planet returned an empty order response for %r" % request,
            source=_SOURCE,
            detail={"request": request},
        )
    order_id = payload.get("id")
    if not order_id:
        raise UpstreamError(
            "Planet order response omitted an order id for %r" % request,
            source=_SOURCE,
            detail={"request": request},
        )
    status = payload.get("state") or "queued"
    created = payload.get("created_on")
    return OrderConfirmation(
        order_id=str(order_id),
        provider=PROVIDER,
        scene_ids=ids,
        status=str(status),
        confirmed=True,
        sqkm=None,
        collection=None,
        created=str(created) if created else None,
    )


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def _raise_for_error(response: httpx.Response, payload: Any, *, request: str) -> None:
    """Map a Planet failure onto the ``Error_Taxonomy``, surfacing its message."""
    status = response.status_code
    http_detail = "HTTP %d %s" % (status, response.reason_phrase or "")
    body_message = _body_message(payload)
    original = "; ".join(p for p in (http_detail.strip(), body_message) if p)
    suffix = (": %s" % body_message) if body_message else ""

    if status in (401, 403):
        raise _auth_error(request, original=original)
    if status == 404:
        raise NotFoundError(
            "Planet request %r was not found (HTTP 404)%s" % (request, suffix),
            source=_SOURCE,
            detail={"request": request, "status_code": status, "planet_message": body_message},
            original=original,
        )
    if status in (400, 422):
        raise ValidationError(
            "Planet rejected request %r as invalid (HTTP %d)%s" % (request, status, suffix),
            source=_SOURCE,
            detail={"request": request, "status_code": status, "planet_message": body_message},
            original=original,
        )
    if status >= 400:
        raise UpstreamError(
            "Planet returned HTTP %d for request %r%s" % (status, request, suffix),
            source=_SOURCE,
            detail={"request": request, "status_code": status, "planet_message": body_message},
            original=original,
        )


def _body_message(payload: Any) -> Optional[str]:
    """Extract a human message from a Planet error body (``general``/``field``)."""
    if not isinstance(payload, Mapping):
        return None
    parts: List[str] = []
    general = payload.get("general")
    if isinstance(general, list):
        for item in general:
            if isinstance(item, Mapping):
                msg = item.get("message")
                if isinstance(msg, str) and msg:
                    parts.append(msg)
            elif isinstance(item, str) and item:
                parts.append(item)
    field = payload.get("field")
    if isinstance(field, Mapping):
        for name, violations in field.items():
            if isinstance(violations, list):
                for v in violations:
                    if isinstance(v, Mapping):
                        msg = v.get("message")
                        if isinstance(msg, str) and msg:
                            parts.append("%s: %s" % (name, msg))
    if not parts:
        for key in ("message", "error", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
                break
    return " | ".join(parts) if parts else None


def _identify_request(exc: GeoError, *, request: str) -> GeoError:
    """Re-tag a taxonomy error so it identifies the failed request."""
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


async def _post(
    http: HttpClient,
    url: str,
    *,
    body: Mapping[str, Any],
    headers: Mapping[str, str],
    request: str,
) -> httpx.Response:
    """POST ``body`` as JSON with the auth headers, normalizing failures."""
    try:
        return await http.post(url, json=dict(body), headers=dict(headers))
    except GeoError as exc:
        raise _identify_request(exc, request=request)
    except Exception as exc:  # pragma: no cover - defensive catch-all
        raise UpstreamError(
            "Planet request %r failed unexpectedly" % request,
            source=_SOURCE,
            detail={"request": request},
            original=str(exc),
        )


async def _get(
    http: HttpClient,
    url: str,
    *,
    headers: Mapping[str, str],
    request: str,
) -> httpx.Response:
    """GET ``url`` with the auth headers, normalizing failures."""
    try:
        return await http.get(url, headers=dict(headers))
    except GeoError as exc:
        raise _identify_request(exc, request=request)
    except Exception as exc:  # pragma: no cover - defensive catch-all
        raise UpstreamError(
            "Planet request %r failed unexpectedly" % request,
            source=_SOURCE,
            detail={"request": request},
            original=str(exc),
        )


def _decode_json(response: httpx.Response, *, request: str) -> Any:
    """Parse a response body as JSON, or raise an upstream error."""
    try:
        return response.json()
    except ValueError as exc:
        raise UpstreamError(
            "Planet returned a non-JSON response for request %r" % request,
            source=_SOURCE,
            detail={"request": request},
            original=str(exc),
        )


def _next_link(payload: Any) -> Optional[str]:
    """Return the ``_links._next`` pagination URL from a Data API page, if any."""
    if not isinstance(payload, Mapping):
        return None
    links = payload.get("_links")
    if isinstance(links, Mapping):
        nxt = links.get("_next")
        if isinstance(nxt, str) and nxt:
            return nxt
    return None


def _resolve_order_name(name: Optional[str], *, scene_ids: Sequence[str]) -> str:
    """Resolve the human-readable order name, validating an explicit one."""
    if name is None:
        return "geo-commercial-imagery Planet order (%d item(s))" % len(scene_ids)
    if not isinstance(name, str) or not name.strip():
        raise ValidationError(
            "name must be a non-empty string when provided",
            source=_SOURCE,
            detail={"parameter": "name"},
        )
    return name.strip()


def _feature_to_item(
    feature: Any, *, query_bbox: BBox, item_type: str
) -> Optional[StacItem]:
    """Convert a Planet Data API feature into a :class:`StacItem`."""
    if not isinstance(feature, Mapping):
        return None
    item_id = feature.get("id")
    if not isinstance(item_id, str) or not item_id:
        return None
    properties = feature.get("properties")
    if not isinstance(properties, Mapping):
        properties = {}
    return StacItem(
        id=item_id,
        bbox=_geometry_to_bbox(feature.get("geometry"), fallback=query_bbox),
        datetime=_planet_datetime(properties),
        collection=item_type,
        assets={},
        properties=dict(properties),
    )


def _planet_datetime(properties: Mapping[str, Any]) -> str:
    """Pick the acquisition timestamp from Planet item properties."""
    for key in ("acquired", "published", "updated"):
        value = properties.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _geometry_to_bbox(geometry: Any, *, fallback: BBox) -> BBox:
    """Compute a ``(w, s, e, n)`` bbox from a GeoJSON geometry, else fallback."""
    coords = _iter_positions(geometry.get("coordinates")) if isinstance(geometry, Mapping) else []
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    if lons and lats:
        return (min(lons), min(lats), max(lons), max(lats))
    return fallback


def _iter_positions(coordinates: Any) -> List[List[float]]:
    """Flatten nested GeoJSON coordinate arrays into a list of [lon, lat]."""
    positions: List[List[float]] = []

    def _walk(node: Any) -> None:
        if (
            isinstance(node, (list, tuple))
            and len(node) >= 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in node[:2])
        ):
            positions.append([float(node[0]), float(node[1])])
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                _walk(child)

    _walk(coordinates)
    return positions

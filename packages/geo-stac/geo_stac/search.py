"""STAC catalog search for ``geo-stac`` (Requirements 7.1, 7.12).

This module implements the MVP Pillar A discovery capability: a STAC catalog
search that wraps an external STAC API (Earth Search / Element 84 by default,
or any STAC-API-compliant endpoint such as Microsoft Planetary Computer) and
adapts it to the Power Pack conventions.

Behavior contract (design.md "Pillar A — Data Connectors"):

* :func:`stac_search` returns matching items, **each** carrying its asset
  references and its spatial + temporal metadata (Requirement 7.1).
* It returns **at most** the configured maximum of :data:`MAX_ITEMS` (1,000)
  items per response, clamping a larger requested ``limit`` down to the cap
  (Requirement 7.1).
* It validates every user-supplied parameter *before* any network call and
  raises a :class:`~geo_common.errors.ValidationError` (taxonomy
  ``validation``) naming the invalid parameter for a malformed bounding box, an
  out-of-range coordinate, a non-positive limit, an unparseable datetime, or a
  time range whose start is later than its end (Requirement 7.12).

The single async :class:`~geo_common.http.HttpClient` from ``geo-common`` is
used for the outbound POST to the STAC ``/search`` endpoint, so retry/backoff,
rate-limit handling, and the 30-second per-request timeout (Requirement 7.1's
"within 30 seconds") are all inherited from the shared client.

Python 3.10+ : uses ``from __future__ import annotations`` and ``X | None``
union syntax is therefore evaluated lazily where written in annotations.
"""

from __future__ import annotations

import asyncio
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from geo_common.errors import GeoError, ValidationError
from geo_common.http import HttpClient

__all__ = [
    "StacItem",
    "stac_search",
    "stac_search_multi",
    "StacSourceStatus",
    "StacSearchResult",
    "KNOWN_STAC_ENDPOINTS",
    "DEFAULT_STAC_API_URL",
    "MAX_ITEMS",
]

#: The configured maximum number of items returned per response (Requirement
#: 7.1). A requested ``limit`` larger than this is clamped down to it.
MAX_ITEMS: int = 1000

#: Default STAC API root used when no explicit ``api_url`` is supplied. Earth
#: Search (Element 84) is an open, no-credential STAC API.
DEFAULT_STAC_API_URL: str = "https://earth-search.aws.element84.com/v1"

#: The open STAC API roots the federated :func:`stac_search_multi` queries by
#: default. Each is a STAC-API ``/search``-capable endpoint; per-source failures
#: degrade gracefully (the source is marked failed and the rest still return).
KNOWN_STAC_ENDPOINTS: Dict[str, str] = {
    "earth-search": "https://earth-search.aws.element84.com/v1",
    "planetary-computer": "https://planetarycomputer.microsoft.com/api/stac/v1",
    "cmr-stac": "https://cmr.earthdata.nasa.gov/stac",
    "copernicus": "https://catalogue.dataspace.copernicus.eu/stac",
    "usgs": "https://landsatlook.usgs.gov/stac-server",
}

#: The secret-free source identifier used on errors raised from this module.
_SOURCE = "geo-stac"


class StacItem(BaseModel):
    """A STAC item returned by :func:`stac_search` (Requirement 7.1).

    Each item carries its asset references (``assets``) and its spatial
    (``bbox``) and temporal (``datetime``) metadata, as required by Requirement
    7.1, plus its STAC ``properties`` for downstream selection.
    """

    id: str
    #: ``(west, south, east, north)`` in EPSG:4326 decimal degrees.
    bbox: Tuple[float, float, float, float]
    #: ISO-8601 timestamp (or interval start) describing the item's time.
    datetime: str
    #: STAC asset references keyed by asset name (Requirement 7.1).
    assets: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    #: STAC item properties.
    properties: Dict[str, Any] = Field(default_factory=dict)


async def stac_search(
    *,
    bbox: Tuple[float, float, float, float],
    datetime_range: Tuple[str, str],
    collections: Optional[List[str]] = None,
    limit: int = MAX_ITEMS,
    http: Optional[HttpClient] = None,
    api_url: str = DEFAULT_STAC_API_URL,
) -> List[StacItem]:
    """Search a STAC catalog by spatial + temporal parameters.

    Parameters
    ----------
    bbox:
        ``(west, south, east, north)`` in EPSG:4326 decimal degrees. Longitudes
        must lie in ``[-180, 180]`` and latitudes in ``[-90, 90]`` with
        ``west <= east`` and ``south <= north`` (Requirement 7.12).
    datetime_range:
        ``(start, end)`` ISO-8601 timestamps. ``start`` must not be later than
        ``end`` (Requirement 7.12).
    collections:
        Optional list of STAC collection ids to restrict the search to.
    limit:
        Requested maximum number of items. Clamped to :data:`MAX_ITEMS`
        (1,000); must be a positive integer (Requirements 7.1, 7.12).
    http:
        Optional shared :class:`~geo_common.http.HttpClient`. A client is
        created (and closed) for the call when omitted; tests inject a client
        wired to an ``httpx.MockTransport``.
    api_url:
        STAC API root. Defaults to Earth Search (Element 84).

    Returns
    -------
    list[StacItem]
        Matching items (at most :data:`MAX_ITEMS`), each with asset references
        and spatio-temporal metadata (Requirement 7.1).

    Raises
    ------
    ValidationError
        For a malformed bounding box, an out-of-range coordinate, a
        non-positive ``limit``, an unparseable datetime, or a ``start`` later
        than ``end`` (Requirement 7.12).
    """
    # --- Validate inputs before any network call (Requirement 7.12) ---------
    validated_bbox = _validate_bbox(bbox)
    start, end = _validate_datetime_range(datetime_range)
    effective_limit = _validate_and_cap_limit(limit)

    body: Dict[str, Any] = {
        "bbox": list(validated_bbox),
        "datetime": "%s/%s" % (
            _normalize_datetime_bound(start, is_end=False),
            _normalize_datetime_bound(end, is_end=True),
        ),
        "limit": effective_limit,
    }
    if collections:
        body["collections"] = list(collections)

    search_url = _search_url(api_url)

    owns_client = http is None
    client = http if http is not None else HttpClient()
    try:
        response = await client.post(search_url, json=body)
    finally:
        if owns_client:
            await client.aclose()

    payload = _parse_json(response)
    features = payload.get("features", [])
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


class StacSourceStatus(BaseModel):
    """Per-source outcome in a federated :func:`stac_search_multi` result.

    ``status`` is ``"ok"`` (the source responded; ``count`` items contributed)
    or ``"error"`` (the source was unavailable; ``error``/``category`` describe
    the taxonomy-classified failure). Graceful degradation: a failed source does
    not fail the whole search - it is reported here and the search is marked
    ``partial``.
    """

    name: str
    url: str
    status: str
    count: int = 0
    error: Optional[str] = None
    category: Optional[str] = None


class StacSearchResult(BaseModel):
    """The result of a federated STAC search across multiple catalogs.

    ``items`` are the merged, de-duplicated matches (capped at the limit);
    ``sources`` reports each queried catalog's outcome; ``partial`` is ``True``
    when **any** source failed, so callers can tell a complete result from a
    degraded one (Requirement 7.11-style graceful degradation with provenance).
    """

    items: List[StacItem] = Field(default_factory=list)
    sources: List[StacSourceStatus] = Field(default_factory=list)
    partial: bool = False


#: Accepted de-duplication modes for :func:`stac_search_multi`.
_DEDUPE_MODES = ("scene", "id")

#: MGRS tile token (e.g. ``10SEG``) - two digits then three uppercase letters.
_MGRS_RE = re.compile(r"(\d{2}[A-Z]{3})")

#: STAC property keys that may directly carry an MGRS tile, in priority order.
_TILE_PROP_KEYS = ("s2:mgrs_tile", "sentinel:grid_square", "grid:code")


def _normalize_instant(value: str) -> Optional[str]:
    """Normalize an ISO-8601 timestamp to whole-second UTC, or ``None``.

    Collapses sub-second and timezone-format differences (e.g.
    ``...:29.024000Z`` vs ``...:29.024Z``) so the *same* acquisition instant
    reported by different catalogs yields an identical key. Returns ``None`` for
    an unparseable or interval value (so it never forces a false match).
    """
    text = (value or "").strip()
    if not text or "/" in text:
        return None
    iso = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.strftime("%Y-%m-%dT%H:%M:%S")


def _extract_tile(item: "StacItem") -> Optional[str]:
    """Extract an MGRS tile id (e.g. ``10SEG``) for ``item``, or ``None``.

    Tries well-known STAC tile properties first, then the discrete MGRS
    components, then falls back to scanning the item id. Returns ``None`` when
    no MGRS tile can be identified (e.g. a non-Sentinel-2 product), so the
    caller declines to scene-merge rather than risk a wrong match.
    """
    props = item.properties or {}
    for key in _TILE_PROP_KEYS:
        raw = props.get(key)
        if isinstance(raw, str):
            match = _MGRS_RE.search(raw)
            if match:
                return match.group(1)
    # Discrete MGRS components (utm zone + latitude band + grid square).
    for prefix in ("mgrs", "sentinel"):
        zone = props.get("%s:utm_zone" % prefix)
        band = props.get("%s:latitude_band" % prefix)
        square = props.get("%s:grid_square" % prefix)
        if zone is not None and band and square:
            token = "%02d%s%s" % (int(zone), str(band), str(square))
            if _MGRS_RE.fullmatch(token):
                return token
    match = _MGRS_RE.search(item.id or "")
    return match.group(1) if match else None


def _scene_key(item: "StacItem") -> Optional[str]:
    """Return a cross-catalog scene identity for ``item``, or ``None``.

    The identity is the acquisition instant (whole-second UTC) plus the MGRS
    tile. Both must be present, so distinct tiles from one overpass (which share
    a datetime) stay distinct, while the same physical granule published by
    different catalogs under different ids collapses to one. When either part is
    missing the function returns ``None`` and the caller falls back to id-based
    de-duplication (never an over-merge).
    """
    instant = _normalize_instant(item.datetime)
    if instant is None:
        return None
    tile = _extract_tile(item)
    if not tile:
        return None
    return "%s|%s" % (instant, tile)


async def stac_search_multi(
    *,
    bbox: Tuple[float, float, float, float],
    datetime_range: Tuple[str, str],
    collections: Optional[List[str]] = None,
    limit: int = MAX_ITEMS,
    http: Optional[HttpClient] = None,
    endpoints: Optional[Dict[str, str]] = None,
    dedupe: str = "scene",
) -> StacSearchResult:
    """Search several STAC catalogs concurrently and merge the results.

    Queries each configured endpoint (Earth Search, Planetary Computer,
    CMR-STAC, Copernicus, USGS by default) in parallel through the shared
    :class:`HttpClient`, then **round-robin merges** the matching items - taking
    one item from each responding catalog in turn - de-duplicating and stopping
    at ``limit``. Round-robin (rather than filling in source order) ensures every
    reachable catalog is represented under a small ``limit``, instead of the
    first catalog consuming the whole quota. Each source's outcome is reported in
    ``sources`` (with the count it contributed); a source that is unreachable or
    errors does **not** fail the whole search - it is recorded and the result is
    marked ``partial`` (graceful degradation). Inputs are validated once up
    front, so a malformed bbox/datetime/limit or an unknown ``dedupe`` raises a
    ``ValidationError`` before any network call (Requirement 7.12).

    ``dedupe`` controls cross-catalog de-duplication:

    * ``"scene"`` (default) - collapse the *same physical scene* served by
      multiple catalogs under different id schemes (e.g. one Sentinel-2 granule
      on Earth Search, Planetary Computer, and Copernicus) into a single item,
      using a conservative identity of acquisition instant + MGRS tile. An item
      with no derivable scene identity falls back to id de-duplication, so
      distinct scenes are never merged.
    * ``"id"`` - de-duplicate only by exact STAC item ``id``, keeping each
      catalog's copy of a scene (useful when the per-catalog asset variants -
      e.g. COG vs JP2 - matter).
    """
    # Validate once up front so bad input fails fast (not five identical errors).
    _validate_bbox(bbox)
    _validate_datetime_range(datetime_range)
    effective_limit = _validate_and_cap_limit(limit)
    if dedupe not in _DEDUPE_MODES:
        raise ValidationError(
            "dedupe must be one of: %s" % ", ".join(_DEDUPE_MODES),
            source=_SOURCE,
            detail={"parameter": "dedupe", "supported": list(_DEDUPE_MODES)},
        )

    targets = dict(endpoints) if endpoints is not None else dict(KNOWN_STAC_ENDPOINTS)

    owns_client = http is None
    client = http if http is not None else HttpClient()

    async def _one(name: str, url: str):
        try:
            found = await stac_search(
                bbox=bbox,
                datetime_range=datetime_range,
                collections=collections,
                limit=effective_limit,
                http=client,
                api_url=url,
            )
            return name, url, found, None
        except GeoError as exc:
            return name, url, None, exc

    try:
        results = await asyncio.gather(
            *(_one(name, url) for name, url in targets.items())
        )
    finally:
        if owns_client:
            await client.aclose()

    merged: List[StacItem] = []
    seen_ids: set = set()
    seen_scenes: set = set()
    sources: List[StacSourceStatus] = []
    partial = False

    def _is_duplicate(it: StacItem) -> bool:
        """A duplicate if the id was already taken, or (scene mode) its scene
        identity matches one already merged."""
        if it.id in seen_ids:
            return True
        if dedupe == "scene":
            key = _scene_key(it)
            if key is not None and key in seen_scenes:
                return True
        return False

    def _record(it: StacItem) -> None:
        seen_ids.add(it.id)
        if dedupe == "scene":
            key = _scene_key(it)
            if key is not None:
                seen_scenes.add(key)

    # First pass: record per-source outcomes (and mark failures partial), and
    # collect each ok source's items as a queue for the round-robin merge.
    ok_queues: List[Tuple[str, List[StacItem]]] = []
    for name, url, found, exc in results:
        if exc is not None:
            partial = True
            sources.append(
                StacSourceStatus(
                    name=name,
                    url=url,
                    status="error",
                    error=str(exc),
                    category=exc.category.value,
                )
            )
        else:
            ok_queues.append((name, list(found or [])))

    # Round-robin merge: take one item from each ok source in turn, so every
    # reachable catalog is represented under a small ``limit`` rather than the
    # first catalog (in arbitrary source order) filling the whole quota. Items
    # already seen (by id, or by scene identity in ``"scene"`` mode) are
    # skipped; the merge stops at the limit. The per-source ``count`` reflects
    # how many of that source's items actually made it into the merged set.
    contributed: Dict[str, int] = {name: 0 for name, _ in ok_queues}
    cursors = {name: 0 for name, _ in ok_queues}
    while len(merged) < effective_limit:
        advanced = False
        for name, queue in ok_queues:
            if len(merged) >= effective_limit:
                break
            idx = cursors[name]
            # Skip past any already-seen items for this source this round.
            while idx < len(queue) and _is_duplicate(queue[idx]):
                idx += 1
            if idx >= len(queue):
                cursors[name] = idx
                continue
            item = queue[idx]
            cursors[name] = idx + 1
            advanced = True
            _record(item)
            merged.append(item)
            contributed[name] += 1
        if not advanced:
            break  # every source exhausted

    for name, _ in ok_queues:
        url = next(u for n, u, _f, _e in results if n == name)
        sources.append(
            StacSourceStatus(name=name, url=url, status="ok", count=contributed[name])
        )

    # Preserve the configured source order in the provenance list.
    order = {name: i for i, name in enumerate(targets)}
    sources.sort(key=lambda s: order.get(s.name, len(order)))

    return StacSearchResult(items=merged, sources=sources, partial=partial)


# ---------------------------------------------------------------------------
# Validation helpers (Requirement 7.12)
# ---------------------------------------------------------------------------


def _validate_bbox(
    bbox: Sequence[float],
) -> Tuple[float, float, float, float]:
    """Validate a ``(west, south, east, north)`` bbox (Requirement 7.12).

    Raises :class:`ValidationError` naming ``bbox`` for any malformed shape,
    non-finite value, out-of-range coordinate, or inverted extent.
    """
    if isinstance(bbox, (str, bytes)) or not isinstance(bbox, Sequence):
        raise ValidationError(
            "bbox must be a sequence of four numbers (west, south, east, north)",
            source=_SOURCE,
            detail={"parameter": "bbox"},
        )
    if len(bbox) != 4:
        raise ValidationError(
            "bbox must contain exactly four values (west, south, east, north), "
            f"got {len(bbox)}",
            source=_SOURCE,
            detail={"parameter": "bbox"},
        )

    coords: List[float] = []
    for name, value in zip(("west", "south", "east", "north"), bbox):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(
                f"bbox component {name!r} must be a number",
                source=_SOURCE,
                detail={"parameter": "bbox"},
            )
        fvalue = float(value)
        if not math.isfinite(fvalue):
            raise ValidationError(
                f"bbox component {name!r} must be finite",
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


def _validate_datetime_range(
    datetime_range: Sequence[str],
) -> Tuple[str, str]:
    """Validate a ``(start, end)`` ISO-8601 range (Requirement 7.12).

    Raises :class:`ValidationError` naming ``datetime_range`` for a malformed
    pair, an unparseable timestamp, or a ``start`` later than ``end``.
    """
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
    return str(start_raw), str(end_raw)


def _parse_iso(value: Any, *, field: str) -> datetime:
    """Parse an ISO-8601 timestamp, tolerating a trailing ``Z`` (UTC)."""
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            f"datetime_range {field} must be a non-empty ISO-8601 string",
            source=_SOURCE,
            detail={"parameter": "datetime_range"},
        )
    text = value.strip()
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValidationError(
            f"datetime_range {field} is not a valid ISO-8601 timestamp: {text!r}",
            source=_SOURCE,
            detail={"parameter": "datetime_range"},
        ) from exc


def _validate_and_cap_limit(limit: int) -> int:
    """Validate ``limit`` and clamp it to :data:`MAX_ITEMS` (Req 7.1, 7.12)."""
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


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _normalize_datetime_bound(value: str, *, is_end: bool) -> str:
    """Expand a date-only bound to a full RFC3339 timestamp for STAC search.

    STAC APIs (e.g. Earth Search) match nothing for a bare date like
    ``"2024-01-01"`` - they expect full RFC3339 timestamps. A date-only bound is
    therefore expanded to span the whole day (``T00:00:00Z`` for the start,
    ``T23:59:59Z`` for the end) so a user passing plain dates still gets results
    rather than a silent empty response. Values that already carry a time
    component are returned unchanged.
    """
    text = value.strip()
    if "T" in text or " " in text:
        return text
    suffix = "T23:59:59Z" if is_end else "T00:00:00Z"
    return text + suffix


def _search_url(api_url: str) -> str:
    """Build the STAC ``/search`` URL from a STAC API root."""
    root = api_url.rstrip("/")
    if root.endswith("/search"):
        return root
    return f"{root}/search"


def _parse_json(response: Any) -> Dict[str, Any]:
    """Parse a STAC response body into a dict, tolerating odd payloads."""
    try:
        payload = response.json()
    except Exception:  # pragma: no cover - defensive; non-JSON upstream body
        return {}
    return payload if isinstance(payload, dict) else {}


def _feature_to_item(
    feature: Any, *, query_bbox: Tuple[float, float, float, float]
) -> Optional[StacItem]:
    """Convert a STAC feature dict into a :class:`StacItem`.

    Returns ``None`` for a structurally unusable feature so a single malformed
    record never breaks an otherwise valid response.
    """
    if not isinstance(feature, dict):
        return None
    item_id = feature.get("id")
    if not isinstance(item_id, str) or not item_id:
        return None

    properties = feature.get("properties")
    if not isinstance(properties, dict):
        properties = {}

    assets = feature.get("assets")
    if not isinstance(assets, dict):
        assets = {}

    item_datetime = _extract_datetime(properties)
    item_bbox = _extract_bbox(feature.get("bbox"), fallback=query_bbox)

    return StacItem(
        id=item_id,
        bbox=item_bbox,
        datetime=item_datetime,
        assets=assets,
        properties=properties,
    )


def _extract_datetime(properties: Dict[str, Any]) -> str:
    """Pick the item's temporal metadata from STAC ``properties``.

    Prefers ``datetime``; falls back to ``start_datetime`` for interval items.
    """
    for key in ("datetime", "start_datetime", "end_datetime"):
        value = properties.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _extract_bbox(
    raw: Any, *, fallback: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    """Normalize a STAC bbox (4- or 6-element) to ``(w, s, e, n)``.

    Falls back to the query bbox when the item omits or malforms its own.
    """
    if isinstance(raw, (list, tuple)):
        nums = [float(v) for v in raw if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(raw) == 4 and len(nums) == 4:
            return nums[0], nums[1], nums[2], nums[3]
        if len(raw) == 6 and len(nums) == 6:
            # [west, south, min_elev, east, north, max_elev] -> drop elevation.
            return nums[0], nums[1], nums[3], nums[4]
    return fallback

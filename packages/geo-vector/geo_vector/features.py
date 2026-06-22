"""Vector feature retrieval from OpenStreetMap (Overpass) and Overture Maps.

This module implements ``vector_features`` (Requirements 7.3, 7.12): given a
bounding box, it validates the extent and its area, then queries the configured
vector sources and merges their results into a single
:class:`~geo_vector.models.FeatureCollection`.

The shipped sources:

* :class:`OverpassSource` - OpenStreetMap data via the Overpass API. It builds
  an Overpass QL query for the bbox (optionally filtered to the requested
  layers as OSM tag keys) and parses the ``out:json`` element list into GeoJSON
  features. This is the **only source enabled by default** (see
  :func:`default_sources`) because it is a live, public REST API.
* :class:`OvertureSource` - Overture Maps data via a GeoJSON features endpoint.
  It requests features for the bbox and passes the returned GeoJSON through,
  tagging each feature with its source. Overture Maps does **not** operate a
  hosted ``/features`` REST API - the data is distributed as cloud-native
  GeoParquet on S3/Azure and is normally queried with DuckDB or the
  ``overturemaps`` CLI - so this source ships **disabled by default** and must
  be pointed at an explicit GeoJSON endpoint (for example an internal service
  that fronts the Overture GeoParquet) to be used.

Every outbound call goes through the shared
:class:`~geo_common.http.HttpClient`, so all sources inherit the same
retry/backoff, rate-limit handling, and 30-second per-request timeout
(Requirement 5) - which is how the 30-second response bound of Requirement 7.3
is enforced. Source/transport failures are mapped onto the ``Error_Taxonomy``
so a malformed upstream response surfaces as an ``UpstreamError`` rather than
an opaque exception.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from geo_common.errors import (
    ErrorCategory,
    GeoError,
    NetworkError,
    UpstreamError,
)
from geo_common.http import HttpClient

from geo_vector.bbox import DEFAULT_MAX_AREA_KM2, bbox_area_km2, validate_bbox
from geo_vector.models import BBox, Feature, FeatureCollection

__all__ = [
    "VectorSource",
    "OverpassSource",
    "OvertureSource",
    "default_sources",
    "vector_features",
    "DEFAULT_MAX_FEATURES",
]

#: Taxonomy categories that represent a configured source being unavailable -
#: unreachable or not responding within the 30-second per-request timeout
#: (Requirement 7.11). The shared :class:`HttpClient` already maps a timeout or
#: connection failure onto ``NETWORK`` and an upstream 5xx / server rate-limit
#: onto ``UPSTREAM`` (Requirement 5.4); ``vector_features`` re-tags such an
#: error with the *logical* source name so the availability error names the
#: unavailable source rather than only the host.
_AVAILABILITY_CATEGORIES = (ErrorCategory.NETWORK, ErrorCategory.UPSTREAM)

#: Default public endpoint for the OpenStreetMap Overpass source. Overridable
#: per source instance so a deployment (or a test) can point at a mirror or a
#: mock transport.
DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

#: Default cap on the number of features a single ``vector_features`` call may
#: return. Overpass returns *every* element in an extent, so even a small dense
#: urban bbox can yield tens of thousands of features - far more than a calling
#: agent's context can absorb. Exceeding this cap raises a ``ValidationError``
#: asking the caller to narrow the request (smaller bbox or a ``layers`` filter)
#: rather than returning an unusable, oversized collection. Overridable per
#: request and per server.
DEFAULT_MAX_FEATURES = 2000


class VectorSource:
    """Base class for a vector data source queried by ``vector_features``.

    A source exposes a human-readable :attr:`name` and an async
    :meth:`fetch` that returns the source's features for a validated bbox.
    Subclasses make all I/O through the shared :class:`HttpClient` and raise a
    :class:`~geo_common.errors.GeoError` (taxonomy-classified) on failure.
    """

    name: str = "vector-source"

    async def fetch(
        self,
        http: HttpClient,
        bbox: BBox,
        layers: Optional[Sequence[str]],
    ) -> List[Feature]:  # pragma: no cover - abstract
        raise NotImplementedError


class OverpassSource(VectorSource):
    """OpenStreetMap features via the Overpass API."""

    name = "openstreetmap"

    def __init__(self, url: str = DEFAULT_OVERPASS_URL, *, timeout_s: int = 25) -> None:
        self.url = url
        self.timeout_s = timeout_s

    def build_query(self, bbox: BBox, layers: Optional[Sequence[str]]) -> str:
        """Build an Overpass QL query for ``bbox`` (optionally tag-filtered).

        Overpass bbox order is ``(south, west, north, east)`` =
        ``(min_lat, min_lon, max_lat, max_lon)``. When ``layers`` are given,
        each is treated as an OSM tag key and the query is restricted to
        elements carrying that key; otherwise every element in the bbox is
        requested. ``out geom;`` returns way/relation geometry inline so it can
        be converted to GeoJSON without a second round-trip.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        extent = "%g,%g,%g,%g" % (min_lat, min_lon, max_lat, max_lon)
        selectors = [""] if not layers else ["[%s]" % _escape_tag(layer) for layer in layers]
        clauses = []
        for sel in selectors:
            clauses.append("node%s(%s);" % (sel, extent))
            clauses.append("way%s(%s);" % (sel, extent))
            clauses.append("relation%s(%s);" % (sel, extent))
        return "[out:json][timeout:%d];(%s);out geom;" % (
            self.timeout_s,
            "".join(clauses),
        )

    async def fetch(
        self,
        http: HttpClient,
        bbox: BBox,
        layers: Optional[Sequence[str]],
    ) -> List[Feature]:
        query = self.build_query(bbox, layers)
        try:
            response = await http.post(
                self.url,
                content=query.encode("utf-8"),
                headers={"Content-Type": "text/plain; charset=utf-8"},
            )
        except GeoError:
            # Already taxonomy-classified (timeout/network/rate-limit/etc.).
            raise
        if response.status_code >= 400:
            raise UpstreamError(
                "Overpass returned HTTP %d" % response.status_code,
                source=self.name,
                detail={"status_code": response.status_code},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(
                "Overpass returned a non-JSON response",
                source=self.name,
                original=str(exc),
            )
        return _parse_overpass(payload)


class OvertureSource(VectorSource):
    """Overture Maps features via a GeoJSON features endpoint.

    Overture Maps publishes its data as cloud-native GeoParquet on S3/Azure
    rather than a hosted REST API, so there is **no public** ``/features``
    endpoint to default to. This source therefore requires an explicit ``url``
    pointing at a GeoJSON features service - for example an internal gateway
    that translates a bbox query into a DuckDB read over the Overture
    GeoParquet and returns a GeoJSON ``FeatureCollection``. Because no such
    endpoint ships by default, :func:`default_sources` does not include this
    source; configure it explicitly to enable Overture-backed results.
    """

    name = "overture"

    def __init__(self, url: str) -> None:
        if not url:
            raise ValueError(
                "OvertureSource requires an explicit GeoJSON features endpoint "
                "URL; Overture Maps has no public hosted /features REST API "
                "(its data is cloud-native GeoParquet queried via DuckDB)."
            )
        self.url = url

    async def fetch(
        self,
        http: HttpClient,
        bbox: BBox,
        layers: Optional[Sequence[str]],
    ) -> List[Feature]:
        params: Dict[str, Any] = {"bbox": ",".join("%g" % v for v in bbox)}
        if layers:
            # Overture calls layers "theme types"; pass them through as a list.
            params["types"] = list(layers)
        try:
            response = await http.get(self.url, params=params)
        except GeoError:
            raise
        if response.status_code >= 400:
            raise UpstreamError(
                "Overture returned HTTP %d" % response.status_code,
                source=self.name,
                detail={"status_code": response.status_code},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(
                "Overture returned a non-JSON response",
                source=self.name,
                original=str(exc),
            )
        return _parse_geojson(payload, source=self.name)


def default_sources() -> List[VectorSource]:
    """The default source set: OpenStreetMap (Overpass) only.

    Only the OpenStreetMap Overpass source is enabled by default because it is
    a live, public REST API. Overture Maps has no hosted ``/features`` endpoint
    (its data is cloud-native GeoParquet queried via DuckDB / the
    ``overturemaps`` CLI), so :class:`OvertureSource` ships disabled and must be
    configured with an explicit GeoJSON endpoint to be added to this set.
    """
    return [OverpassSource()]


async def vector_features(
    *,
    bbox: Sequence[float],
    http: HttpClient,
    sources: Sequence[VectorSource],
    layers: Optional[Sequence[str]] = None,
    max_area_km2: float = DEFAULT_MAX_AREA_KM2,
    max_features: int = DEFAULT_MAX_FEATURES,
) -> FeatureCollection:
    """Return vector features within ``bbox`` from every configured source.

    Validates ``bbox`` and its area first (Requirement 7.12): an area greater
    than ``max_area_km2`` (default 2,500 km²) raises a
    :class:`~geo_common.errors.ValidationError` before any source is queried
    (Requirement 7.3). On a valid request, each source is queried through the
    shared :class:`HttpClient` (inheriting its 30-second per-request timeout,
    Requirement 7.3) and the results are merged into one
    :class:`FeatureCollection`. A source/transport failure propagates as a
    taxonomy-classified :class:`~geo_common.errors.GeoError` so no partial
    collection is returned (Requirement 7.11).

    The merged collection is capped at ``max_features`` (default 2,000). Because
    Overpass returns *every* element in an extent, an under-area but dense bbox
    can still yield tens of thousands of features - too many for a calling
    agent to consume. When the merged count exceeds the cap, a
    :class:`~geo_common.errors.ValidationError` is raised asking the caller to
    narrow the request (a smaller bbox or a ``layers`` filter) rather than
    returning an oversized, hard-to-use collection.
    """
    validated = validate_bbox(bbox)
    area = bbox_area_km2(validated)
    if area > max_area_km2:
        from geo_common.errors import ValidationError

        raise ValidationError(
            "bbox area %.3f km² exceeds the configured maximum of %.3f km²"
            % (area, max_area_km2),
            source="geo-vector",
            detail={
                "parameter": "bbox",
                "area_km2": area,
                "max_area_km2": max_area_km2,
            },
        )

    features: List[Feature] = []
    for source in sources:
        try:
            features.extend(await source.fetch(http, validated, layers))
        except GeoError as exc:
            # A source that is unreachable or does not respond within the 30s
            # per-request timeout surfaces here as a taxonomy-classified
            # availability error (NETWORK/UPSTREAM). Re-tag it with the logical
            # source name so the error identifies the unavailable source, and
            # raise *before* returning so no partial collection escapes
            # (Requirement 7.11). Non-availability errors pass through with
            # their original taxonomy classification (Requirement 11.2).
            raise _identify_unavailable_source(exc, source_name=source.name)

    if len(features) > max_features:
        from geo_common.errors import ValidationError

        raise ValidationError(
            "request matched %d features, exceeding the configured maximum of "
            "%d; narrow the request with a smaller bbox or a 'layers' filter"
            % (len(features), max_features),
            source="geo-vector",
            detail={
                "parameter": "max_features",
                "feature_count": len(features),
                "max_features": max_features,
            },
        )
    return FeatureCollection(features=features, bbox=validated)


def _identify_unavailable_source(exc: GeoError, *, source_name: str) -> GeoError:
    """Ensure an availability failure identifies the unavailable source (Req 7.11).

    The shared :class:`HttpClient` tags a timeout/connection failure (``NETWORK``)
    or an upstream 5xx/server rate-limit (``UPSTREAM``) with the request *host*.
    For an availability error this re-tags the error's ``source`` with the
    configured source's logical name (e.g. ``"openstreetmap"``) so the returned
    Error_Taxonomy availability error names the unavailable source. The category,
    message, structured ``detail``, retained ``original`` detail, and any
    ``retry_after`` are preserved. Errors in any other taxonomy category (for
    example a validation or authentication error) pass through unchanged.
    """
    if exc.category not in _AVAILABILITY_CATEGORIES:
        return exc
    if exc.source == source_name:
        return exc
    cls = NetworkError if exc.category is ErrorCategory.NETWORK else UpstreamError
    return cls(
        exc.message,
        source=source_name,
        detail=exc.detail,
        original=exc.original,
        retry_after=exc.retry_after,
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _escape_tag(layer: str) -> str:
    """Escape a layer name for safe inclusion in an Overpass tag selector."""
    return str(layer).replace("\\", "\\\\").replace('"', '\\"').replace("]", "")


def _parse_overpass(payload: Any) -> List[Feature]:
    """Convert an Overpass ``out:json`` payload into GeoJSON features.

    ``node`` elements become ``Point`` features; ``way`` elements with inline
    ``geometry`` become ``Polygon`` (closed rings) or ``LineString`` features.
    Each feature's ``properties`` carries the element's OSM tags plus a
    ``"source"`` of ``"openstreetmap"`` and the OSM element type/id.
    """
    if not isinstance(payload, dict):
        raise UpstreamError(
            "Overpass payload was not a JSON object",
            source="openstreetmap",
        )
    elements = payload.get("elements", [])
    if not isinstance(elements, list):
        raise UpstreamError(
            "Overpass 'elements' was not a list",
            source="openstreetmap",
        )

    features: List[Feature] = []
    for element in elements:
        if not isinstance(element, dict):
            continue
        etype = element.get("type")
        geometry: Optional[Dict[str, Any]] = None

        if etype == "node":
            lon, lat = element.get("lon"), element.get("lat")
            if lon is None or lat is None:
                continue
            geometry = {"type": "Point", "coordinates": [lon, lat]}
        elif etype == "way":
            coords = [
                [pt["lon"], pt["lat"]]
                for pt in element.get("geometry", []) or []
                if isinstance(pt, dict) and "lon" in pt and "lat" in pt
            ]
            if not coords:
                continue
            if len(coords) >= 4 and coords[0] == coords[-1]:
                geometry = {"type": "Polygon", "coordinates": [coords]}
            else:
                geometry = {"type": "LineString", "coordinates": coords}
        else:
            # Relations and other element types are skipped (no inline geom).
            continue

        properties = dict(element.get("tags", {}) or {})
        properties["source"] = "openstreetmap"
        properties["osm_type"] = etype
        features.append(
            Feature(
                geometry=geometry,
                properties=properties,
                id="osm/%s/%s" % (etype, element.get("id")),
            )
        )
    return features


def _parse_geojson(payload: Any, *, source: str) -> List[Feature]:
    """Parse a GeoJSON ``FeatureCollection`` payload into :class:`Feature`s.

    Each feature is tagged with ``properties["source"] = source`` so merged
    results stay attributable to the originating dataset.
    """
    if not isinstance(payload, dict):
        raise UpstreamError(
            "%s payload was not a JSON object" % source, source=source
        )
    raw_features = payload.get("features", [])
    if not isinstance(raw_features, list):
        raise UpstreamError(
            "%s 'features' was not a list" % source, source=source
        )

    features: List[Feature] = []
    for raw in raw_features:
        if not isinstance(raw, dict):
            continue
        properties = dict(raw.get("properties") or {})
        properties["source"] = source
        features.append(
            Feature(
                geometry=raw.get("geometry"),
                properties=properties,
                id=raw.get("id"),
            )
        )
    return features

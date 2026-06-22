"""Species-occurrence retrieval from GBIF (and other biodiversity sources).

This module implements ``species_occurrences`` (Requirements 7.7, 7.12): given
a bounding box (and an optional taxon filter), it validates the parameters,
queries the configured biodiversity source(s), and returns the matching
occurrence records as a list of dicts, capped at the configured maximum
(:data:`~geo_biodiversity.validation.DEFAULT_MAX_RECORDS`, 10,000).

One source ships by default:

* :class:`GBIFSource` - the Global Biodiversity Information Facility occurrence
  search API. It expresses the bbox as a WKT polygon ``geometry`` filter,
  restricts to georeferenced records (``hasCoordinate=true``), optionally adds
  the ``scientificName`` filter, and parses the ``results`` array into records.

Every outbound call goes through the shared
:class:`~geo_common.http.HttpClient`, so all sources inherit the same
retry/backoff, rate-limit handling, and 30-second per-request timeout
(Requirement 5) - which is how the 30-second response bound of Requirement 7.7
is enforced. Source/transport failures are mapped onto the ``Error_Taxonomy``
so a malformed upstream response surfaces as an ``UpstreamError`` rather than
an opaque exception.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from geo_common.errors import (
    AuthenticationError,
    ErrorCategory,
    GeoError,
    NetworkError,
    UpstreamError,
    ValidationError,
)
from geo_common.http import HttpClient

from geo_biodiversity.models import BBox, OccurrenceRecord
from geo_biodiversity.validation import (
    DEFAULT_MAX_RECORDS,
    validate_and_cap_limit,
    validate_bbox,
    validate_taxon,
)

__all__ = [
    "BiodiversitySource",
    "GBIFSource",
    "INaturalistSource",
    "IUCNSource",
    "default_sources",
    "species_occurrences",
    "DEFAULT_GBIF_URL",
    "DEFAULT_INATURALIST_URL",
    "DEFAULT_IUCN_URL",
]

#: Default public GBIF occurrence-search endpoint. Overridable per source
#: instance so a deployment (or a test) can point at a mirror or a mock
#: transport.
DEFAULT_GBIF_URL = "https://api.gbif.org/v1/occurrence/search"

#: Default public iNaturalist observations endpoint (open for read; an optional
#: token raises rate limits / unlocks authenticated queries).
DEFAULT_INATURALIST_URL = "https://api.inaturalist.org/v1/observations"

#: Default IUCN Red List API v4 root. Credentialed: every request carries the
#: account token in the ``Authorization`` header (from ``IUCN_TOKEN``).
DEFAULT_IUCN_URL = "https://api.iucnredlist.org/api/v4"

#: The secret-free server identifier used on errors raised from this module.
_SERVER_NAME = "geo-biodiversity"

#: Taxonomy categories that mean a configured source was unavailable -
#: unreachable or not responding within the 30-second per-request timeout
#: (Requirement 7.11). The shared :class:`HttpClient` tags such a failure with
#: the request *host*; :func:`species_occurrences` re-tags it with the logical
#: source name so the availability error identifies the unavailable source.
_AVAILABILITY_CATEGORIES = (ErrorCategory.NETWORK, ErrorCategory.UPSTREAM)

#: GBIF caps a single occurrence-search page at 300 records; the connector
#: never requests more than this per request and stops once the effective
#: record cap is reached.
_GBIF_PAGE_LIMIT = 300

#: iNaturalist caps a single observations page at 200 records.
_INATURALIST_PAGE_LIMIT = 200


class BiodiversitySource:
    """Base class for a biodiversity source queried by ``species_occurrences``.

    A source exposes a human-readable :attr:`name` and an async :meth:`fetch`
    that returns the source's occurrence records (as :class:`OccurrenceRecord`)
    for a validated bbox, optional taxon, and a record cap. Subclasses make all
    I/O through the shared :class:`HttpClient` and raise a
    :class:`~geo_common.errors.GeoError` (taxonomy-classified) on failure.
    """

    name: str = "biodiversity-source"

    #: Whether this source participates in the *default* merge (when the caller
    #: does not name a ``source``). Sources that return something other than
    #: point occurrences - e.g. IUCN conservation status - set this ``False`` so
    #: they are opt-in only (reachable by ``source="..."``) and never silently
    #: mixed into a merged occurrence result.
    default_merge: bool = True

    #: Whether this source can only be queried with a ``taxon`` (it looks species
    #: up by scientific name and ignores the bbox). When such a source is
    #: selected without a taxon, the request is rejected with a clear validation
    #: error rather than silently returning nothing.
    requires_taxon: bool = False

    async def fetch(
        self,
        http: HttpClient,
        bbox: BBox,
        taxon: Optional[str],
        limit: int,
    ) -> List[OccurrenceRecord]:  # pragma: no cover - abstract
        raise NotImplementedError


class GBIFSource(BiodiversitySource):
    """Species occurrences via the GBIF occurrence-search API."""

    name = "gbif"

    def __init__(self, url: str = DEFAULT_GBIF_URL) -> None:
        self.url = url

    def build_params(
        self, bbox: BBox, taxon: Optional[str], limit: int
    ) -> Dict[str, Any]:
        """Build GBIF occurrence-search query parameters for ``bbox``.

        The bbox becomes a WKT polygon ``geometry`` filter wound
        counter-clockwise (GBIF's required winding for a search polygon):
        ``(min_lon min_lat, max_lon min_lat, max_lon max_lat, min_lon max_lat,
        min_lon min_lat)``. ``hasCoordinate=true`` restricts to georeferenced
        records, and a ``taxon`` filter, when present, is passed as
        ``scientificName``. ``limit`` is bounded by GBIF's per-page maximum.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        polygon = (
            "POLYGON((%g %g, %g %g, %g %g, %g %g, %g %g))"
            % (
                min_lon, min_lat,
                max_lon, min_lat,
                max_lon, max_lat,
                min_lon, max_lat,
                min_lon, min_lat,
            )
        )
        params: Dict[str, Any] = {
            "geometry": polygon,
            "hasCoordinate": "true",
            "limit": min(limit, _GBIF_PAGE_LIMIT),
        }
        if taxon:
            params["scientificName"] = taxon
        return params

    async def fetch(
        self,
        http: HttpClient,
        bbox: BBox,
        taxon: Optional[str],
        limit: int,
    ) -> List[OccurrenceRecord]:
        params = self.build_params(bbox, taxon, limit)
        try:
            response = await http.get(self.url, params=params)
        except GeoError:
            # Already taxonomy-classified (timeout/network/rate-limit/etc.).
            raise
        if response.status_code >= 400:
            raise UpstreamError(
                "GBIF returned HTTP %d" % response.status_code,
                source=self.name,
                detail={"status_code": response.status_code},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(
                "GBIF returned a non-JSON response",
                source=self.name,
                original=str(exc),
            )
        return _parse_gbif(payload)


class INaturalistSource(BiodiversitySource):
    """Species occurrences via the iNaturalist observations API.

    iNaturalist's observation search is open (no credential needed); an optional
    bearer ``token`` (from ``INATURALIST_TOKEN``) is forwarded for higher rate
    limits / authenticated queries when configured. The bbox is expressed as the
    ``nelat``/``nelng``/``swlat``/``swlng`` corner filter with ``geo=true`` to
    restrict to georeferenced observations; ``taxon`` (when present) is passed
    as ``taxon_name``.
    """

    name = "inaturalist"

    def __init__(
        self,
        url: str = DEFAULT_INATURALIST_URL,
        *,
        token: Optional[str] = None,
    ) -> None:
        self.url = url
        self.token = token or None

    def build_params(
        self, bbox: BBox, taxon: Optional[str], limit: int
    ) -> Dict[str, Any]:
        """Build iNaturalist observation-search query parameters for ``bbox``."""
        min_lon, min_lat, max_lon, max_lat = bbox
        params: Dict[str, Any] = {
            "nelat": max_lat,
            "nelng": max_lon,
            "swlat": min_lat,
            "swlng": min_lon,
            "geo": "true",
            "per_page": min(limit, _INATURALIST_PAGE_LIMIT),
        }
        if taxon:
            params["taxon_name"] = taxon
        return params

    async def fetch(
        self,
        http: HttpClient,
        bbox: BBox,
        taxon: Optional[str],
        limit: int,
    ) -> List[OccurrenceRecord]:
        params = self.build_params(bbox, taxon, limit)
        headers = {"Authorization": "Bearer %s" % self.token} if self.token else None
        try:
            response = await http.get(self.url, params=params, headers=headers)
        except GeoError:
            raise
        if response.status_code >= 400:
            raise UpstreamError(
                "iNaturalist returned HTTP %d" % response.status_code,
                source=self.name,
                detail={"status_code": response.status_code},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(
                "iNaturalist returned a non-JSON response",
                source=self.name,
                original=str(exc),
            )
        return _parse_inaturalist(payload)


class IUCNSource(BiodiversitySource):
    """Conservation-status records via the IUCN Red List API v4 (credentialed).

    The IUCN Red List API is **credentialed** (the account token is sent in the
    ``Authorization`` header, read from ``IUCN_TOKEN``) and **name-based** rather
    than spatial: it is queried by scientific name, so this source only
    contributes when a ``taxon`` filter is supplied. For a taxon it returns the
    species' Red List assessment(s) as occurrence records carrying the
    conservation ``category`` (e.g. ``"VU"``, ``"EN"``) under ``properties``;
    such records have **no point coordinate** (``longitude``/``latitude`` are
    ``None``), since an assessment describes a species' global status, not a
    sighting.

    When no token is configured the source raises an ``Error_Taxonomy``
    :class:`~geo_common.errors.AuthenticationError` naming the ``mcp.json`` key
    (never the secret value) before any request is made.
    """

    name = "iucn"

    #: IUCN returns conservation *status*, not point occurrences, so it is
    #: opt-in only - reachable via ``source="iucn"`` but never part of the
    #: default merged occurrence result.
    default_merge = False

    #: IUCN is queried by scientific name (it ignores the bbox), so a ``taxon``
    #: is required when it is selected.
    requires_taxon = True

    def __init__(
        self,
        url: str = DEFAULT_IUCN_URL,
        *,
        token: Optional[str] = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.token = token or None

    async def fetch(
        self,
        http: HttpClient,
        bbox: BBox,
        taxon: Optional[str],
        limit: int,
    ) -> List[OccurrenceRecord]:
        if not self.token:
            # Credential guard: deny before any request, naming the key only.
            raise AuthenticationError(
                "IUCN Red List is not configured: set IUCN_TOKEN in mcp.json",
                source=self.name,
                detail={"mcp_json_key": "IUCN_TOKEN"},
            )
        if not taxon:
            # IUCN is queried by scientific name; with no taxon there is nothing
            # to look up, so this source contributes no records.
            return []

        genus, _, species = taxon.strip().partition(" ")
        params: Dict[str, Any] = {"genus_name": genus}
        if species:
            params["species_name"] = species
        try:
            response = await http.get(
                "%s/taxa/scientific_name" % self.url,
                params=params,
                headers={"Authorization": self.token},
            )
        except GeoError:
            raise
        if response.status_code == 404:
            return []  # no assessment for this name
        if response.status_code >= 400:
            raise UpstreamError(
                "IUCN Red List returned HTTP %d" % response.status_code,
                source=self.name,
                detail={"status_code": response.status_code},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(
                "IUCN Red List returned a non-JSON response",
                source=self.name,
                original=str(exc),
            )
        return _parse_iucn(payload, taxon)


def default_sources(inaturalist_token: Optional[str] = None) -> List[BiodiversitySource]:
    """The default source set: GBIF + iNaturalist (both open, no credential).

    iNaturalist's observation search is open; an optional ``inaturalist_token``
    (from ``INATURALIST_TOKEN``) is forwarded as a bearer token for higher rate
    limits / authenticated queries when configured.
    """
    return [GBIFSource(), INaturalistSource(token=inaturalist_token)]


async def species_occurrences(
    *,
    bbox: Sequence[float],
    http: HttpClient,
    sources: Sequence[BiodiversitySource],
    taxon: Optional[str] = None,
    limit: int = DEFAULT_MAX_RECORDS,
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return species-occurrence records within ``bbox`` (Req 7.7).

    Validates ``bbox``, ``taxon``, ``limit``, and ``source`` first
    (Requirement 7.12): a malformed bbox/coordinate, a non-blank-string
    ``taxon``, a non-positive ``limit``, or an unknown ``source`` name raises a
    :class:`~geo_common.errors.ValidationError` before any source is queried. A
    valid ``limit`` larger than the configured maximum
    (:data:`DEFAULT_MAX_RECORDS`, 10,000) is clamped to it, and the result is
    capped at the effective limit so a response never exceeds the maximum
    (Requirement 7.7).

    ``source`` selects which configured source(s) to query:

    * omitted (``None``) - query the configured sources and **merge** their
      records, in source order, up to the cap (the default behaviour); and
    * a name (e.g. ``"gbif"``, ``"inaturalist"``, ``"iucn"``) - query only that
      source, so a single provider can be isolated.

    Each selected source is queried through the shared :class:`HttpClient`
    (inheriting its 30-second per-request timeout, Requirement 7.7); a
    source/transport failure propagates as a taxonomy-classified
    :class:`~geo_common.errors.GeoError`.
    """
    validated = validate_bbox(bbox)
    taxon_filter = validate_taxon(taxon)
    effective_limit = validate_and_cap_limit(limit)
    selected = _select_sources(sources, source)

    # A name-based source (e.g. IUCN) cannot do anything useful without a taxon
    # - it looks species up by scientific name and ignores the bbox - so reject
    # the request with a clear message rather than silently returning nothing.
    if taxon_filter is None:
        needs_taxon = [s.name for s in selected if getattr(s, "requires_taxon", False)]
        if needs_taxon:
            raise ValidationError(
                "source %s looks species up by scientific name and ignores the "
                "bbox, so it requires a 'taxon'; provide one (e.g. "
                "'Panthera leo')" % ", ".join(sorted(needs_taxon)),
                source=_SERVER_NAME,
                detail={"parameter": "taxon", "sources": sorted(needs_taxon)},
            )

    records: List[Dict[str, Any]] = []
    for chosen in selected:
        if len(records) >= effective_limit:
            break
        remaining = effective_limit - len(records)
        try:
            fetched = await chosen.fetch(http, validated, taxon_filter, remaining)
        except GeoError as exc:
            # Re-tag an availability failure with the logical source name so the
            # error identifies the unavailable source, and raise before merging
            # so no partial list escapes (Requirement 7.11). Other taxonomy
            # categories (e.g. authentication) pass through unchanged.
            raise _identify_unavailable_source(exc, source_name=chosen.name)
        for record in fetched:
            if len(records) >= effective_limit:
                break
            records.append(record.as_dict())
    return records


def _select_sources(
    sources: Sequence[BiodiversitySource], name: Optional[str]
) -> List[BiodiversitySource]:
    """Resolve the source(s) to query: all (merge) when ``name`` is omitted.

    When ``name`` is given it must match a configured source's
    :attr:`~BiodiversitySource.name`, otherwise a
    :class:`~geo_common.errors.ValidationError` naming the available sources is
    raised (Requirement 7.12).
    """
    if name is None:
        return [s for s in sources if getattr(s, "default_merge", True)]
    if not isinstance(name, str) or not name.strip():
        raise ValidationError(
            "source must be a non-empty source name",
            source=_SERVER_NAME,
            detail={"parameter": "source"},
        )
    for candidate in sources:
        if candidate.name == name:
            return [candidate]
    available = ", ".join(sorted(c.name for c in sources)) or "none"
    raise ValidationError(
        "unknown source %r; configured sources: %s" % (name, available),
        source=_SERVER_NAME,
        detail={"parameter": "source", "available": sorted(c.name for c in sources)},
    )


def _identify_unavailable_source(exc: GeoError, *, source_name: str) -> GeoError:
    """Re-tag an availability error with the logical source name (Req 7.11)."""
    if exc.category not in _AVAILABILITY_CATEGORIES or exc.source == source_name:
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


def _parse_gbif(payload: Any) -> List[OccurrenceRecord]:
    """Convert a GBIF occurrence-search payload into :class:`OccurrenceRecord`s.

    Reads the ``results`` array; each entry's stable ``key`` becomes the record
    id, with ``scientificName``, ``decimalLongitude`` / ``decimalLatitude``, and
    ``eventDate`` mapped onto the record's typed fields. The full GBIF entry is
    retained under ``properties`` (minus the promoted coordinate keys) so no
    source attribute is lost, and ``properties["source"]`` is set to ``"gbif"``.
    """
    if not isinstance(payload, dict):
        raise UpstreamError(
            "GBIF payload was not a JSON object",
            source="gbif",
        )
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise UpstreamError(
            "GBIF 'results' was not a list",
            source="gbif",
        )

    records: List[OccurrenceRecord] = []
    for entry in results:
        if not isinstance(entry, dict):
            continue
        properties = {k: v for k, v in entry.items()}
        properties["source"] = "gbif"
        key = entry.get("key")
        records.append(
            OccurrenceRecord(
                id=str(key) if key is not None else None,
                scientific_name=_as_optional_str(entry.get("scientificName")),
                longitude=_as_optional_float(entry.get("decimalLongitude")),
                latitude=_as_optional_float(entry.get("decimalLatitude")),
                event_date=_as_optional_str(entry.get("eventDate")),
                source="gbif",
                properties=properties,
            )
        )
    return records


def _parse_inaturalist(payload: Any) -> List[OccurrenceRecord]:
    """Convert an iNaturalist observations payload into :class:`OccurrenceRecord`s.

    Reads the ``results`` array; each entry's ``id`` becomes the record id,
    with the taxon name, coordinates (from ``geojson.coordinates`` ``[lon, lat]``
    or the ``location`` ``"lat,lng"`` string), and ``observed_on`` mapped onto
    the record's typed fields. The full entry is retained under ``properties``
    with ``properties["source"] = "inaturalist"``.
    """
    if not isinstance(payload, dict):
        raise UpstreamError(
            "iNaturalist payload was not a JSON object", source="inaturalist"
        )
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise UpstreamError(
            "iNaturalist 'results' was not a list", source="inaturalist"
        )

    records: List[OccurrenceRecord] = []
    for entry in results:
        if not isinstance(entry, dict):
            continue
        lon, lat = _inaturalist_coords(entry)
        taxon = entry.get("taxon")
        scientific_name = (
            taxon.get("name") if isinstance(taxon, dict) else None
        )
        properties = {k: v for k, v in entry.items()}
        properties["source"] = "inaturalist"
        key = entry.get("id")
        records.append(
            OccurrenceRecord(
                id=str(key) if key is not None else None,
                scientific_name=_as_optional_str(scientific_name),
                longitude=lon,
                latitude=lat,
                event_date=_as_optional_str(
                    entry.get("observed_on") or entry.get("observed_on_string")
                ),
                source="inaturalist",
                properties=properties,
            )
        )
    return records


def _parse_iucn(payload: Any, taxon: str) -> List[OccurrenceRecord]:
    """Convert an IUCN Red List v4 ``scientific_name`` response into records.

    The v4 response carries a ``taxon`` object (scientific name, ``sis_id``) and
    an ``assessments`` array. Each assessment becomes one record tagged
    ``source="iucn"`` whose ``properties`` hold the assessment (including the
    Red List ``category``/``red_list_category_code``); the record has no point
    coordinate, since an assessment describes a species' global status. When the
    payload has no assessments, a single status-less record for the taxon is
    returned so the lookup is still attributable.
    """
    if not isinstance(payload, dict):
        raise UpstreamError(
            "IUCN Red List payload was not a JSON object", source="iucn"
        )
    taxon_obj = payload.get("taxon") if isinstance(payload.get("taxon"), dict) else {}
    scientific_name = _as_optional_str(taxon_obj.get("scientific_name")) or taxon
    sis_id = taxon_obj.get("sis_id")
    assessments = payload.get("assessments")
    if not isinstance(assessments, list) or not assessments:
        return [
            OccurrenceRecord(
                id=str(sis_id) if sis_id is not None else None,
                scientific_name=scientific_name,
                source="iucn",
                properties={"source": "iucn", "taxon": taxon_obj},
            )
        ]

    records: List[OccurrenceRecord] = []
    for assessment in assessments:
        if not isinstance(assessment, dict):
            continue
        category = (
            assessment.get("red_list_category_code")
            or assessment.get("category")
        )
        properties = {k: v for k, v in assessment.items()}
        properties["source"] = "iucn"
        if category is not None:
            properties["category"] = category
        assessment_id = assessment.get("assessment_id")
        records.append(
            OccurrenceRecord(
                id=str(assessment_id) if assessment_id is not None else None,
                scientific_name=scientific_name,
                event_date=_as_optional_str(assessment.get("year_published")),
                source="iucn",
                properties=properties,
            )
        )
    return records


def _inaturalist_coords(entry: Dict[str, Any]) -> "tuple[Optional[float], Optional[float]]":
    """Extract ``(lon, lat)`` from an iNaturalist observation entry."""
    geojson = entry.get("geojson")
    if isinstance(geojson, dict):
        coords = geojson.get("coordinates")
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            return _as_optional_float(coords[0]), _as_optional_float(coords[1])
    location = entry.get("location")
    if isinstance(location, str) and "," in location:
        lat_s, _, lon_s = location.partition(",")
        return _as_optional_float(lon_s), _as_optional_float(lat_s)
    return None, None


def _as_optional_str(value: Any) -> Optional[str]:
    """Coerce ``value`` to a string, or ``None`` when absent/empty."""
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _as_optional_float(value: Any) -> Optional[float]:
    """Coerce ``value`` to a float, or ``None`` when absent/non-numeric."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

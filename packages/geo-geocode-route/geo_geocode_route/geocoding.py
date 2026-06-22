"""Address geocoding for ``geo-geocode-route`` (Requirements 7.4, 7.12).

``geocode`` turns an address into a :class:`~geo_geocode_route.models.Coordinate`
within 10 seconds (Requirement 7.4). It validates the address first - an
unparseable address is rejected with a ``ValidationError`` before any source
is queried (Requirement 7.12) - then tries the configured geocoders in order
and returns the first match.

Two open geocoders ship by default:

* :class:`NominatimSource` - OpenStreetMap's Nominatim search API.
* :class:`PhotonSource` - Komoot's Photon API (also OSM-backed).

(The design also names Pelias, and the routing side names OSRM/Valhalla/Amazon
Location; further sources slot in as additional :class:`GeocodeSource`
subclasses.)

Every outbound call goes through the shared
:class:`~geo_common.http.HttpClient`, so all sources inherit identical
retry/backoff, rate-limit handling, and the per-request timeout that enforces
the response bound (Requirement 5). A source that is unreachable or does not
respond surfaces as a taxonomy-classified availability error
(``NETWORK``/``UPSTREAM``) re-tagged with the logical source name; when no
source can resolve the address a :class:`~geo_common.errors.NotFoundError`
naming the address is raised.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from geo_common.errors import (
    AuthenticationError,
    ErrorCategory,
    GeoError,
    NetworkError,
    NotFoundError,
    UpstreamError,
    ValidationError,
)
from geo_common.http import HttpClient

from geo_geocode_route.models import Coordinate
from geo_geocode_route.validate import SERVER_NAME, validate_address

__all__ = [
    "GeocodeSource",
    "NominatimSource",
    "PhotonSource",
    "AmazonLocationSource",
    "default_geocoders",
    "geocode",
    "reverse_geocode",
]

#: Taxonomy categories that mean a configured source was unavailable -
#: unreachable or not responding within the per-request timeout
#: (Requirement 7.11). The shared :class:`HttpClient` maps a timeout/connection
#: failure onto ``NETWORK`` and an upstream 5xx / server rate-limit onto
#: ``UPSTREAM``; ``geocode`` re-tags such an error with the logical source name.
_AVAILABILITY_CATEGORIES = (ErrorCategory.NETWORK, ErrorCategory.UPSTREAM)

#: A descriptive User-Agent. Nominatim's usage policy requires identifying the
#: client; sending one on every geocode keeps the default source usable.
_USER_AGENT = "geo-geocode-route/0.1 (Geospatial Power Pack)"

DEFAULT_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
DEFAULT_NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
DEFAULT_PHOTON_URL = "https://photon.komoot.io/api"
DEFAULT_PHOTON_REVERSE_URL = "https://photon.komoot.io/reverse"

#: Default AWS region used to build the Amazon Location Places endpoints.
DEFAULT_AMAZON_LOCATION_REGION = "us-east-1"


class GeocodeSource:
    """Base class for an address geocoder queried by :func:`geocode`.

    A source exposes a human-readable :attr:`name` and an async
    :meth:`fetch` that returns the best-matching :class:`Coordinate` for a
    validated address, or ``None`` when the source has no match. Subclasses
    make all I/O through the shared :class:`HttpClient` and raise a
    taxonomy-classified :class:`~geo_common.errors.GeoError` on failure.

    A source may also support **reverse** geocoding via :meth:`reverse`
    (coordinate -> nearest address); the base returns ``None`` so a source
    that does not implement reverse simply contributes no reverse match.
    """

    name: str = "geocode-source"

    #: Whether this source is used in the *default* geocoder chain (when the
    #: caller does not name a ``source``). Credentialed providers that should be
    #: invoked only on explicit request - so an open, no-credential geocode is
    #: never blocked by them - set this ``False``; they remain reachable via
    #: ``source="..."``.
    default_chain: bool = True

    async def fetch(
        self, http: HttpClient, address: str
    ) -> Optional[Coordinate]:  # pragma: no cover - abstract
        raise NotImplementedError

    async def reverse(
        self, http: HttpClient, lon: float, lat: float
    ) -> Optional[Coordinate]:
        """Return the nearest address for ``(lon, lat)``, or ``None``.

        The default implementation returns ``None`` (no reverse support);
        sources that support reverse geocoding override this.
        """
        return None


class NominatimSource(GeocodeSource):
    """Geocoding via OpenStreetMap's Nominatim search API."""

    name = "nominatim"

    def __init__(
        self,
        url: str = DEFAULT_NOMINATIM_URL,
        *,
        reverse_url: str = DEFAULT_NOMINATIM_REVERSE_URL,
    ) -> None:
        self.url = url
        self.reverse_url = reverse_url

    async def fetch(self, http: HttpClient, address: str) -> Optional[Coordinate]:
        try:
            response = await http.get(
                self.url,
                params={"q": address, "format": "jsonv2", "limit": 1},
                headers={"User-Agent": _USER_AGENT},
            )
        except GeoError:
            raise
        if response.status_code >= 400:
            raise UpstreamError(
                "Nominatim returned HTTP %d" % response.status_code,
                source=self.name,
                detail={"status_code": response.status_code},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(
                "Nominatim returned a non-JSON response",
                source=self.name,
                original=str(exc),
            )
        return _parse_nominatim(payload)

    async def reverse(
        self, http: HttpClient, lon: float, lat: float
    ) -> Optional[Coordinate]:
        try:
            response = await http.get(
                self.reverse_url,
                params={"lon": lon, "lat": lat, "format": "jsonv2"},
                headers={"User-Agent": _USER_AGENT},
            )
        except GeoError:
            raise
        if response.status_code >= 400:
            raise UpstreamError(
                "Nominatim returned HTTP %d" % response.status_code,
                source=self.name,
                detail={"status_code": response.status_code},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(
                "Nominatim returned a non-JSON response",
                source=self.name,
                original=str(exc),
            )
        return _parse_nominatim_reverse(payload)


class PhotonSource(GeocodeSource):
    """Geocoding via Komoot's Photon API (OSM-backed)."""

    name = "photon"

    def __init__(
        self,
        url: str = DEFAULT_PHOTON_URL,
        *,
        reverse_url: str = DEFAULT_PHOTON_REVERSE_URL,
    ) -> None:
        self.url = url
        self.reverse_url = reverse_url

    async def fetch(self, http: HttpClient, address: str) -> Optional[Coordinate]:
        try:
            response = await http.get(
                self.url,
                params={"q": address, "limit": 1},
                headers={"User-Agent": _USER_AGENT},
            )
        except GeoError:
            raise
        if response.status_code >= 400:
            raise UpstreamError(
                "Photon returned HTTP %d" % response.status_code,
                source=self.name,
                detail={"status_code": response.status_code},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(
                "Photon returned a non-JSON response",
                source=self.name,
                original=str(exc),
            )
        return _parse_photon(payload)

    async def reverse(
        self, http: HttpClient, lon: float, lat: float
    ) -> Optional[Coordinate]:
        try:
            response = await http.get(
                self.reverse_url,
                params={"lon": lon, "lat": lat, "limit": 1},
                headers={"User-Agent": _USER_AGENT},
            )
        except GeoError:
            raise
        if response.status_code >= 400:
            raise UpstreamError(
                "Photon returned HTTP %d" % response.status_code,
                source=self.name,
                detail={"status_code": response.status_code},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(
                "Photon returned a non-JSON response",
                source=self.name,
                original=str(exc),
            )
        # Photon's reverse endpoint returns the same GeoJSON shape as search.
        return _parse_photon(payload)


class AmazonLocationSource(GeocodeSource):
    """Geocoding via the Amazon Location Service Places API (credentialed).

    Amazon Location is **credentialed**: every request authenticates with an
    API key (``AMAZON_LOCATION_API_KEY``) passed as the ``key`` query parameter,
    and the endpoint host embeds the AWS ``region``. Both forward geocoding
    (``/v2/geocode``, ``{"QueryText": ...}``) and reverse geocoding
    (``/v2/reverse-geocode``, ``{"QueryPosition": [lon, lat]}``) are supported.

    When no API key is configured the source raises an ``Error_Taxonomy``
    :class:`~geo_common.errors.AuthenticationError` naming the ``mcp.json`` key
    (never the secret value) before any request is made, so an unconfigured
    Amazon Location provider fails cleanly rather than silently.
    """

    name = "amazon-location"

    #: Credentialed: invoked only on explicit ``source="amazon-location"``,
    #: never in the default chain (so an open geocode is never blocked by it).
    default_chain = False

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        region: str = DEFAULT_AMAZON_LOCATION_REGION,
        url: Optional[str] = None,
        reverse_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or None
        self.region = region
        base = "https://places.geo.%s.amazonaws.com" % region
        self.url = url or "%s/v2/geocode" % base
        self.reverse_url = reverse_url or "%s/v2/reverse-geocode" % base

    def _require_key(self) -> str:
        if not self.api_key:
            raise AuthenticationError(
                "Amazon Location is not configured: set "
                "AMAZON_LOCATION_API_KEY in mcp.json",
                source=self.name,
                detail={"mcp_json_key": "AMAZON_LOCATION_API_KEY"},
            )
        return self.api_key

    async def _post(self, http: HttpClient, url: str, body: Any) -> Any:
        key = self._require_key()
        try:
            response = await http.post(
                url,
                params={"key": key},
                json=body,
                headers={"User-Agent": _USER_AGENT},
            )
        except GeoError:
            raise
        if response.status_code >= 400:
            raise UpstreamError(
                "Amazon Location returned HTTP %d" % response.status_code,
                source=self.name,
                detail={"status_code": response.status_code},
            )
        try:
            return response.json()
        except ValueError as exc:
            raise UpstreamError(
                "Amazon Location returned a non-JSON response",
                source=self.name,
                original=str(exc),
            )

    async def fetch(self, http: HttpClient, address: str) -> Optional[Coordinate]:
        payload = await self._post(
            http, self.url, {"QueryText": address, "MaxResults": 1}
        )
        return _parse_amazon_location(payload)

    async def reverse(
        self, http: HttpClient, lon: float, lat: float
    ) -> Optional[Coordinate]:
        payload = await self._post(
            http, self.reverse_url, {"QueryPosition": [lon, lat], "MaxResults": 1}
        )
        return _parse_amazon_location(payload)


def default_geocoders() -> List[GeocodeSource]:
    """The default geocoder set: Nominatim, then Photon."""
    return [NominatimSource(), PhotonSource()]


async def geocode(
    *,
    address: str,
    http: HttpClient,
    sources: Sequence[GeocodeSource],
    source: Optional[str] = None,
) -> Coordinate:
    """Return the coordinate for ``address`` within 10s (Requirement 7.4).

    Validates the address and ``source`` first (Requirement 7.12): an
    unparseable address or an unknown ``source`` name raises a
    :class:`~geo_common.errors.ValidationError` before any source is queried.

    ``source`` selects which geocoder(s) to try:

    * omitted (``None``) - try the default-chain geocoders in order
      (Nominatim, then Photon) and return the first match; and
    * a name (e.g. ``"amazon-location"``) - use only that geocoder, so a
      specific provider can be forced (a credentialed provider selected without
      its key raises an ``AuthenticationError`` naming the key).

    The chosen geocoders are tried in order through the shared
    :class:`HttpClient`; the first that resolves the address wins. If a source
    is unreachable, its availability error is recorded and the next is tried;
    only when **every** chosen source is unavailable is a re-tagged availability
    error raised. When all respond but none resolve the address, a
    :class:`~geo_common.errors.NotFoundError` naming the address is raised.
    """
    normalized = validate_address(address)
    chosen = _select_geocoders(sources, source)

    last_unavailable: Optional[GeoError] = None
    for candidate in chosen:
        try:
            match = await candidate.fetch(http, normalized)
        except GeoError as exc:
            # An unavailable source (timeout/unreachable/upstream 5xx) does not
            # fail the whole request; record it and try the next source. A
            # non-availability error (e.g. authentication) propagates as-is.
            if exc.category in _AVAILABILITY_CATEGORIES:
                last_unavailable = _identify_source(exc, source_name=candidate.name)
                continue
            raise
        if match is not None:
            match.source = candidate.name
            return match

    # No source resolved the address.
    if last_unavailable is not None:
        # At least one source was unavailable: surface the availability error.
        raise last_unavailable
    raise NotFoundError(
        "no geocoding source could resolve the address",
        source=SERVER_NAME,
        detail={"parameter": "address"},
    )


def _select_geocoders(
    sources: Sequence[GeocodeSource], name: Optional[str]
) -> List[GeocodeSource]:
    """Resolve which geocoders to query (Requirement 7.12).

    When ``name`` is omitted, returns the default-chain geocoders (open
    providers); when ``name`` is given it must match a configured source's
    :attr:`~GeocodeSource.name`, else a
    :class:`~geo_common.errors.ValidationError` naming the available sources is
    raised. Selecting a credentialed provider by name reaches it even when it is
    not in the default chain (its own credential guard then applies).
    """
    if name is None:
        return [s for s in sources if getattr(s, "default_chain", True)]
    if not isinstance(name, str) or not name.strip():
        raise ValidationError(
            "source must be a non-empty source name",
            source=SERVER_NAME,
            detail={"parameter": "source"},
        )
    for candidate in sources:
        if candidate.name == name:
            return [candidate]
    available = ", ".join(sorted(s.name for s in sources)) or "none"
    raise ValidationError(
        "unknown source %r; configured geocoders: %s" % (name, available),
        source=SERVER_NAME,
        detail={"parameter": "source", "available": sorted(s.name for s in sources)},
    )


async def reverse_geocode(
    *,
    lon: float,
    lat: float,
    http: HttpClient,
    sources: Sequence[GeocodeSource],
    source: Optional[str] = None,
) -> Coordinate:
    """Return the nearest address for ``(lon, lat)`` (reverse geocoding, Req 7.4).

    Validates the coordinate and ``source`` first (Requirement 7.12): a
    malformed/out-of-range ``lon``/``lat`` or an unknown ``source`` name raises
    a :class:`~geo_common.errors.ValidationError` before any source is queried.
    ``source`` selects the geocoder the same way as :func:`geocode` (omit for
    the default chain; name one - e.g. ``"amazon-location"`` - to force it).
    The chosen sources are tried in order; the first that resolves an address
    wins and the returned :class:`Coordinate` carries that address in its
    ``label``.

    Availability errors are recorded and the next source is tried; only when
    **every** chosen source is unavailable is a re-tagged availability error
    raised. When all respond but none can resolve the point, a
    :class:`~geo_common.errors.NotFoundError` is raised.
    """
    from geo_geocode_route.validate import validate_coordinate

    point = validate_coordinate({"lon": lon, "lat": lat}, parameter="location")
    chosen = _select_geocoders(sources, source)

    last_unavailable: Optional[GeoError] = None
    for candidate in chosen:
        try:
            match = await candidate.reverse(http, point.lon, point.lat)
        except GeoError as exc:
            if exc.category in _AVAILABILITY_CATEGORIES:
                last_unavailable = _identify_source(exc, source_name=candidate.name)
                continue
            raise
        if match is not None:
            match.source = candidate.name
            return match

    if last_unavailable is not None:
        raise last_unavailable
    raise NotFoundError(
        "no geocoding source could resolve the coordinate to an address",
        source=SERVER_NAME,
        detail={"parameter": "location", "lon": point.lon, "lat": point.lat},
    )


def _identify_source(exc: GeoError, *, source_name: str) -> GeoError:
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


def _parse_nominatim(payload: Any) -> Optional[Coordinate]:
    """Parse a Nominatim ``jsonv2`` search response into a :class:`Coordinate`.

    Nominatim returns a JSON array of matches, each with string ``lon``/``lat``
    and a ``display_name``. The first (highest-ranked) match is used; an empty
    array means no match (``None``).
    """
    if not isinstance(payload, list):
        raise UpstreamError(
            "Nominatim response was not a JSON array",
            source="nominatim",
        )
    if not payload:
        return None
    first = payload[0]
    if not isinstance(first, dict):
        raise UpstreamError(
            "Nominatim match was not a JSON object",
            source="nominatim",
        )
    return _coordinate_from(
        first.get("lon"),
        first.get("lat"),
        label=first.get("display_name"),
        source="nominatim",
    )


def _parse_nominatim_reverse(payload: Any) -> Optional[Coordinate]:
    """Parse a Nominatim ``reverse`` (``jsonv2``) response into a :class:`Coordinate`.

    Unlike search, reverse returns a **single** JSON object with string
    ``lon``/``lat`` and ``display_name``. An ``{"error": ...}`` body (no match
    for the point) yields ``None``.
    """
    if not isinstance(payload, dict):
        raise UpstreamError(
            "Nominatim reverse response was not a JSON object",
            source="nominatim",
        )
    if payload.get("error") or "lon" not in payload or "lat" not in payload:
        return None
    return _coordinate_from(
        payload.get("lon"),
        payload.get("lat"),
        label=payload.get("display_name"),
        source="nominatim",
    )


def _parse_photon(payload: Any) -> Optional[Coordinate]:
    """Parse a Photon GeoJSON response into a :class:`Coordinate`.

    Photon returns a GeoJSON ``FeatureCollection``; each feature's ``geometry``
    is a ``Point`` whose ``coordinates`` are ``[lon, lat]``. The first feature
    is used; an empty ``features`` list means no match (``None``).
    """
    if not isinstance(payload, dict):
        raise UpstreamError(
            "Photon response was not a JSON object", source="photon"
        )
    features = payload.get("features")
    if not isinstance(features, list):
        raise UpstreamError(
            "Photon 'features' was not a list", source="photon"
        )
    if not features:
        return None
    first = features[0]
    if not isinstance(first, dict):
        raise UpstreamError("Photon feature was not a JSON object", source="photon")
    geometry = first.get("geometry") or {}
    coords = geometry.get("coordinates") if isinstance(geometry, dict) else None
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        raise UpstreamError(
            "Photon feature had no point coordinates", source="photon"
        )
    properties = first.get("properties") or {}
    label = properties.get("name") if isinstance(properties, dict) else None
    return _coordinate_from(coords[0], coords[1], label=label, source="photon")


def _parse_amazon_location(payload: Any) -> Optional[Coordinate]:
    """Parse an Amazon Location Places ``ResultItems`` response.

    Both ``/v2/geocode`` and ``/v2/reverse-geocode`` return
    ``{"ResultItems": [{"Position": [lon, lat], "Address": {"Label": ...},
    "Title": ...}]}``. The first result is used; an empty/absent ``ResultItems``
    means no match (``None``).
    """
    if not isinstance(payload, dict):
        raise UpstreamError(
            "Amazon Location response was not a JSON object",
            source="amazon-location",
        )
    items = payload.get("ResultItems")
    if not isinstance(items, list) or not items:
        return None
    first = items[0]
    if not isinstance(first, dict):
        raise UpstreamError(
            "Amazon Location result was not a JSON object",
            source="amazon-location",
        )
    position = first.get("Position")
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        raise UpstreamError(
            "Amazon Location result had no position",
            source="amazon-location",
        )
    address = first.get("Address")
    label = address.get("Label") if isinstance(address, dict) else first.get("Title")
    return _coordinate_from(
        position[0], position[1], label=label, source="amazon-location"
    )


def _coordinate_from(
    lon: Any, lat: Any, *, label: Optional[str], source: str
) -> Coordinate:
    """Build a :class:`Coordinate` from a source's lon/lat, mapping bad data."""
    try:
        lon_f = float(lon)
        lat_f = float(lat)
    except (TypeError, ValueError) as exc:
        raise UpstreamError(
            "%s returned a non-numeric coordinate" % source,
            source=source,
            original=str(exc),
        )
    return Coordinate(lon=lon_f, lat=lat_f, label=label, source=source)

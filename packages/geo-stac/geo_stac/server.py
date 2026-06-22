"""The ``geo-stac`` MCP server (Pillar A, MVP).

:class:`GeoStacServer` is the :class:`~geo_common.server.BaseGeoServer`
subclass for STAC catalog discovery. It owns the shared
:class:`~geo_common.http.HttpClient` and registers :func:`geo_stac.search.stac_search`
as an MCP tool, binding the shared client into each invocation so retry/backoff
and rate-limit handling are inherited from ``geo-common`` (Requirement 5).

This module also fulfils the catalog/credential/error contract for ``geo-stac``
(task 5.2):

* :meth:`GeoStacServer.catalog_entries` registers the ``stac_search`` capability
  in the Resource Catalog naming ``geo-stac`` as provider (Requirements 2.1,
  11.3).
* :meth:`GeoStacServer.required_credentials` declares the ``mcp.json`` keys the
  wrapped STAC sources can use - both Optional, since open access works without
  them (Requirement 16.1).
* :meth:`GeoStacServer.stac_search` routes every wrapped-source error through
  :meth:`GeoStacServer.map_error` and refines it onto the ``Error_Taxonomy``
  (Requirement 11.2): an unreachable / >30s source becomes an **availability**
  error identifying the source with no partial results (Requirement 7.11), and
  a missing/invalid proprietary credential becomes an **authentication** error
  naming the credential with no partial data (Requirement 7.8).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from geo_common.errors import (
    AuthenticationError,
    GeoError,
    NetworkError,
)
from geo_common.http import HttpClient
from geo_common.models import (
    CatalogEntry,
    CredentialClassification,
    CredentialSpec,
    OpennessTier,
)
from geo_common.server import BaseGeoServer

from geo_stac.search import (
    DEFAULT_STAC_API_URL,
    KNOWN_STAC_ENDPOINTS,
    StacItem,
    StacSearchResult,
    stac_search,
    stac_search_multi,
)

__all__ = ["GeoStacServer", "main"]


class GeoStacServer(BaseGeoServer):
    """STAC catalog-discovery server (Pillar A, MVP).

    Registers the ``stac_search`` tool, routing it through this server's shared
    :class:`~geo_common.http.HttpClient` so all outbound STAC API calls inherit
    the Power Pack's retry/backoff/rate-limit behavior, and maps every
    wrapped-source failure onto the shared ``Error_Taxonomy``.
    """

    pillar = "A"
    server_name = "geo-stac"
    version = "0.2.0"

    #: ``geo-stac`` wraps open STAC APIs (Requirement 7.9 MVP). The two keys
    #: below are *Optional*: open access works without them, but configuring
    #: them unlocks higher rate limits / access-restricted collections. Both
    #: mirror ``bundle-manifest.json``'s ``geo-stac`` credential block.
    _CREDENTIAL_SPECS: Tuple[CredentialSpec, ...] = (
        CredentialSpec(
            source="Microsoft Planetary Computer",
            mcp_json_key="PC_SDK_SUBSCRIPTION_KEY",
            classification=CredentialClassification.OPTIONAL,
        ),
        CredentialSpec(
            source="NASA Earthdata / CMR-STAC",
            mcp_json_key="EARTHDATA_TOKEN",
            classification=CredentialClassification.OPTIONAL,
        ),
    )

    #: Maps a known credentialed STAC source host to the ``mcp.json`` key whose
    #: missing/invalid value an authentication failure should name (Req 7.8).
    _CREDENTIAL_KEY_BY_HOST = {
        "planetarycomputer.microsoft.com": "PC_SDK_SUBSCRIPTION_KEY",
        "planetarycomputer-staging.microsoft.com": "PC_SDK_SUBSCRIPTION_KEY",
        "cmr.earthdata.nasa.gov": "EARTHDATA_TOKEN",
    }

    def __init__(
        self,
        http: Optional[HttpClient] = None,
        *,
        api_url: str = DEFAULT_STAC_API_URL,
    ) -> None:
        super().__init__(http=http)
        self.api_url = api_url
        self.register_tool("stac_search", self.stac_search)
        self.register_tool("stac_search_multi", self.stac_search_multi)

    # ------------------------------------------------------------------
    # Catalog + credential declaration (Requirements 2.1, 11.3, 16.1)
    # ------------------------------------------------------------------

    def catalog_entries(self) -> List[CatalogEntry]:
        """Register ``geo-stac``'s capability in the Resource Catalog.

        Declares the ``stac_search`` capability with its pillar, description,
        and :class:`~geo_common.models.OpennessTier`, naming ``geo-stac`` as the
        provider (Requirements 2.1, 11.3). The wrapped STAC APIs (Earth Search,
        Planetary Computer, CMR-STAC, Copernicus, USGS) are open, so the entry
        is :attr:`~geo_common.models.OpennessTier.OPEN`.
        """
        return [
            CatalogEntry(
                name="stac_search",
                pillar=self.pillar,
                capability_description=(
                    "Search STAC catalogs (Earth Search / Element84, Microsoft "
                    "Planetary Computer, NASA CMR-STAC, Copernicus Data Space, "
                    "USGS) by bounding box and datetime range, returning matching "
                    "items with asset references and spatio-temporal metadata "
                    "(capped at 1,000 items)."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
            ),
            CatalogEntry(
                name="stac_search_multi",
                pillar=self.pillar,
                capability_description=(
                    "Federated STAC search across multiple catalogs (Earth "
                    "Search, Planetary Computer, CMR-STAC, Copernicus, USGS) "
                    "concurrently, merging and de-duplicating results with "
                    "per-source provenance and graceful degradation (partial "
                    "results when a source is unavailable)."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
            ),
        ]

    def required_credentials(self) -> List[CredentialSpec]:
        """Declare the ``mcp.json`` keys ``geo-stac``'s sources can use.

        Both are :attr:`~geo_common.models.CredentialClassification.OPTIONAL`:
        the wrapped STAC sources serve open data without a credential, so an
        absent key never blocks startup (Requirement 16.5).
        """
        return list(self._CREDENTIAL_SPECS)

    # ------------------------------------------------------------------
    # Tool entry point + error mapping (Requirements 7.8, 7.11, 11.2)
    # ------------------------------------------------------------------

    async def stac_search(
        self,
        *,
        bbox: Tuple[float, float, float, float],
        datetime_range: Tuple[str, str],
        collections: Optional[List[str]] = None,
        limit: int = 1000,
    ) -> List[StacItem]:
        """Search the configured STAC catalog (see :func:`geo_stac.search.stac_search`).

        Validation errors for malformed parameters (Requirement 7.12) are raised
        by the wrapped :func:`geo_stac.search.stac_search` before any network
        call and propagate unchanged. Any failure reaching the wrapped STAC
        source is mapped onto the shared ``Error_Taxonomy`` (Requirement 11.2)
        and refined by :meth:`_refine_stac_error`:

        * an unreachable / >30s source -> an **availability** error
          (``NETWORK``) identifying the source, with no partial results
          (Requirement 7.11);
        * a missing/invalid proprietary credential -> an **authentication**
          error naming the credential, with no partial data (Requirement 7.8).
        """
        try:
            return await stac_search(
                bbox=bbox,
                datetime_range=datetime_range,
                collections=collections,
                limit=limit,
                http=self.http,
                api_url=self.api_url,
            )
        except GeoError as exc:
            # Already taxonomy-classified (HttpClient / inner validation).
            raise self._refine_stac_error(exc) from exc
        except Exception as exc:  # pragma: no cover - defensive catch-all
            # Unmapped library/transport error -> UPSTREAM, original retained.
            mapped = self.map_error(exc, source=self.server_name)
            raise self._refine_stac_error(mapped) from exc

    async def stac_search_multi(
        self,
        *,
        bbox: Tuple[float, float, float, float],
        datetime_range: Tuple[str, str],
        collections: Optional[List[str]] = None,
        limit: int = 1000,
        dedupe: str = "scene",
    ) -> StacSearchResult:
        """Federated STAC search across the known catalogs (graceful degradation).

        Queries Earth Search, Planetary Computer, CMR-STAC, Copernicus, and USGS
        concurrently through the shared HTTP client, **round-robin merges** the
        matches (one per catalog in turn) up to ``limit``, and returns a
        :class:`~geo_stac.search.StacSearchResult` with per-source provenance.
        ``dedupe`` controls cross-catalog de-duplication: ``"scene"`` (default)
        collapses the same physical scene published by multiple catalogs under
        different ids into one item (by acquisition instant + MGRS tile), while
        ``"id"`` keeps each catalog's copy. A malformed bbox/datetime/limit or an
        unknown ``dedupe`` raises a ``ValidationError`` before any network call
        (Requirement 7.12); an unavailable source does not fail the search - it
        is reported in ``sources`` and the result is marked ``partial``
        (Requirement 7.11-style graceful degradation).
        """
        return await stac_search_multi(
            bbox=bbox,
            datetime_range=datetime_range,
            collections=collections,
            limit=limit,
            http=self.http,
            endpoints=dict(KNOWN_STAC_ENDPOINTS),
            dedupe=dedupe,
        )

    def _refine_stac_error(self, exc: GeoError) -> GeoError:
        """Refine a taxonomy-classified STAC error for Requirements 7.8 / 7.11.

        * :class:`~geo_common.errors.NetworkError` (the source is unreachable or
          exceeded the 30s budget) is re-stated as an **availability** error
          that identifies the unavailable source and records that no partial
          results were returned (Requirement 7.11).
        * :class:`~geo_common.errors.AuthenticationError` (the source rejected
          the credential, e.g. HTTP 401) is re-stated to **name** the proprietary
          credential associated with the source, with no partial data
          (Requirement 7.8).

        Every other category (``validation``, ``authorization``, ``rate-limit``,
        ``not-found``, ``upstream``) already names a correct taxonomy mapping of
        the STAC API error and is returned unchanged (Requirement 11.2).
        """
        source = exc.source or self._source_label()

        if isinstance(exc, NetworkError):
            detail = dict(exc.detail or {})
            detail.update({"source": source, "partial_results": False})
            return NetworkError(
                f"STAC source {source!r} is unavailable or did not respond "
                "within 30 seconds; no partial results were returned",
                source=source,
                detail=detail,
                original=exc.original,
            )

        if isinstance(exc, AuthenticationError):
            credential_key = self._credential_key_for_source(source)
            detail = dict(exc.detail or {})
            detail.update({"source": source, "partial_results": False})
            if credential_key:
                detail["missing_or_invalid_key"] = credential_key
                message = (
                    f"STAC source {source!r} rejected the request: the "
                    f"{credential_key!r} credential is missing or invalid; no "
                    "data was returned"
                )
            else:
                message = (
                    f"STAC source {source!r} rejected the request: a missing or "
                    "invalid credential; no data was returned"
                )
            return AuthenticationError(
                message,
                source=source,
                detail=detail,
                original=exc.original,
            )

        return exc

    def _source_label(self) -> str:
        """A secret-free identifier for the configured STAC source."""
        try:
            host = httpx_url_host(self.api_url)
        except Exception:  # pragma: no cover - defensive; malformed URL
            host = None
        return host or self.server_name

    def _credential_key_for_source(self, source: Optional[str]) -> Optional[str]:
        """Resolve the ``mcp.json`` key for a credentialed STAC source host."""
        if not source:
            return None
        return self._CREDENTIAL_KEY_BY_HOST.get(source)


def httpx_url_host(url: str) -> Optional[str]:
    """Return the host of ``url`` (secret-free), or ``None`` if unparseable."""
    import httpx

    return httpx.URL(url).host or None


def main() -> None:
    """Console entry point: serve geo-stac over MCP stdio.

    Runs the startup credential guard, then serves this server's registered
    tools over stdin/stdout via the shared geo-common MCP runtime until the
    client disconnects.
    """
    GeoStacServer().run()


if __name__ == "__main__":  # pragma: no cover - module CLI shim
    main()

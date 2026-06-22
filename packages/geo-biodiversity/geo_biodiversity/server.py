"""The ``geo-biodiversity`` MCP server (Pillar A, expansion).

Wires the species-occurrence connector (:mod:`geo_biodiversity.occurrences`)
into the shared :class:`~geo_common.server.BaseGeoServer` contract and exposes
``species_occurrences`` as an MCP tool (Requirements 7.7, 7.12). It also
declares the server's Resource Catalog entry and credential specs so the Power
Hub gets a uniform view of the capability.

All outbound calls share the inherited :class:`~geo_common.http.HttpClient` so
they get identical retry/backoff and the 30-second per-request timeout that
backs the 30-second response bound of Requirement 7.7.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from geo_common.http import HttpClient
from geo_common.models import (
    CatalogEntry,
    CredentialClassification,
    CredentialSpec,
    OpennessTier,
)
from geo_common.server import BaseGeoServer

from geo_biodiversity.occurrences import (
    BiodiversitySource,
    IUCNSource,
    default_sources,
    species_occurrences as _species_occurrences,
)
from geo_biodiversity.validation import DEFAULT_MAX_RECORDS

__all__ = ["GeoBiodiversityServer", "INSTALL_COMMAND", "main"]

#: The ``uvx`` command that installs this server (Requirements 2.6, 16.1). The
#: Resource Catalog surfaces it on entries whose provider is not yet installed.
INSTALL_COMMAND = "uvx geo-biodiversity"


class GeoBiodiversityServer(BaseGeoServer):
    """Pillar A (expansion) server exposing species-occurrence access (Req 7.7).

    Holds the configurable source set (GBIF by default) and the maximum record
    cap, and registers ``species_occurrences`` as an MCP tool. All outbound
    calls share the inherited :class:`HttpClient`.
    """

    pillar = "A"
    server_name = "geo-biodiversity"
    version = "0.2.0"

    #: ``geo-biodiversity``'s default source (GBIF) is open and needs no
    #: credential. iNaturalist *can* use an optional token for authenticated
    #: queries, and the IUCN Red List source requires an account token; both are
    #: Optional so an absent key never blocks startup (Req 16.5). This mirrors
    #: ``bundle-manifest.json``'s ``geo-biodiversity`` credential block.
    _CREDENTIAL_SPECS = (
        CredentialSpec(
            source="iNaturalist",
            mcp_json_key="INATURALIST_TOKEN",
            classification=CredentialClassification.OPTIONAL,
        ),
        CredentialSpec(
            source="IUCN Red List",
            mcp_json_key="IUCN_TOKEN",
            classification=CredentialClassification.OPTIONAL,
        ),
    )

    def __init__(
        self,
        *,
        sources: Optional[Sequence[BiodiversitySource]] = None,
        max_records: int = DEFAULT_MAX_RECORDS,
        http: Optional[HttpClient] = None,
    ) -> None:
        super().__init__(http=http)
        self.sources: List[BiodiversitySource] = (
            list(sources) if sources is not None else self._default_source_set()
        )
        self.max_records = max_records
        self.register_tool("species_occurrences", self.species_occurrences)

    @staticmethod
    def _default_source_set() -> List[BiodiversitySource]:
        """The default occurrence sources (GBIF + iNaturalist), plus IUCN.

        GBIF and iNaturalist are open (iNaturalist with an optional token) and
        form the default *merged* occurrence result. The IUCN Red List source is
        always **selectable** (``source="iucn"``) but excluded from the default
        merge (``default_merge=False``) because it returns conservation status,
        not point occurrences. It reads ``IUCN_TOKEN``; selecting it without a
        token yields a clean ``AuthenticationError`` naming the key (mirroring
        the NOAA CDO pattern), so an unconfigured IUCN fails informatively rather
        than looking like an unknown source.
        """
        sources: List[BiodiversitySource] = list(
            default_sources(
                inaturalist_token=os.environ.get("INATURALIST_TOKEN") or None
            )
        )
        sources.append(IUCNSource(token=os.environ.get("IUCN_TOKEN") or None))
        return sources

    def catalog_entries(self) -> List[CatalogEntry]:
        """Register ``geo-biodiversity``'s capability in the Resource Catalog.

        Declares the ``species_occurrences`` capability for the Power Hub's
        catalog (Requirements 2.1, 11.3). The default source (GBIF) is open
        data, so the entry is tier :attr:`OpennessTier.OPEN`, with
        ``geo-biodiversity`` named as the provider and the ``uvx`` install
        command attached (Requirement 2.6).
        """
        return [
            CatalogEntry(
                name="species_occurrences",
                pillar=self.pillar,
                capability_description=(
                    "Retrieve species occurrence records for a spatial extent "
                    "(optionally filtered by taxon) from open biodiversity "
                    "sources (GBIF and iNaturalist); results are capped at a "
                    "configurable maximum (default 10,000 records)."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                install_command=INSTALL_COMMAND,
            )
        ]

    def required_credentials(self) -> List[CredentialSpec]:
        """Declare the ``mcp.json`` keys this server can use (Req 16.1).

        The default GBIF source is open and needs no credential; the
        ``INATURALIST_TOKEN`` key is :attr:`CredentialClassification.OPTIONAL`,
        so an absent key never blocks startup (Requirement 16.5).
        """
        return list(self._CREDENTIAL_SPECS)

    async def species_occurrences(
        self,
        *,
        bbox: Sequence[float],
        taxon: Optional[str] = None,
        limit: Optional[int] = None,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return occurrence records within ``bbox`` from the configured sources.

        Validates ``bbox``, ``taxon``, ``limit``, and ``source`` before querying
        any source (Requirement 7.12). The number of records returned is capped
        at the configured maximum (default 10,000); a larger requested ``limit``
        is clamped down to it (Requirement 7.7).

        ``source`` selects the provider: omit it to query the default
        occurrence sources (GBIF + iNaturalist) and **merge** their records;
        pass a name (``"gbif"``, ``"inaturalist"``, or ``"iucn"`` when
        configured) to query just that one. IUCN is opt-in (it returns
        conservation status, not point occurrences) and is reachable only by
        ``source="iucn"``.
        """
        effective_limit = self.max_records if limit is None else limit
        return await _species_occurrences(
            bbox=bbox,
            http=self.http,
            sources=self.sources,
            taxon=taxon,
            limit=effective_limit,
            source=source,
        )


def main() -> None:
    """Console entry point: serve geo-biodiversity over MCP stdio.

    Runs the startup credential guard, then serves this server's registered
    tools over stdin/stdout via the shared geo-common MCP runtime until the
    client disconnects.
    """
    GeoBiodiversityServer().run()


if __name__ == "__main__":  # pragma: no cover - module CLI shim
    main()

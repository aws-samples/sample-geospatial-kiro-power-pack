"""The ``geo-vector`` MCP server (Pillar A, MVP).

Wires the vector-feature connector (:mod:`geo_vector.features`) into the shared
:class:`~geo_common.server.BaseGeoServer` contract and exposes
``vector_features`` as an MCP tool (Requirements 7.3, 7.12). It also registers
the server's Resource Catalog entry and credential specs and ensures source
failures surface as Error_Taxonomy availability errors identifying the
unavailable source (Requirements 2.1, 7.11, 11.2, 11.3).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from geo_common.http import HttpClient
from geo_common.models import (
    CatalogEntry,
    CredentialSpec,
    OpennessTier,
)
from geo_common.server import BaseGeoServer

from geo_vector.bbox import DEFAULT_MAX_AREA_KM2
from geo_vector.features import (
    DEFAULT_MAX_FEATURES,
    VectorSource,
    default_sources,
    vector_features as _vector_features,
)
from geo_vector.models import FeatureCollection

__all__ = ["GeoVectorServer", "INSTALL_COMMAND", "main"]

#: The ``uvx`` command that installs this server (Requirements 2.6, 16.1). The
#: Resource Catalog surfaces it on entries whose provider is not yet installed
#: so a user can install the capability they searched for.
INSTALL_COMMAND = "uvx geo-vector"


class GeoVectorServer(BaseGeoServer):
    """Pillar A (MVP) server exposing vector feature access (Req 7.3, 7.12).

    Holds the configurable source set (OpenStreetMap via Overpass by default;
    Overture Maps is supported but ships disabled because it has no public
    REST endpoint and must be configured with an explicit GeoJSON service) and
    the maximum bounding-box area, and registers ``vector_features`` as an MCP
    tool. All outbound calls share the inherited :class:`HttpClient` so they
    get identical retry/backoff and the 30-second per-request timeout.
    """

    pillar = "A"
    server_name = "geo-vector"
    version = "0.2.0"

    def __init__(
        self,
        *,
        sources: Optional[Sequence[VectorSource]] = None,
        max_area_km2: float = DEFAULT_MAX_AREA_KM2,
        max_features: int = DEFAULT_MAX_FEATURES,
        http: Optional[HttpClient] = None,
    ) -> None:
        super().__init__(http=http)
        self.sources: List[VectorSource] = (
            list(sources) if sources is not None else default_sources()
        )
        self.max_area_km2 = max_area_km2
        self.max_features = max_features
        self.register_tool("vector_features", self.vector_features)

    def catalog_entries(self) -> List[CatalogEntry]:
        """Register ``geo-vector``'s capabilities in the Resource Catalog.

        Declares the ``vector_features`` capability for the Power Hub's catalog
        (Requirements 2.1, 11.3). The default source - OpenStreetMap via the
        Overpass API - is open data (Overture Maps is supported as an optional,
        explicitly configured source), so the entry is tier
        :attr:`OpennessTier.OPEN`. The wrapping ``geo-vector`` server names
        itself as the provider (Requirement 11.3), and the entry carries the
        ``uvx`` install command so the catalog can surface it when the provider
        is not yet installed (Requirement 2.6).
        """
        return [
            CatalogEntry(
                name="vector_features",
                pillar=self.pillar,
                capability_description=(
                    "Retrieve vector features (points, lines, polygons) for a "
                    "spatial extent from OpenStreetMap (Overpass) by default, "
                    "with optional Overture Maps support, merged into one "
                    "GeoJSON FeatureCollection; bounding-box area is capped at a "
                    "configurable maximum (default 2,500 km²)."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                install_command=INSTALL_COMMAND,
            )
        ]

    def required_credentials(self) -> List[CredentialSpec]:
        """Declare the ``mcp.json`` credentials this server needs (Req 16.1).

        The default OpenStreetMap/Overpass source is open and requires no
        credential (and the optional Overture source, when configured, points at
        a customer-provided endpoint), so ``geo-vector`` declares none. With no
        Required credentials, the startup credential guard always lets the
        server start (Requirement 16.5).
        """
        return []

    async def vector_features(
        self,
        *,
        bbox: Sequence[float],
        layers: Optional[Sequence[str]] = None,
        max_area_km2: Optional[float] = None,
        max_features: Optional[int] = None,
    ) -> FeatureCollection:
        """Return features within ``bbox`` from OpenStreetMap + Overture (Req 7.3).

        Validates the bbox and its area before querying any source: a bbox area
        greater than the configured maximum (default 2,500 km²), or a malformed
        bbox, raises an ``Error_Taxonomy`` ``ValidationError`` (Requirement
        7.12). On a valid request the configured sources are queried through
        the shared HTTP client and merged into one :class:`FeatureCollection`.
        The merged result is capped at ``max_features`` (default 2,000); a
        request matching more raises a ``ValidationError`` asking for a narrower
        bbox or a ``layers`` filter, so an oversized collection is never
        returned.
        """
        effective_max = self.max_area_km2 if max_area_km2 is None else max_area_km2
        effective_max_features = (
            self.max_features if max_features is None else max_features
        )
        return await _vector_features(
            bbox=bbox,
            http=self.http,
            sources=self.sources,
            layers=layers,
            max_area_km2=effective_max,
            max_features=effective_max_features,
        )


def main() -> None:
    """Console entry point: serve geo-vector over MCP stdio.

    Runs the startup credential guard, then serves this server's registered
    tools over stdin/stdout via the shared geo-common MCP runtime until the
    client disconnects.
    """
    GeoVectorServer().run()

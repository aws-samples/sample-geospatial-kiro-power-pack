"""The ``geo-ogc`` MCP server (Pillar A).

Bridges **OGC API - Features** services into the pack. ``geo-ogc`` exposes one
tool, ``ogc_features``, that fetches the GeoJSON features for a spatial extent
from any conformant OGC API - Features service (pygeoapi, GeoServer's OGC API,
ldproxy, …) named per call. OGC API - Features is an open standard, so the
server needs no credential; the service base URL is a tool parameter, so a user
points it at whichever instance they want.

This wires the feature connector (:mod:`geo_ogc.features`) into the shared
:class:`~geo_common.server.BaseGeoServer` contract: validation rejects a
malformed request before any network call (Requirement 7.12), failures map onto
the ``Error_Taxonomy``, and the capability is registered in the Resource
Catalog (Req 2.1).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from geo_common.http import HttpClient
from geo_common.models import CatalogEntry, CredentialSpec, OpennessTier
from geo_common.server import BaseGeoServer

from geo_ogc.features import DEFAULT_FEATURES, ogc_features as _ogc_features
from geo_ogc.models import FeatureCollectionResult

__all__ = ["GeoOgcServer", "INSTALL_COMMAND", "main"]

#: The ``uvx`` command that installs this server (Req 2.6; bundle-manifest).
INSTALL_COMMAND = "uvx geo-ogc"


class GeoOgcServer(BaseGeoServer):
    """Pillar A server exposing OGC API - Features access.

    Registers the single ``ogc_features`` tool. ``geo-ogc`` is open and
    credential-free (OGC API - Features is an open standard and the service URL
    is supplied per call), so it declares no ``mcp.json`` credentials and always
    starts.
    """

    pillar = "A"
    server_name = "geo-ogc"
    version = "0.2.0"

    def __init__(self, http: Optional[HttpClient] = None) -> None:
        super().__init__(http=http)
        self.register_tool("ogc_features", self.ogc_features)

    async def ogc_features(
        self,
        *,
        endpoint: str,
        collection: str,
        bbox: Sequence[float],
        limit: int = DEFAULT_FEATURES,
        datetime_range: Optional[str] = None,
    ) -> FeatureCollectionResult:
        """Fetch features from an OGC API - Features ``collection`` within ``bbox``.

        ``endpoint`` is the OGC API base URL (e.g.
        ``https://demo.pygeoapi.io/master``) and ``collection`` is a collection
        id at that service. ``bbox`` is ``(west, south, east, north)`` in
        EPSG:4326; ``limit`` caps the number of features (bounded server-side);
        ``datetime_range`` optionally filters by an ISO-8601 instant or interval.

        Validates all parameters before any network call; a malformed
        bbox/limit/endpoint/collection raises an ``Error_Taxonomy``
        ``ValidationError`` (Requirement 7.12). On a valid request the service
        is queried through the shared HTTP client and the matching features are
        returned as a :class:`~geo_ogc.models.FeatureCollectionResult`.
        """
        return await _ogc_features(
            endpoint=endpoint,
            collection=collection,
            bbox=bbox,
            http=self.http,
            limit=limit,
            datetime_range=datetime_range,
        )

    def catalog_entries(self) -> "List[CatalogEntry]":
        """Register ``geo-ogc``'s capability in the Resource Catalog (Req 2.1)."""
        return [
            CatalogEntry(
                name="ogc_features",
                pillar=self.pillar,
                capability_description=(
                    "Fetch GeoJSON features for a spatial extent from any OGC "
                    "API - Features service (pygeoapi, GeoServer, ldproxy) by "
                    "endpoint + collection id."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                install_command=INSTALL_COMMAND,
            )
        ]

    def required_credentials(self) -> "List[CredentialSpec]":
        """``geo-ogc`` is open and credential-free, so this is empty (Req 16.5)."""
        return []


def main() -> None:
    """Console entry point: serve geo-ogc over MCP stdio."""
    GeoOgcServer().run()


if __name__ == "__main__":  # pragma: no cover - module CLI shim
    main()

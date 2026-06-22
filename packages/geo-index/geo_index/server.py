"""The ``geo-index`` MCP server (Pillar B, expansion).

``geo-index`` provides local, pure-function spatial indexing built on the H3
and S2 libraries. This module wires the cell-lookup logic
(:mod:`geo_index.indexing`) into the shared
:class:`~geo_common.server.BaseGeoServer` contract and registers it as an MCP
tool.

Task 14.9 delivers the indexing capability:

* ``index_cell`` - return the H3 cell id (resolutions 0-15) or S2 cell id
  (levels 0-30) for a coordinate at a requested resolution (Requirement 8.9),
  raising a taxonomy ``ValidationError`` for an out-of-range resolution or a
  coordinate outside longitude ``[-180, 180]`` / latitude ``[-90, 90]``
  (Requirement 8.10).

``geo-index`` is open and credential-free: all logic is local (no network or
proprietary source), so it declares no ``mcp.json`` credentials and always
starts.
"""

from __future__ import annotations

from typing import List

from geo_common.models import CatalogEntry, CredentialSpec, OpennessTier
from geo_common.server import BaseGeoServer

from geo_index.indexing import index_cell as _index_cell

__all__ = ["GeoIndexServer", "main"]


class GeoIndexServer(BaseGeoServer):
    """Pillar B (expansion) server exposing H3 / S2 spatial-index lookups.

    Registers the single ``index_cell`` tool. ``geo-index`` is open and
    credential-free (all logic is local), so it starts without any configured
    credentials.
    """

    pillar = "B"
    server_name = "geo-index"
    version = "0.2.0"

    #: The native capabilities ``geo-index`` implements itself (h3 / s2geometry).
    #: Each is an open-tier catalog entry provided by ``geo-index`` (Req 2.1).
    _NATIVE_CAPABILITIES = (
        (
            "index_cell",
            "Return the H3 (res 0-15) or S2 (level 0-30) spatial-index cell id "
            "for a longitude/latitude coordinate at a requested resolution.",
        ),
    )

    def __init__(self, http=None) -> None:
        super().__init__(http=http)
        self.register_tool("index_cell", self.index_cell)

    async def index_cell(
        self,
        *,
        lon: float,
        lat: float,
        scheme: str,
        resolution: int,
    ) -> str:
        """Return the spatial-index cell id for ``(lon, lat)`` (Req 8.9).

        For ``scheme == "h3"`` returns the H3 cell id at ``resolution`` (0-15);
        for ``scheme == "s2"`` returns the S2 cell id at the requested level
        (0-30). The result is well-formed for the scheme and deterministic -
        the same input always yields the same id (design Property 15).

        Raises :class:`~geo_common.errors.ValidationError` for an unknown
        scheme, an out-of-range resolution, or a coordinate outside longitude
        ``[-180, 180]`` / latitude ``[-90, 90]`` (Requirement 8.10), producing
        no output.
        """
        return _index_cell(
            lon=lon,
            lat=lat,
            scheme=scheme,
            resolution=resolution,
            source=self.server_name,
        )

    # ------------------------------------------------------------------
    # Catalog + credential declaration (Req 2.1)
    # ------------------------------------------------------------------

    def catalog_entries(self) -> "List[CatalogEntry]":
        """Register every ``geo-index`` capability in the Resource Catalog (Req 2.1).

        Returns the native h3/s2 capabilities (open tier, provided by
        ``geo-index``).
        """
        return [
            CatalogEntry(
                name=name,
                pillar=self.pillar,
                capability_description=description,
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                installed=True,
            )
            for name, description in self._NATIVE_CAPABILITIES
        ]

    def required_credentials(self) -> "List[CredentialSpec]":
        """``geo-index`` is open and credential-free, so this is empty.

        All indexing is local (h3 / s2geometry), so ``geo-index`` declares no
        ``mcp.json`` credential keys and always starts (Requirement 16.5).
        """
        return []


def main() -> None:
    """Console entry point: serve geo-index over MCP stdio.

    Runs the startup credential guard, then serves this server's registered
    tools over stdin/stdout via the shared geo-common MCP runtime until the
    client disconnects.
    """
    GeoIndexServer().run()

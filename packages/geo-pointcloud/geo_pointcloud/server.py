"""The ``geo-pointcloud`` MCP server (Pillar B, expansion).

``geo-pointcloud`` reads and writes point-cloud data in Cloud-Optimized Point
Cloud (COPC) format (Requirement 8.11). This module wires the read/write core
(:mod:`geo_pointcloud.pointcloud`) into the shared
:class:`~geo_common.server.BaseGeoServer` contract and exposes
``read_pointcloud`` and ``write_pointcloud`` as MCP tools.

It also declares the server's Resource Catalog entries (one per capability -
Req 2.1, 11.3) and credential specs (Req 16.1), and inherits the taxonomy error
mapping from :class:`~geo_common.server.BaseGeoServer` so any
library/transport/upstream failure maps onto exactly one ``Error_Taxonomy``
category (Req 11.2, 11.5).
"""

from __future__ import annotations

from typing import List, Optional

from geo_common.models import (
    CatalogEntry,
    CredentialClassification,
    CredentialSpec,
    OpennessTier,
)
from geo_common.server import BaseGeoServer

from geo_pointcloud.models import FormatResult, GeoWindow, PointCloudChunk
from geo_pointcloud.pointcloud import (
    PointCloudBackend,
    read_pointcloud as _read_pointcloud,
    write_pointcloud as _write_pointcloud,
)

__all__ = ["GeoPointcloudServer", "INSTALL_COMMAND", "main"]

#: The ``uvx`` command that installs this server (Req 2.6, 16.1).
INSTALL_COMMAND = "uvx geo-pointcloud"


class GeoPointcloudServer(BaseGeoServer):
    """Pillar B (expansion) server for COPC read/write (Requirement 8.11).

    Registers ``read_pointcloud`` and ``write_pointcloud`` as MCP tools. The
    read/write engine is pluggable and the default routes per operation
    (:class:`~geo_pointcloud.pointcloud.SmartCopcBackend`): it reads a real
    ``.copc.laz`` / ``.laz`` (local or remote) via ``laspy`` (the ``[copc]``
    extra, COPC octree windowed reads) or PDAL (the ``[pdal]`` extra), and writes
    standards-compliant COPC via PDAL when installed. With no extra it falls back
    to a portable, lossless local container that is explicitly not interoperable
    COPC. COPC/LAZ are open formats and all processing is local; the two AWS keys
    it declares are *Optional* (private S3 assets only), so an absent key never
    blocks startup and the server always starts (Req 16.5).
    """

    pillar = "B"
    server_name = "geo-pointcloud"
    version = "0.3.0"

    #: Both AWS keys are Optional: public COPC assets work without them, while
    #: private S3 point-cloud assets can supply them (bundle-manifest.json
    #: ``geo-pointcloud`` credential block). Optional credentials never block
    #: startup (Requirement 16.5).
    _CREDENTIAL_SPECS = (
        CredentialSpec(
            source="AWS S3 (COPC assets)",
            mcp_json_key="AWS_ACCESS_KEY_ID",
            classification=CredentialClassification.OPTIONAL,
        ),
        CredentialSpec(
            source="AWS S3 (COPC assets)",
            mcp_json_key="AWS_SECRET_ACCESS_KEY",
            classification=CredentialClassification.OPTIONAL,
        ),
    )

    def __init__(
        self,
        *,
        backend: Optional[PointCloudBackend] = None,
        http=None,
    ) -> None:
        super().__init__(http=http)
        self.backend = backend
        self.register_tool("read_pointcloud", self.read_pointcloud)
        self.register_tool("write_pointcloud", self.write_pointcloud)

    # ------------------------------------------------------------------
    # Tools (Requirement 8.11)
    # ------------------------------------------------------------------

    async def read_pointcloud(
        self,
        *,
        copc_href: str,
        bounds: Optional[GeoWindow] = None,
    ) -> PointCloudChunk:
        """Read a COPC point cloud, optionally clipped to ``bounds`` (Req 8.11).

        Returns a :class:`PointCloudChunk`. When ``bounds`` is given only the
        points inside that :class:`GeoWindow` are returned. A missing source
        raises an ``Error_Taxonomy`` ``not-found`` error and a malformed request
        a ``validation`` error.
        """
        return _read_pointcloud(
            copc_href=copc_href, bounds=bounds, backend=self.backend
        )

    async def write_pointcloud(
        self,
        *,
        points: PointCloudChunk,
        dst_href: str,
    ) -> FormatResult:
        """Write a point-cloud chunk to COPC at ``dst_href`` (Req 8.11).

        Returns a :class:`FormatResult` describing the output. The written cloud
        read back via :meth:`read_pointcloud` preserves the point set (design
        Property 20).
        """
        return _write_pointcloud(
            points=points, dst_href=dst_href, backend=self.backend
        )

    # ------------------------------------------------------------------
    # Catalog + credential declaration (Req 2.1, 11.3, 16.1)
    # ------------------------------------------------------------------

    def catalog_entries(self) -> "List[CatalogEntry]":
        """Register ``geo-pointcloud``'s capabilities in the Resource Catalog.

        Declares the ``read_pointcloud`` and ``write_pointcloud`` capabilities
        (Req 2.1, 11.3), each naming this server as ``provider_server`` and
        carrying the ``uvx`` install command for surfacing when the provider is
        not yet installed (Req 2.6). COPC is an open format, so both entries are
        tier :attr:`OpennessTier.OPEN`.
        """
        return [
            CatalogEntry(
                name="read_pointcloud",
                pillar=self.pillar,
                capability_description=(
                    "Read a point cloud, optionally clipped to a spatial window. "
                    "Reads a real Cloud-Optimized Point Cloud / LAZ (local or "
                    "remote) via the COPC octree index when the [copc] (laspy) "
                    "or [pdal] extra is installed - windowed reads fetch only "
                    "points inside the window; otherwise reads the pack's "
                    "portable local container."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                install_command=INSTALL_COMMAND,
            ),
            CatalogEntry(
                name="write_pointcloud",
                pillar=self.pillar,
                capability_description=(
                    "Write a point cloud, preserving the point set and per-point "
                    "attributes. Writes standards-compliant COPC when the [pdal] "
                    "extra is installed (or interoperable LAZ via the [copc] "
                    "extra); otherwise writes a portable, lossless local "
                    "container that is not interoperable COPC."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                install_command=INSTALL_COMMAND,
            ),
        ]

    def required_credentials(self) -> "List[CredentialSpec]":
        """Declare the Optional AWS keys ``geo-pointcloud`` can use (Req 16.1).

        Reading and writing COPC is local processing over open-format data, so
        both AWS keys are Optional and never block startup (Requirement 16.5):
        public assets work with no configuration, while private S3 point-cloud
        assets can supply credentials. These specs agree exactly with the
        ``geo-pointcloud`` credential block in ``bundle-manifest.json``.
        """
        return list(self._CREDENTIAL_SPECS)


def main() -> None:
    """Console entry point: serve geo-pointcloud over MCP stdio.

    Runs the startup credential guard, then serves this server's registered
    tools over stdin/stdout via the shared geo-common MCP runtime until the
    client disconnects.
    """
    GeoPointcloudServer().run()

"""The ``geo-raster`` MCP server (Pillar B, expansion).

Wires the zonal-statistics engine (:mod:`geo_raster.zonal`) and the windowed COG
read/band-math engine (:mod:`geo_raster.window_reader`) into the shared
:class:`~geo_common.server.BaseGeoServer` contract and exposes
``zonal_statistics``, ``read_window``, and ``band_math`` as MCP tools
(Requirements 7.2, 7.12, 8.7, 8.8, 12.1, 12.6). It registers the server's
Resource Catalog entries and the optional AWS credential specs, and maps any
failure reaching a wrapped source onto the shared ``Error_Taxonomy``
(Requirement 11.2).

``geo-raster`` reads raster assets directly from S3/HTTP using byte ranges, so
the two AWS keys it declares are *Optional*: public open-data raster buckets
work without them, while private buckets can supply them
(bundle-manifest.json). Optional credentials never block startup (Req 16.5).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from geo_common.errors import GeoError
from geo_common.http import HttpClient
from geo_common.models import (
    CatalogEntry,
    CredentialClassification,
    CredentialSpec,
    OpennessTier,
)
from geo_common.server import BaseGeoServer

from geo_raster.cog import ByteRangeReader
from geo_raster.models import FeatureCollection, ZoneStat
from geo_raster.reader import DEFAULT_S3_REGION
from geo_raster.window_models import RasterArray
from geo_raster.window_reader import (
    band_math as _band_math,
    read_window as _read_window,
)
from geo_raster.zonal import zonal_statistics as _zonal_statistics

__all__ = ["GeoRasterServer", "INSTALL_COMMAND", "main"]

#: The ``uvx`` command that installs this server (Requirements 2.6, 16.1).
INSTALL_COMMAND = "uvx geo-raster"


class GeoRasterServer(BaseGeoServer):
    """Pillar B (expansion) server for raster windowed reads + analytics.

    Exposes ``zonal_statistics`` (per-zone min/max/mean/sum/count with a no-data
    indication for zones with no overlapping cells, Requirements 8.7/8.8),
    ``read_window`` (return only the pixels inside a requested COG window via S3
    byte ranges, Requirements 7.2/12.1), and ``band_math`` (NDVI/NDWI/NBR-style
    expressions over a windowed read). All outbound reads share the inherited
    :class:`HttpClient`, so they inherit retry/backoff and the 30-second
    per-request timeout; an S3 read failure after those retries aborts with
    local storage unchanged (Requirement 12.6).
    """

    pillar = "B"
    server_name = "geo-raster"
    version = "0.2.0"

    #: Both AWS keys are Optional: public raster buckets work without them
    #: (bundle-manifest.json ``geo-raster`` credential block).
    _CREDENTIAL_SPECS = (
        CredentialSpec(
            source="AWS S3 (raster assets)",
            mcp_json_key="AWS_ACCESS_KEY_ID",
            classification=CredentialClassification.OPTIONAL,
        ),
        CredentialSpec(
            source="AWS S3 (raster assets)",
            mcp_json_key="AWS_SECRET_ACCESS_KEY",
            classification=CredentialClassification.OPTIONAL,
        ),
    )

    def __init__(
        self,
        *,
        http: Optional[HttpClient] = None,
        region: str = DEFAULT_S3_REGION,
    ) -> None:
        super().__init__(http=http)
        self.region = region
        self.register_tool("zonal_statistics", self.zonal_statistics)
        self.register_tool("read_window", self.read_window)
        self.register_tool("band_math", self.band_math)

    # ------------------------------------------------------------------
    # Catalog + credential declaration (Requirements 2.1, 11.3, 16.1)
    # ------------------------------------------------------------------

    def catalog_entries(self) -> List[CatalogEntry]:
        """Register ``geo-raster``'s capabilities in the Resource Catalog."""
        return [
            CatalogEntry(
                name="zonal_statistics",
                pillar=self.pillar,
                capability_description=(
                    "Compute per-zone raster statistics (minimum, maximum, mean, "
                    "sum, count) over a set of vector zones, reading only the "
                    "overlapping window directly from S3 byte ranges; zones with "
                    "no overlapping cells get a no-data indication."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                install_command=INSTALL_COMMAND,
            ),
            CatalogEntry(
                name="read_window",
                pillar=self.pillar,
                capability_description=(
                    "Read only the pixels inside a requested window of a "
                    "Cloud-Optimized GeoTIFF directly from S3/HTTP byte ranges, "
                    "without transferring out-of-window data or copying the full "
                    "asset (Sentinel/Landsat/HLS/MODIS/NAIP)."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                install_command=INSTALL_COMMAND,
            ),
            CatalogEntry(
                name="band_math",
                pillar=self.pillar,
                capability_description=(
                    "Evaluate NDVI/NDWI/NBR-style band-math expressions over a "
                    "windowed Cloud-Optimized GeoTIFF read, fetching only the "
                    "referenced bands and overlapping tiles."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                install_command=INSTALL_COMMAND,
            ),
        ]

    def required_credentials(self) -> List[CredentialSpec]:
        """Declare the Optional AWS keys ``geo-raster`` can use (Req 16.1)."""
        return list(self._CREDENTIAL_SPECS)

    # ------------------------------------------------------------------
    # Tool entry point (Requirements 8.7, 8.8, 12.6, 11.2)
    # ------------------------------------------------------------------

    async def zonal_statistics(
        self,
        *,
        raster_href: str,
        zones: FeatureCollection,
        stats: Optional[Sequence[str]] = None,
        band: int = 1,
        reader: Optional[ByteRangeReader] = None,
    ) -> List[ZoneStat]:
        """Return per-zone statistics (Requirements 8.7, 8.8).

        Validation errors for malformed stats/zones (Requirement 8.10
        validation surface) propagate unchanged, as does the ``network`` error
        indicating an S3 read failure after retries (Requirement 12.6); any
        other failure reaching the wrapped source is mapped onto the shared
        ``Error_Taxonomy`` (Requirement 11.2).
        """
        try:
            return await _zonal_statistics(
                raster_href=raster_href,
                zones=zones,
                stats=stats,
                band=band,
                http=self.http,
                region=self.region,
                reader=reader,
            )
        except GeoError:
            raise
        except Exception as exc:  # pragma: no cover - defensive catch-all
            raise self.map_error(exc, source=self.server_name) from exc

    async def read_window(
        self,
        *,
        asset_href: str,
        window,
        bands: Optional[Sequence[int]] = None,
    ) -> RasterArray:
        """Return only the pixels inside ``window`` (Requirements 7.2, 12.1).

        Validation errors for a malformed href/window/band list (Requirement
        7.12) propagate unchanged; any other failure reaching the wrapped source
        is mapped onto the shared ``Error_Taxonomy`` (Requirement 11.2).
        """
        try:
            return await _read_window(
                asset_href=asset_href,
                window=window,
                bands=bands,
                http=self.http,
                region=self.region,
            )
        except GeoError:
            raise
        except Exception as exc:  # pragma: no cover - defensive catch-all
            raise self.map_error(exc, source=self.server_name) from exc

    async def band_math(
        self,
        *,
        asset_href: str,
        expression: str,
        window,
    ) -> RasterArray:
        """Evaluate ``expression`` over a windowed read of ``asset_href``.

        Validation errors (Requirement 7.12) propagate unchanged; any other
        failure is mapped onto the shared ``Error_Taxonomy`` (Requirement 11.2).
        """
        try:
            return await _band_math(
                asset_href=asset_href,
                expression=expression,
                window=window,
                http=self.http,
                region=self.region,
            )
        except GeoError:
            raise
        except Exception as exc:  # pragma: no cover - defensive catch-all
            raise self.map_error(exc, source=self.server_name) from exc


def main() -> None:
    """Console entry point: serve geo-raster over MCP stdio.

    Runs the startup credential guard, then serves this server's registered
    tools over stdin/stdout via the shared geo-common MCP runtime until the
    client disconnects.
    """
    GeoRasterServer().run()


if __name__ == "__main__":  # pragma: no cover - module CLI shim
    main()

"""The ``geo-formats`` MCP server (Pillar B, expansion).

``geo-formats`` converts geospatial data into cloud-optimized formats and
validates the results. It wires the conversion engines
(:mod:`geo_formats.cog`, :mod:`geo_formats.geoparquet`,
:mod:`geo_formats.validate`) into the shared
:class:`~geo_common.server.BaseGeoServer` contract and exposes three MCP tools:

* ``to_cog`` — convert a raster to a Cloud-Optimized GeoTIFF with overviews
  (Requirements 8.4, 12.2).
* ``to_geoparquet`` — convert a vector dataset to GeoParquet (Requirements 8.5,
  12.3).
* ``validate_format`` — check that an output is a well-formed COG/GeoParquet.

All conversion is local and credential-free (GDAL/rasterio + GeoPandas), so the
server starts without any configured credentials. On a write failure the
conversion tools abort, remove any partial output, and surface a taxonomy error
(Requirement 12.7); any other unexpected failure is mapped onto the shared
``Error_Taxonomy`` (Requirement 11.2).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from geo_common.errors import GeoError
from geo_common.http import HttpClient
from geo_common.models import (
    CatalogEntry,
    CredentialClassification,
    CredentialSpec,
    OpennessTier,
)
from geo_common.server import BaseGeoServer

from geo_formats.cog import to_cog as _to_cog
from geo_formats.geoparquet import to_geoparquet as _to_geoparquet
from geo_formats.models import FeatureCollection, FormatResult, FormatValidity
from geo_formats.validate import validate_format as _validate_format

__all__ = ["GeoFormatsServer", "INSTALL_COMMAND", "main"]

#: The ``uvx`` command that installs this server (Requirements 2.6, 16.1).
INSTALL_COMMAND = "uvx geo-formats"


class GeoFormatsServer(BaseGeoServer):
    """Pillar B (expansion) server for cloud-optimized format conversion.

    Exposes ``to_cog`` (Cloud-Optimized GeoTIFF with overviews), ``to_geoparquet``
    (GeoParquet), and ``validate_format``. Conversion is local; the two AWS keys
    it declares are *Optional* (S3 input/output only), so an absent key never
    blocks startup and the server always starts (Requirement 16.5).
    """

    pillar = "B"
    server_name = "geo-formats"
    version = "0.2.0"

    #: Both AWS keys are Optional: local file I/O works without them, while
    #: reading or writing cloud-optimized outputs on S3 can supply them
    #: (bundle-manifest.json ``geo-formats`` credential block). Optional
    #: credentials never block startup (Requirement 16.5).
    _CREDENTIAL_SPECS = (
        CredentialSpec(
            source="AWS S3 (output destination)",
            mcp_json_key="AWS_ACCESS_KEY_ID",
            classification=CredentialClassification.OPTIONAL,
        ),
        CredentialSpec(
            source="AWS S3 (output destination)",
            mcp_json_key="AWS_SECRET_ACCESS_KEY",
            classification=CredentialClassification.OPTIONAL,
        ),
    )

    #: Native capabilities ``geo-formats`` implements (GDAL/rasterio + GeoPandas).
    _NATIVE_CAPABILITIES = (
        (
            "to_cog",
            "Convert a raster to a Cloud-Optimized GeoTIFF (internally tiled, "
            "with overviews) preserving pixels exactly, including nodata "
            "(GDAL/rasterio).",
        ),
        (
            "to_geoparquet",
            "Convert a vector dataset to GeoParquet, preserving feature count, "
            "geometries, and attributes exactly (GeoPandas).",
        ),
        (
            "validate_format",
            "Validate that an output is a well-formed Cloud-Optimized GeoTIFF "
            "or GeoParquet.",
        ),
    )

    def __init__(self, http: Optional[HttpClient] = None) -> None:
        super().__init__(http=http)
        self.register_tool("to_cog", self.to_cog)
        self.register_tool("to_geoparquet", self.to_geoparquet)
        self.register_tool("validate_format", self.validate_format)

    # ------------------------------------------------------------------
    # Tool entry points (Requirements 8.4, 8.5, 12.2, 12.3, 12.7)
    # ------------------------------------------------------------------

    async def to_cog(
        self,
        *,
        src_href: str,
        dst_href: str,
        overviews: bool = True,
    ) -> FormatResult:
        """Convert ``src_href`` to a Cloud-Optimized GeoTIFF at ``dst_href``.

        Validation errors for malformed/unreadable input propagate unchanged;
        on a write failure the conversion aborts, removes any partial output,
        and raises a taxonomy error (Requirement 12.7). Any other unexpected
        failure is mapped onto the shared ``Error_Taxonomy`` (Requirement 11.2).
        """
        try:
            return _to_cog(
                src_href, dst_href, overviews=overviews, source=self.server_name
            )
        except GeoError:
            raise
        except Exception as exc:  # pragma: no cover - defensive catch-all
            raise self.map_error(exc, source=self.server_name) from exc

    async def to_geoparquet(
        self,
        *,
        src: Union[FeatureCollection, Dict[str, Any], str],
        dst_href: str,
    ) -> FormatResult:
        """Convert vector ``src`` to GeoParquet at ``dst_href`` (Req 8.5, 12.3).

        ``src`` accepts any of: an inline GeoJSON ``FeatureCollection`` (features
        with geometry + properties), an equivalent GeoJSON mapping, or a **path
        to an existing ``.geojson``/``.json`` or ``.parquet`` file** — so a large
        vector produced by an earlier step can be referenced by path instead of
        pasted inline.

        Validation errors for malformed input propagate unchanged; on a write
        failure the conversion aborts, removes any partial output, and raises a
        taxonomy error (Requirement 12.7).
        """
        try:
            return _to_geoparquet(src, dst_href, source=self.server_name)
        except GeoError:
            raise
        except Exception as exc:  # pragma: no cover - defensive catch-all
            raise self.map_error(exc, source=self.server_name) from exc

    async def validate_format(self, *, href: str, fmt: str) -> FormatValidity:
        """Validate that ``href`` is a well-formed instance of ``fmt``."""
        try:
            return _validate_format(href, fmt, source=self.server_name)
        except GeoError:
            raise
        except Exception as exc:  # pragma: no cover - defensive catch-all
            raise self.map_error(exc, source=self.server_name) from exc

    # ------------------------------------------------------------------
    # Catalog + credential declaration (Requirements 2.1, 16.1)
    # ------------------------------------------------------------------

    def catalog_entries(self) -> List[CatalogEntry]:
        """Register every ``geo-formats`` capability in the Resource Catalog."""
        return [
            CatalogEntry(
                name=name,
                pillar=self.pillar,
                capability_description=description,
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                install_command=INSTALL_COMMAND,
            )
            for name, description in self._NATIVE_CAPABILITIES
        ]

    def required_credentials(self) -> List[CredentialSpec]:
        """Declare the Optional AWS keys ``geo-formats`` can use (Req 16.1).

        Both keys are Optional, so they never block startup (Req 16.5): local
        file conversion works with no configuration, while S3 input/output can
        supply AWS credentials. These specs agree exactly with the
        ``geo-formats`` credential block in ``bundle-manifest.json``.
        """
        return list(self._CREDENTIAL_SPECS)


def main() -> None:
    """Console entry point: serve geo-formats over MCP stdio.

    Runs the startup credential guard, then serves this server's registered
    tools over stdin/stdout via the shared geo-common MCP runtime until the
    client disconnects.
    """
    GeoFormatsServer().run()


if __name__ == "__main__":  # pragma: no cover - module CLI shim
    main()

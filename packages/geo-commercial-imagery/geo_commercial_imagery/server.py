"""The ``geo-commercial-imagery`` MCP server (Credentialed pillar, expansion).

:class:`GeoCommercialImageryServer` fronts commercial satellite imagery behind
two MCP tools, routing by ``provider``:

* ``MAXAR`` -> **Sentinel Hub TPDI** (:mod:`geo_commercial_imagery.sentinelhub`),
  authenticated with ``SENTINELHUB_CLIENT_ID`` / ``SENTINELHUB_CLIENT_SECRET``.
* ``PLANET`` -> **Planet's own Data + Orders APIs**
  (:mod:`geo_commercial_imagery.planet`), authenticated with ``PL_API_KEY``.

Planet moved off Sentinel Hub TPDI (which is being sunset for Planet data), and
Airbus is no longer offered through TPDI, so those paths route to Planet's
first-party APIs and to Maxar-only TPDI respectively. All credentials are
License-Needed (they never block startup, Req 16.5); each tool enforces the
relevant credential per-invocation, naming the missing ``mcp.json`` key without
returning partial data (Req 7.8).
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

from geo_common.http import HttpClient
from geo_common.models import (
    CatalogEntry,
    CredentialClassification,
    CredentialSpec,
    OpennessTier,
)
from geo_common.server import BaseGeoServer
from geo_common.errors import ValidationError

from geo_commercial_imagery import planet as planet_api
from geo_commercial_imagery import sentinelhub as sh
from geo_commercial_imagery.models import OrderConfirmation, StacItem
from geo_commercial_imagery.planet import PLANET_API_KEY_KEY
from geo_commercial_imagery.sentinelhub import (
    SENTINELHUB_CLIENT_ID_KEY,
    SENTINELHUB_CLIENT_SECRET_KEY,
)

__all__ = [
    "GeoCommercialImageryServer",
    "INSTALL_COMMAND",
    "SENTINELHUB_LICENSE_REFERENCE",
    "PLANET_LICENSE_REFERENCE",
    "SUPPORTED_PROVIDERS",
    "main",
]

#: The ``uvx`` command that installs this server (Requirements 2.6, 16.1).
INSTALL_COMMAND = "uvx geo-commercial-imagery"

#: The providers this server routes, and the backend each uses.
SUPPORTED_PROVIDERS = ("MAXAR", "PLANET")

#: Licensing reference for the Maxar-via-Sentinel-Hub credentials (Req 3.7).
SENTINELHUB_LICENSE_REFERENCE = (
    "Requires a Sentinel Hub commercial subscription with Maxar access. See "
    "https://www.sentinel-hub.com/."
)

#: Licensing reference for the Planet API credential (Req 3.7).
PLANET_LICENSE_REFERENCE = (
    "Requires a Planet account/subscription and API key. See "
    "https://docs.planet.com/."
)


class GeoCommercialImageryServer(BaseGeoServer):
    """Credentialed (expansion) server for commercial imagery (Maxar + Planet).

    Routes ``search_commercial`` / ``order_scene`` to Sentinel Hub TPDI (Maxar)
    or Planet's native APIs (Planet), resolving each provider's credential
    per-invocation from the single ``mcp.json`` surface (the environment) unless
    supplied explicitly (tests). Credentials are never logged.
    """

    pillar = "expansion"
    server_name = "geo-commercial-imagery"
    version = "0.2.0"

    def __init__(
        self,
        http: Optional[HttpClient] = None,
        *,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        planet_api_key: Optional[str] = None,
    ) -> None:
        super().__init__(http=http)
        self._client_id = client_id
        self._client_secret = client_secret
        self._planet_api_key = planet_api_key
        self.register_tool("search_commercial", self.search_commercial)
        self.register_tool("order_scene", self.order_scene)

    # ------------------------------------------------------------------
    # Catalog + credential declaration (Requirements 2.1, 3.7, 11.3, 16.1)
    # ------------------------------------------------------------------

    def catalog_entries(self) -> List[CatalogEntry]:
        """Register the two commercial-imagery capabilities (Req 2.1, 11.3)."""
        provider = self.server_name
        return [
            CatalogEntry(
                name="search_commercial",
                pillar=self.pillar,
                capability_description=(
                    "Search commercial imagery scenes by provider, bounding box, "
                    "and time range: Maxar WorldView via the Sentinel Hub TPDI "
                    "API, or Planet (PlanetScope/SkySat) via the Planet Data API."
                ),
                openness_tier=OpennessTier.PROPRIETARY,
                provider_server=provider,
                install_command=INSTALL_COMMAND,
            ),
            CatalogEntry(
                name="order_scene",
                pillar=self.pillar,
                capability_description=(
                    "Order commercial imagery scenes by product id: Maxar via the "
                    "Sentinel Hub TPDI Orders API (create-only by default, confirm "
                    "to spend quota), or Planet via the Planet Orders API (placed "
                    "on confirm). Returns the order confirmation."
                ),
                openness_tier=OpennessTier.PROPRIETARY,
                provider_server=provider,
                install_command=INSTALL_COMMAND,
            ),
        ]

    def required_credentials(self) -> List[CredentialSpec]:
        """Declare the Maxar (Sentinel Hub) and Planet ``mcp.json`` keys.

        All License-Needed, so absence never blocks startup (Req 16.5); each
        tool enforces the relevant credential at invocation time based on the
        ``provider`` (Req 7.8).
        """
        return [
            CredentialSpec(
                source="Sentinel Hub (Maxar)",
                mcp_json_key=SENTINELHUB_CLIENT_ID_KEY,
                classification=CredentialClassification.LICENSE_NEEDED,
                license_reference=SENTINELHUB_LICENSE_REFERENCE,
            ),
            CredentialSpec(
                source="Sentinel Hub (Maxar)",
                mcp_json_key=SENTINELHUB_CLIENT_SECRET_KEY,
                classification=CredentialClassification.LICENSE_NEEDED,
                license_reference=SENTINELHUB_LICENSE_REFERENCE,
            ),
            CredentialSpec(
                source="Planet",
                mcp_json_key=PLANET_API_KEY_KEY,
                classification=CredentialClassification.LICENSE_NEEDED,
                license_reference=PLANET_LICENSE_REFERENCE,
            ),
        ]

    # ------------------------------------------------------------------
    # Credential resolution (single mcp.json surface)
    # ------------------------------------------------------------------

    def _resolve_client_id(self) -> Optional[str]:
        if self._client_id:
            return self._client_id
        return os.environ.get(SENTINELHUB_CLIENT_ID_KEY) or None

    def _resolve_client_secret(self) -> Optional[str]:
        if self._client_secret:
            return self._client_secret
        return os.environ.get(SENTINELHUB_CLIENT_SECRET_KEY) or None

    def _resolve_planet_api_key(self) -> Optional[str]:
        if self._planet_api_key:
            return self._planet_api_key
        return os.environ.get(PLANET_API_KEY_KEY) or None

    def _validate_provider(self, provider: str) -> str:
        """Validate and normalize ``provider`` to ``MAXAR`` or ``PLANET``."""
        if not isinstance(provider, str) or not provider.strip():
            raise ValidationError(
                "provider must be one of: %s" % ", ".join(SUPPORTED_PROVIDERS),
                source=self.server_name,
                detail={"parameter": "provider", "supported": list(SUPPORTED_PROVIDERS)},
            )
        value = provider.strip().upper()
        if value not in SUPPORTED_PROVIDERS:
            raise ValidationError(
                "provider %r is not supported; choose one of: %s "
                "(Airbus is no longer available via this server)"
                % (provider, ", ".join(SUPPORTED_PROVIDERS)),
                source=self.server_name,
                detail={"parameter": "provider", "supported": list(SUPPORTED_PROVIDERS)},
            )
        return value

    # ------------------------------------------------------------------
    # Tool entry points (Requirements 2.1, 7.8)
    # ------------------------------------------------------------------

    async def search_commercial(
        self,
        *,
        provider: str,
        bbox: Sequence[float],
        datetime_range: Sequence[str],
        item_type: Optional[str] = None,
        product_bands: Optional[str] = None,
        max_cloud_coverage: Optional[float] = None,
        limit: int = 1000,
    ) -> List[StacItem]:
        """Search commercial scenes by provider + bbox + time range (Req 2.1, 7.8).

        ``provider`` is ``MAXAR`` (Sentinel Hub TPDI; ``product_bands`` selects
        the band product) or ``PLANET`` (Planet Data API; ``item_type`` selects
        the item type). ``max_cloud_coverage`` (0-100) optionally caps cloud
        cover for both.
        """
        prov = self._validate_provider(provider)
        if prov == "MAXAR":
            return await sh.search_commercial(
                bbox=bbox,
                datetime_range=datetime_range,
                product_bands=product_bands,
                max_cloud_coverage=max_cloud_coverage,
                http=self.http,
                client_id=self._resolve_client_id(),
                client_secret=self._resolve_client_secret(),
                limit=limit,
            )
        return await planet_api.search_commercial(
            bbox=bbox,
            datetime_range=datetime_range,
            item_type=item_type,
            max_cloud_coverage=max_cloud_coverage,
            http=self.http,
            api_key=self._resolve_planet_api_key(),
            limit=limit,
        )

    async def order_scene(
        self,
        *,
        provider: str,
        scene_ids: Sequence[str],
        bbox: Optional[Sequence[float]] = None,
        item_type: Optional[str] = None,
        product_bands: Optional[str] = None,
        product_bundle: Optional[str] = None,
        name: Optional[str] = None,
        confirm: bool = False,
    ) -> OrderConfirmation:
        """Order commercial scenes by product id (Req 2.1, 7.8).

        ``MAXAR`` routes to Sentinel Hub TPDI (requires ``bbox`` for the order
        area; create-only unless ``confirm=True``). ``PLANET`` routes to the
        Planet Orders API (``item_type``/``product_bundle`` select the product;
        the order is placed immediately, so ``confirm=True`` is required).
        """
        prov = self._validate_provider(provider)
        if prov == "MAXAR":
            if bbox is None:
                raise ValidationError(
                    "Maxar orders require a bbox for the order area",
                    source=self.server_name,
                    detail={"parameter": "bbox", "provider": prov},
                )
            return await sh.order_scene(
                scene_ids=scene_ids,
                bbox=bbox,
                product_bands=product_bands,
                name=name,
                confirm=confirm,
                http=self.http,
                client_id=self._resolve_client_id(),
                client_secret=self._resolve_client_secret(),
            )
        return await planet_api.order_scene(
            scene_ids=scene_ids,
            item_type=item_type,
            product_bundle=product_bundle,
            name=name,
            confirm=confirm,
            http=self.http,
            api_key=self._resolve_planet_api_key(),
        )


def main() -> None:
    """Console entry point: serve geo-commercial-imagery over MCP stdio."""
    GeoCommercialImageryServer().run()


if __name__ == "__main__":  # pragma: no cover - module CLI shim
    main()

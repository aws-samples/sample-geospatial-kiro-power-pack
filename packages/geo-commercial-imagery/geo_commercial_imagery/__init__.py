"""geo-commercial-imagery: Credentialed (expansion) commercial-imagery MCP server.

Exposes commercial satellite imagery behind the Geospatial Power Pack
conventions, routing two MCP tools by ``provider``:

* **Maxar** (``provider="MAXAR"``) via the **Sentinel Hub TPDI** API
  (:mod:`geo_commercial_imagery.sentinelhub`), authenticated with
  ``SENTINELHUB_CLIENT_ID`` / ``SENTINELHUB_CLIENT_SECRET``.
* **Planet** (``provider="PLANET"``) via **Planet's own Data + Orders APIs**
  (:mod:`geo_commercial_imagery.planet`), authenticated with ``PL_API_KEY``.

Sentinel Hub TPDI is being sunset for Planet data (and no longer offers Airbus),
so Planet routes to its first-party APIs and Maxar remains on TPDI. All
credentials are License-Needed and enforced per-invocation (Req 7.8).

* :func:`search_commercial` / :func:`order_scene` are exposed as methods of
  :class:`GeoCommercialImageryServer` (the provider router); the per-provider
  request builders live in the ``sentinelhub`` and ``planet`` submodules.
"""

from __future__ import annotations

from geo_commercial_imagery.models import BBox, OrderConfirmation, StacItem
from geo_commercial_imagery.sentinelhub import (
    CREDENTIAL_KEYS,
    DEFAULT_MAXAR_PRODUCT_BANDS,
    DEFAULT_ORDERS_URL,
    DEFAULT_SEARCH_URL,
    DEFAULT_TOKEN_URL,
    SENTINELHUB_CLIENT_ID_KEY,
    SENTINELHUB_CLIENT_SECRET_KEY,
)
from geo_commercial_imagery.planet import (
    DEFAULT_DATA_SEARCH_URL as PLANET_DATA_SEARCH_URL,
    DEFAULT_ORDERS_URL as PLANET_ORDERS_URL,
    DEFAULT_PLANET_ITEM_TYPE,
    DEFAULT_PLANET_PRODUCT_BUNDLE,
    PLANET_API_KEY_KEY,
)
from geo_commercial_imagery.server import (
    INSTALL_COMMAND,
    PLANET_LICENSE_REFERENCE,
    SENTINELHUB_LICENSE_REFERENCE,
    SUPPORTED_PROVIDERS,
    GeoCommercialImageryServer,
    main,
)

__all__ = [
    "GeoCommercialImageryServer",
    "StacItem",
    "OrderConfirmation",
    "BBox",
    "SUPPORTED_PROVIDERS",
    # Sentinel Hub (Maxar)
    "SENTINELHUB_CLIENT_ID_KEY",
    "SENTINELHUB_CLIENT_SECRET_KEY",
    "CREDENTIAL_KEYS",
    "DEFAULT_MAXAR_PRODUCT_BANDS",
    "DEFAULT_TOKEN_URL",
    "DEFAULT_SEARCH_URL",
    "DEFAULT_ORDERS_URL",
    "SENTINELHUB_LICENSE_REFERENCE",
    # Planet
    "PLANET_API_KEY_KEY",
    "PLANET_DATA_SEARCH_URL",
    "PLANET_ORDERS_URL",
    "DEFAULT_PLANET_ITEM_TYPE",
    "DEFAULT_PLANET_PRODUCT_BUNDLE",
    "PLANET_LICENSE_REFERENCE",
    # Server
    "INSTALL_COMMAND",
    "main",
]

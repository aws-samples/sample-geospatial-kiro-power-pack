"""The ``geo-weather-climate`` MCP server (Pillar A, expansion).

Wires the observation connector (:mod:`geo_weather_climate.observations`) into
the shared :class:`~geo_common.server.BaseGeoServer` contract and exposes
``observations`` as an MCP tool (Requirements 7.6, 7.12). It also registers the
server's Resource Catalog entry and credential specs and ensures source
failures surface as Error_Taxonomy availability errors identifying the
unavailable source (Requirements 2.1, 7.11, 11.2, 11.3).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from geo_common.http import HttpClient
from geo_common.models import (
    CatalogEntry,
    CredentialClassification,
    CredentialSpec,
    OpennessTier,
)
from geo_common.server import BaseGeoServer

from geo_weather_climate.observations import (
    DEFAULT_SOURCE,
    NoaaCdoSource,
    WeatherSource,
    default_sources,
    observations as _observations,
)

__all__ = ["GeoWeatherClimateServer", "INSTALL_COMMAND", "main"]

#: The ``uvx`` command that installs this server (Requirements 2.6, 16.1). The
#: Resource Catalog surfaces it on entries whose provider is not yet installed.
INSTALL_COMMAND = "uvx geo-weather-climate"


class GeoWeatherClimateServer(BaseGeoServer):
    """Pillar A (expansion) server exposing weather/climate observations.

    Holds the configurable source set (Open-Meteo weather + Open-Meteo
    air-quality by default) and
    registers ``observations`` as an MCP tool. All outbound calls share the
    inherited :class:`HttpClient` so they get identical retry/backoff and the
    30-second per-request timeout (Requirement 7.6).
    """

    pillar = "A"
    server_name = "geo-weather-climate"
    version = "0.3.0"

    #: The wrapped NOAA CDO source can use this ``mcp.json`` key for richer
    #: access, but it is *Optional*: Open-Meteo (weather + air-quality) / NWS
    #: serve open data
    #: without a credential, so an absent key never blocks startup
    #: (Requirement 16.5). Mirrors ``bundle-manifest.json``'s
    #: ``geo-weather-climate`` block. (ERA5 is covered via the Open-Meteo
    #: archive rather than a direct Copernicus CDS client, so no CDS key.)
    _CREDENTIAL_SPECS: Tuple[CredentialSpec, ...] = (
        CredentialSpec(
            source="NOAA Climate Data Online",
            mcp_json_key="NOAA_CDO_TOKEN",
            classification=CredentialClassification.OPTIONAL,
        ),
    )

    def __init__(
        self,
        *,
        sources: Optional[Sequence[WeatherSource]] = None,
        default_source: str = DEFAULT_SOURCE,
        http: Optional[HttpClient] = None,
    ) -> None:
        super().__init__(http=http)
        self.sources: List[WeatherSource] = (
            list(sources) if sources is not None else self._default_source_set()
        )
        self.default_source = default_source
        self.register_tool("observations", self.observations)

    @staticmethod
    def _default_source_set() -> List[WeatherSource]:
        """The open default sources plus the credentialed NOAA CDO source.

        CDO is always *selectable* (``source="cdo"``) so a caller gets a clean
        ``AuthenticationError`` naming ``NOAA_CDO_TOKEN`` when it is invoked
        without a token; when the token is configured in ``mcp.json`` the source
        queries NOAA CDO. It is appended (not made the default) because it needs
        a credential, while the open sources do not.
        """
        token = os.environ.get("NOAA_CDO_TOKEN") or None
        return [*default_sources(), NoaaCdoSource(token=token)]

    def catalog_entries(self) -> List[CatalogEntry]:
        """Register ``geo-weather-climate``'s capability in the Resource Catalog.

        Declares the ``observations`` capability naming ``geo-weather-climate``
        as the provider (Requirements 2.1, 11.3). The default sources
        (Open-Meteo weather + air-quality) are open data, so the entry is tier
        :attr:`OpennessTier.OPEN`; the ``uvx`` install command lets the catalog
        surface it when the provider is not yet installed (Requirement 2.6).
        """
        return [
            CatalogEntry(
                name="observations",
                pillar=self.pillar,
                capability_description=(
                    "Retrieve weather, climate, and environmental observations "
                    "for a point location and time range: Open-Meteo weather "
                    "(default, source='open-meteo'), Open-Meteo air-quality "
                    "(PM2.5/PM10/CO/NO2/SO2/O3, source='air-quality'), and "
                    "NOAA/NWS (US coverage, source='nws'); NOAA CDO when "
                    "configured. Rejects a time range whose start is later than "
                    "its end."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                install_command=INSTALL_COMMAND,
            )
        ]

    def required_credentials(self) -> List[CredentialSpec]:
        """Declare the ``mcp.json`` keys ``geo-weather-climate``'s sources can use.

        Both are :attr:`~geo_common.models.CredentialClassification.OPTIONAL`:
        the default sources serve open data without a credential, so an absent
        key never blocks startup (Requirement 16.5).
        """
        return list(self._CREDENTIAL_SPECS)

    async def observations(
        self,
        *,
        location: Any,
        time_range: Sequence[str],
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return observations for ``location`` over ``time_range`` (Req 7.6).

        Validates the location and time range before querying any source: a
        malformed coordinate, an unparseable timestamp, or a ``start`` later
        than its ``end`` raises an ``Error_Taxonomy`` ``ValidationError``
        (Requirement 7.12). On a valid request the named ``source`` (default
        Open-Meteo) is queried through the shared HTTP client and the matching
        observations are returned as a list of mappings.
        """
        chosen_source = self.default_source if source is None else source
        return await _observations(
            location=location,
            time_range=time_range,
            http=self.http,
            sources=self.sources,
            source=chosen_source,
        )


def main() -> None:
    """Console entry point: serve geo-weather-climate over MCP stdio.

    Runs the startup credential guard, then serves this server's registered
    tools over stdin/stdout via the shared geo-common MCP runtime until the
    client disconnects.
    """
    GeoWeatherClimateServer().run()


if __name__ == "__main__":  # pragma: no cover - module CLI shim
    main()

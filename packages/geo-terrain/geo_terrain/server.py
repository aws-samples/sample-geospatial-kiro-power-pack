"""The ``geo-terrain`` MCP server (Pillar A, expansion).

Wires the elevation connector (:mod:`geo_terrain.elevation`) into the shared
:class:`~geo_common.server.BaseGeoServer` contract and exposes ``elevation`` as
an MCP tool (Requirements 7.5, 7.12). It registers the server's Resource
Catalog entry and (open, credential-free) credential specs, and source failures
surface as taxonomy-classified errors through the shared
:class:`~geo_common.http.HttpClient`.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Union

from geo_common.http import HttpClient
from geo_common.models import (
    CatalogEntry,
    CredentialClassification,
    CredentialSpec,
    OpennessTier,
)
from geo_common.server import BaseGeoServer

from geo_common.raster_models import FeatureCollection, ZoneStat

from geo_terrain.dem_zonal import dem_zonal as _dem_zonal
from geo_terrain.derivatives import (
    compute_aspect,
    compute_hillshade,
    compute_slope,
    meters_per_degree,
)
from geo_terrain.elevation import (
    DEFAULT_SOURCE,
    TerrainSource,
    default_sources,
    elevation as _elevation,
)
from geo_terrain.models import RasterArray

__all__ = ["GeoTerrainServer", "INSTALL_COMMAND", "main"]

#: The ``uvx`` command that installs this server (Requirements 2.6, 16.1). The
#: Resource Catalog surfaces it on entries whose provider is not yet installed.
INSTALL_COMMAND = "uvx geo-terrain"


class GeoTerrainServer(BaseGeoServer):
    """Pillar A (expansion) server exposing elevation queries (Req 7.5, 7.12).

    Holds the configurable terrain source set (SRTM the default, plus USGS
    3DEP) and the default source, and registers ``elevation`` as an MCP tool.
    Both sources work against the public OpenTopoData endpoint out of the box.
    All outbound calls share the inherited :class:`HttpClient` so they get
    identical retry/backoff and the 30-second per-request timeout.
    """

    pillar = "A"
    server_name = "geo-terrain"
    version = "0.2.0"

    #: ``geo-terrain``'s default sources (SRTM, 3DEP) are open and need no
    #: credential. OpenTopography-hosted datasets *can* use an optional API key
    #: for higher limits / restricted layers; this mirrors
    #: ``bundle-manifest.json``'s ``geo-terrain`` credential block. The key is
    #: :attr:`CredentialClassification.OPTIONAL`, so an absent key never blocks
    #: startup (Requirement 16.5).
    _CREDENTIAL_SPECS = (
        CredentialSpec(
            source="OpenTopography",
            mcp_json_key="OPENTOPOGRAPHY_API_KEY",
            classification=CredentialClassification.OPTIONAL,
        ),
    )

    def __init__(
        self,
        *,
        sources: Optional[Mapping[str, TerrainSource]] = None,
        default_source: str = DEFAULT_SOURCE,
        http: Optional[HttpClient] = None,
    ) -> None:
        super().__init__(http=http)
        self.sources: dict = dict(sources) if sources is not None else default_sources()
        self.default_source = default_source
        self.register_tool("elevation", self.elevation)
        self.register_tool("slope", self.slope)
        self.register_tool("aspect", self.aspect)
        self.register_tool("hillshade", self.hillshade)
        self.register_tool("dem_zonal", self.dem_zonal)

    def catalog_entries(self) -> List[CatalogEntry]:
        """Register ``geo-terrain``'s capability in the Resource Catalog.

        Declares the ``elevation`` capability (Requirements 2.1, 11.3). The
        default sources - SRTM and USGS 3DEP - are open data, so the entry is
        tier :attr:`OpennessTier.OPEN`, names ``geo-terrain`` as provider
        (Requirement 11.3), and carries the ``uvx`` install command so the
        catalog can surface it when the provider is not yet installed
        (Requirement 2.6).
        """
        return [
            CatalogEntry(
                name="elevation",
                pillar=self.pillar,
                capability_description=(
                    "Return elevation for a location (a scalar in metres) or an "
                    "extent (a sampled elevation grid) from a configured terrain "
                    "source: SRTM (default) or USGS 3DEP."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                install_command=INSTALL_COMMAND,
            ),
            CatalogEntry(
                name="slope",
                pillar=self.pillar,
                capability_description=(
                    "Compute per-cell slope (in degrees) over an extent from a "
                    "configured terrain source's elevation grid (Horn-style "
                    "gradient)."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                install_command=INSTALL_COMMAND,
            ),
            CatalogEntry(
                name="aspect",
                pillar=self.pillar,
                capability_description=(
                    "Compute per-cell aspect (compass degrees 0-360, the "
                    "downslope direction; -1 for flat) over an extent from a "
                    "configured terrain source's elevation grid."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                install_command=INSTALL_COMMAND,
            ),
            CatalogEntry(
                name="hillshade",
                pillar=self.pillar,
                capability_description=(
                    "Compute shaded-relief (hillshade, 0-255) over an extent "
                    "from a configured terrain source's elevation grid, for a "
                    "given sun azimuth and altitude."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                install_command=INSTALL_COMMAND,
            ),
            CatalogEntry(
                name="dem_zonal",
                pillar=self.pillar,
                capability_description=(
                    "Per-zone statistics (min/max/mean/sum/count/std) over a "
                    "user-supplied DEM COG for elevation, slope, or aspect: "
                    "reads only the tiles overlapping the vector zones by byte "
                    "range, derives the surface, and reduces per zone. Zones "
                    "must be in the DEM's CRS."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                install_command=INSTALL_COMMAND,
            ),
        ]

    def required_credentials(self) -> List[CredentialSpec]:
        """Declare the ``mcp.json`` keys ``geo-terrain`` can use (Req 16.1).

        The default terrain sources are open and require no credential; the
        ``OPENTOPOGRAPHY_API_KEY`` key is
        :attr:`CredentialClassification.OPTIONAL`, so an absent key never blocks
        startup (Requirement 16.5) and the declaration matches
        ``bundle-manifest.json``.
        """
        return list(self._CREDENTIAL_SPECS)

    async def elevation(
        self,
        *,
        location: Any,
        source: Optional[str] = None,
    ) -> Union[float, RasterArray]:
        """Return elevation for ``location`` from the selected ``source`` (Req 7.5).

        ``location`` is a single point (a ``Coordinate``, ``{"lon", "lat"}``
        mapping, or ``(lon, lat)`` pair) → a scalar elevation in metres, or an
        extent (a ``GeoWindow``, ``{"bbox", "width", "height"}`` mapping, or a
        4-element bbox) → a :class:`RasterArray` of sampled elevations. The
        location and the source are validated before any source is queried; a
        malformed location or an unknown source raises an ``Error_Taxonomy``
        ``ValidationError`` (Requirement 7.12).
        """
        return await _elevation(
            location=location,
            http=self.http,
            sources=self.sources,
            source=self.default_source if source is None else source,
        )

    async def _elevation_grid(self, location: Any, source: Optional[str]) -> RasterArray:
        """Fetch an elevation :class:`RasterArray` for an extent (Req 7.5).

        Shared by ``slope`` and ``hillshade``: queries the selected terrain
        source for ``location`` and requires the result to be a grid. A point
        location (which yields a scalar) raises an ``Error_Taxonomy``
        ``ValidationError`` because slope/hillshade need neighbouring cells.
        """
        from geo_common.errors import ValidationError

        result = await _elevation(
            location=location,
            http=self.http,
            sources=self.sources,
            source=self.default_source if source is None else source,
        )
        if not isinstance(result, RasterArray):
            raise ValidationError(
                "slope/hillshade require an extent (a bbox), not a single point; "
                "pass {'bbox': [min_lon, min_lat, max_lon, max_lat]} (optionally "
                "with 'width'/'height')",
                source=self.server_name,
                detail={"parameter": "location"},
            )
        if result.width < 2 or result.height < 2:
            raise ValidationError(
                "slope/hillshade need a grid of at least 2x2 samples; increase "
                "the extent's width/height",
                source=self.server_name,
                detail={"parameter": "location", "width": result.width, "height": result.height},
            )
        return result

    def _cell_sizes_m(self, grid: RasterArray) -> "tuple[float, float]":
        """Approximate ground cell size (metres) in x and y for ``grid``'s extent."""
        min_lon, min_lat, max_lon, max_lat = grid.bbox
        dlon = (max_lon - min_lon) / (grid.width - 1)
        dlat = (max_lat - min_lat) / (grid.height - 1)
        m_per_deg_lon, m_per_deg_lat = meters_per_degree((min_lat + max_lat) / 2.0)
        return (abs(dlon) * m_per_deg_lon, abs(dlat) * m_per_deg_lat)

    async def slope(self, *, location: Any, source: Optional[str] = None) -> RasterArray:
        """Per-cell slope (degrees) over an extent from the DEM grid (Req 7.5).

        Fetches the elevation grid for ``location`` (which must be an extent),
        then computes slope as the gradient magnitude of the surface in degrees
        (0 = flat). Returns a :class:`RasterArray` with ``units='degrees'``.
        """
        grid = await self._elevation_grid(location, source)
        cx, cy = self._cell_sizes_m(grid)
        values = compute_slope(grid.values, cellsize_x_m=cx, cellsize_y_m=cy)
        return RasterArray(
            values=values,
            width=grid.width,
            height=grid.height,
            bbox=grid.bbox,
            source=grid.source,
            units="degrees",
        )

    async def aspect(self, *, location: Any, source: Optional[str] = None) -> RasterArray:
        """Per-cell aspect (compass degrees) over an extent from the DEM grid (Req 7.5).

        Fetches the elevation grid for ``location`` (which must be an extent),
        then computes aspect: the compass bearing of the downslope direction
        (0 = north, 90 = east, 180 = south, 270 = west), with ``-1`` for flat
        cells. Returns a :class:`RasterArray` with ``units='degrees'``.
        """
        grid = await self._elevation_grid(location, source)
        cx, cy = self._cell_sizes_m(grid)
        values = compute_aspect(grid.values, cellsize_x_m=cx, cellsize_y_m=cy)
        return RasterArray(
            values=values,
            width=grid.width,
            height=grid.height,
            bbox=grid.bbox,
            source=grid.source,
            units="degrees",
        )

    async def hillshade(
        self,
        *,
        location: Any,
        azimuth: float = 315.0,
        altitude: float = 45.0,
        source: Optional[str] = None,
    ) -> RasterArray:
        """Shaded-relief (hillshade, 0-255) over an extent from the DEM grid (Req 7.5).

        Fetches the elevation grid for ``location`` (an extent) and computes
        hillshade for a sun at ``azimuth`` (degrees clockwise from north,
        default 315=NW) and ``altitude`` (degrees above the horizon, default
        45). Returns a :class:`RasterArray` with ``units='hillshade'``.
        """
        from geo_common.errors import ValidationError

        if not (0.0 <= azimuth <= 360.0):
            raise ValidationError(
                "azimuth must be within [0, 360] degrees",
                source=self.server_name,
                detail={"parameter": "azimuth", "value": azimuth},
            )
        if not (0.0 <= altitude <= 90.0):
            raise ValidationError(
                "altitude must be within [0, 90] degrees",
                source=self.server_name,
                detail={"parameter": "altitude", "value": altitude},
            )
        grid = await self._elevation_grid(location, source)
        cx, cy = self._cell_sizes_m(grid)
        values = compute_hillshade(
            grid.values,
            cellsize_x_m=cx,
            cellsize_y_m=cy,
            azimuth_deg=azimuth,
            altitude_deg=altitude,
        )
        return RasterArray(
            values=values,
            width=grid.width,
            height=grid.height,
            bbox=grid.bbox,
            source=grid.source,
            units="hillshade",
        )

    async def dem_zonal(
        self,
        *,
        zones: FeatureCollection,
        dem_href: Optional[str] = None,
        dem_source: Optional[str] = None,
        measure: str = "elevation",
        stats: Optional[List[str]] = None,
        band: int = 1,
        pixel_size_m: Optional[List[float]] = None,
    ) -> List[ZoneStat]:
        """Per-zone elevation/slope/aspect over a DEM COG (byte-range read).

        Supply exactly one of ``dem_href`` (a specific DEM COG) or ``dem_source``
        (a named source such as ``"glo30"`` whose overlapping tiles are resolved
        and mosaicked automatically). Reads only the DEM tiles overlapping
        ``zones`` (which must be in the DEM's CRS), builds the requested
        ``measure`` surface, and reduces it to one ``ZoneStat`` per zone.
        ``slope``/``aspect`` use the DEM's pixel size (or ``pixel_size_m`` metres)
        for the gradient.
        """
        return await _dem_zonal(
            dem_href=dem_href,
            dem_source=dem_source,
            zones=zones,
            measure=measure,
            stats=stats,
            band=band,
            pixel_size_m=pixel_size_m,
            http=self.http,
        )


def main() -> None:
    """Console entry point: serve geo-terrain over MCP stdio.

    Runs the startup credential guard, then serves this server's registered
    tools over stdin/stdout via the shared geo-common MCP runtime until the
    client disconnects.
    """
    GeoTerrainServer().run()

"""The ``geo-ops`` MCP server (Pillar B, MVP).

``geo-ops`` provides local, pure-function geospatial processing built on
Shapely/GEOS and GeoPandas. This module wires the geometry operations
(:mod:`geo_ops.geometry`) into the shared
:class:`~geo_common.server.BaseGeoServer` contract and registers them as MCP
tools.

Task 4.3 delivers the geometry operations:

* ``validate_geometry`` - report validity and a reason when invalid
  (Requirements 8.2, 8.3).
* ``spatial_join`` - attribute join by spatial predicate.
* ``overlay`` - geometric set operations.

The ``transform_crs`` CRS-reprojection tool (task 4.1, Requirement 8.1) is
registered separately when its module lands; the server is structured so it can
be added without disturbing the geometry tools.
"""

from __future__ import annotations

from typing import List

from geo_common.models import CatalogEntry, CredentialSpec, OpennessTier
from geo_common.server import BaseGeoServer

from geo_ops.geometry import (
    buffer as _buffer,
    convex_hull as _convex_hull,
    overlay as _overlay,
    simplify as _simplify,
    spatial_join as _spatial_join,
    validate_geometry as _validate_geometry,
)
from geo_ops.transform import transform_crs as _transform_crs
from geo_ops.models import (
    FeatureCollection,
    GeoJSONGeometry,
    GeometryValidity,
)

__all__ = ["GeoOpsServer", "main"]


class GeoOpsServer(BaseGeoServer):
    """Pillar B (MVP) processing server exposing local geometry operations.

    Registers the GEOS-backed geometry tools. ``geo-ops`` is open and
    credential-free (all logic is local), so it starts without any configured
    credentials.
    """

    pillar = "B"
    server_name = "geo-ops"
    version = "0.3.0"

    #: The native capabilities ``geo-ops`` implements itself (PyProj/Shapely).
    #: Each is an open-tier catalog entry provided by ``geo-ops`` (Req 2.1).
    _NATIVE_CAPABILITIES = (
        ("transform_crs", "Reproject a geometry between coordinate reference systems (PyProj)."),
        ("validate_geometry", "Report whether a geometry is topologically valid and why, if not (Shapely/GEOS)."),
        ("spatial_join", "Attribute-join two feature collections by a spatial predicate (Shapely/GEOS)."),
        ("overlay", "Compute a geometric set operation (intersection/union/difference/...) between feature collections."),
        ("buffer", "Compute a buffer polygon around a geometry at a given distance (Shapely/GEOS)."),
        ("convex_hull", "Compute the convex hull of a geometry (Shapely/GEOS)."),
        ("simplify", "Reduce a geometry's vertex count while preserving its shape (Douglas-Peucker; Shapely/GEOS)."),
    )

    def __init__(self, http=None) -> None:
        super().__init__(http=http)
        self.register_tool("transform_crs", self.transform_crs)
        self.register_tool("validate_geometry", self.validate_geometry)
        self.register_tool("spatial_join", self.spatial_join)
        self.register_tool("overlay", self.overlay)
        self.register_tool("buffer", self.buffer)
        self.register_tool("convex_hull", self.convex_hull)
        self.register_tool("simplify", self.simplify)

    async def transform_crs(
        self,
        *,
        geometry: GeoJSONGeometry,
        src_crs: str,
        dst_crs: str,
    ) -> GeoJSONGeometry:
        """Reproject ``geometry`` from ``src_crs`` to ``dst_crs`` (Req 8.1).

        Returns a new geometry with every coordinate reprojected. A round-trip
        to a target CRS and back reproduces the original within the documented
        tolerance (Req 15.2; design Property 1). An unknown CRS or malformed
        coordinates raise a taxonomy ``ValidationError``.

        For a high-vertex geometry, ``simplify`` it first and reproject the
        reduced result — transforming thousands of vertices you are about to
        discard is wasted work and bloats the payload.
        """
        return _transform_crs(
            geometry, src_crs=src_crs, dst_crs=dst_crs, source=self.server_name
        )

    async def validate_geometry(self, *, geometry: GeoJSONGeometry) -> GeometryValidity:
        """Report whether ``geometry`` is valid and why, if not (Req 8.2, 8.3).

        Returns a :class:`GeometryValidity`: ``valid`` reflects the GEOS
        validity test, and ``reason`` carries a non-empty explanation exactly
        when the geometry is invalid.
        """
        return _validate_geometry(geometry, source=self.server_name)

    async def spatial_join(
        self,
        *,
        left: FeatureCollection,
        right: FeatureCollection,
        predicate: str = "intersects",
    ) -> FeatureCollection:
        """Attribute-join ``left`` to ``right`` by a spatial ``predicate``.

        Both collections are inline GeoJSON; reduce high-vertex geometries with
        ``simplify`` first to keep the request payload small.
        """
        return _spatial_join(left, right, predicate=predicate, source=self.server_name)

    async def overlay(
        self,
        *,
        a: FeatureCollection,
        b: FeatureCollection,
        op: str,
    ) -> FeatureCollection:
        """Compute a geometric set operation (``op``) between ``a`` and ``b``.

        Both collections are inline GeoJSON; reduce high-vertex geometries with
        ``simplify`` first to keep the request payload small.
        """
        return _overlay(a, b, op=op, source=self.server_name)

    async def buffer(
        self,
        *,
        geometry: GeoJSONGeometry,
        distance: float,
        resolution: int = 16,
    ) -> GeoJSONGeometry:
        """Compute a buffer polygon around ``geometry`` at ``distance``.

        ``distance`` is in the geometry's coordinate units and may be negative
        (erosion); ``resolution`` sets the segments per quarter circle. A
        non-numeric/non-finite ``distance``, non-positive ``resolution``, or
        unparseable geometry raises a taxonomy ``ValidationError``.
        """
        return _buffer(
            geometry, distance=distance, resolution=resolution, source=self.server_name
        )

    async def convex_hull(self, *, geometry: GeoJSONGeometry) -> GeoJSONGeometry:
        """Compute the convex hull of ``geometry`` (Shapely/GEOS)."""
        return _convex_hull(geometry, source=self.server_name)

    async def simplify(
        self,
        *,
        geometry: GeoJSONGeometry,
        tolerance: float,
        preserve_topology: bool = True,
    ) -> GeoJSONGeometry:
        """Reduce ``geometry``'s vertex count while preserving its shape.

        Douglas-Peucker simplification (Shapely/GEOS): every point of the result
        stays within ``tolerance`` (in the geometry's coordinate units) of the
        input, so concavity is preserved — the shape-preserving alternative to
        ``convex_hull`` for shrinking a high-vertex perimeter before an inline
        geometry op. ``tolerance=0`` returns an equivalent geometry;
        ``preserve_topology=True`` (default) avoids invalid/collapsed output. A
        non-numeric/non-finite/negative ``tolerance``, a non-boolean
        ``preserve_topology``, or unparseable geometry raises a taxonomy
        ``ValidationError``.
        """
        return _simplify(
            geometry,
            tolerance=tolerance,
            preserve_topology=preserve_topology,
            source=self.server_name,
        )

    # ------------------------------------------------------------------
    # Catalog + credential declaration (Req 2.1, 11.3)
    # ------------------------------------------------------------------

    def catalog_entries(self) -> "List[CatalogEntry]":
        """Register every ``geo-ops`` capability in the Resource Catalog (Req 2.1).

        Returns the native PyProj/Shapely capabilities, all open tier and
        provided by ``geo-ops``. ``geo-ops`` implements its full generic-GIS
        surface natively, so there are no wrapped external capabilities to
        merge in.
        """
        native = [
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
        return native

    def required_credentials(self) -> "List[CredentialSpec]":
        """``geo-ops`` is open and credential-free, so this is empty.

        All processing is local (PyProj/Shapely/GeoPandas), so ``geo-ops``
        declares no ``mcp.json`` credential keys and always starts
        (Requirement 16.5).
        """
        return []


def main() -> None:
    """Console entry point: serve geo-ops over MCP stdio.

    Runs the startup credential guard, then serves this server's registered
    tools over stdin/stdout via the shared geo-common MCP runtime until the
    client disconnects.
    """
    GeoOpsServer().run()

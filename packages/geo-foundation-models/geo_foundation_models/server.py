"""The ``geo-foundation-models`` MCP server (Pillar C, MVP).

This module wires the embedding core (:mod:`geo_foundation_models.embedding`)
into the shared :class:`~geo_common.server.BaseGeoServer` contract and exposes
``embed_tile``, ``detect_change``, and ``segment`` as MCP tools.

It also declares the server's Resource Catalog entries (one per capability -
Req 2.1, 11.3) and its single ``mcp.json`` credential (Req 16.1), and inherits
the taxonomy error mapping from :class:`~geo_common.server.BaseGeoServer` so any
library/transport/upstream failure maps onto exactly one ``Error_Taxonomy``
category (Req 11.2, 11.5).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from geo_common.models import (
    CatalogEntry,
    CredentialClassification,
    CredentialSpec,
    OpennessTier,
)
from geo_common.raster_models import FeatureCollection
from geo_common.server import BaseGeoServer

from geo_foundation_models.asset_embedding import (
    detect_change_from_assets as _detect_change_from_assets,
    embed_asset as _embed_asset,
    embed_assets as _embed_assets,
)
from geo_foundation_models.change import detect_change as _detect_change
from geo_foundation_models.change_map import (
    DEFAULT_MAX_TILES,
    DEFAULT_TILE_SIZE,
    change_map as _change_map,
)
from geo_foundation_models.remote_backend import backend_from_env
from geo_foundation_models.clay_embeddings import (
    DEFAULT_LOOKUP_LIMIT,
    DEFAULT_MAX_LOOKUP_AREA_KM2,
    EmbeddingReader,
    available_periods as _available_periods,
    lookup_embeddings as _lookup_embeddings,
)
from geo_foundation_models.embedding import (
    DEFAULT_SUPPORTED_FORMATS,
    EmbeddingBackend,
    ModelRegistry,
    embed_tile as _embed_tile,
    registry_from_env,
)
from geo_foundation_models.segmentation import (
    DEFAULT_MAX_MASK_PIXELS,
    SAMGEO_MODEL_NAME,
    segment as _segment,
)
from geo_foundation_models.models import (
    MAX_TILE_DIMENSION,
    AssetChangeResult,
    ChangeMapResult,
    EmbeddingRecord,
    EmbeddingResult,
    RasterTile,
    SegmentationMask,
)

__all__ = ["GeoFoundationModelsServer", "main"]

#: ``uvx`` command that installs this server (Req 2.6, 6.2; bundle-manifest).
INSTALL_COMMAND = "uvx geo-foundation-models"

#: Optional Hugging Face token for gated model weights; openly licensed models
#: load without it, so the credential never blocks startup (Req 16.5).
HF_TOKEN_KEY = "HF_TOKEN"
HF_TOKEN_SOURCE = "Hugging Face (model weights)"

#: Optional API key for a remote embedding endpoint (generic HTTPS transport).
#: When a remote backend is configured (GEO_FM_EMBED_ENDPOINT_URL or
#: GEO_FM_SAGEMAKER_ENDPOINT), real-weight embeddings replace the deterministic
#: stand-in; the key is Optional so its absence never blocks startup (Req 16.5).
EMBED_ENDPOINT_KEY = "GEO_FM_EMBED_API_KEY"
EMBED_ENDPOINT_KEY_SOURCE = "Remote embedding endpoint (optional real-weight backend)"


class GeoFoundationModelsServer(BaseGeoServer):
    """Foundation_Model_Server exposing geospatial embedding tools (Req 9).

    Holds the configurable :class:`ModelRegistry` (Clay, Prithvi-EO-2.0,
    SatCLIP and any registered additions - Requirement 9.2) and the embedding
    backend, and registers ``embed_tile`` as an MCP tool.
    """

    pillar = "C"
    server_name = "geo-foundation-models"
    version = "0.3.0"

    def __init__(
        self,
        *,
        registry: Optional[ModelRegistry] = None,
        backend: Optional[EmbeddingBackend] = None,
        supported_formats: Iterable[str] = DEFAULT_SUPPORTED_FORMATS,
        max_dimension: int = MAX_TILE_DIMENSION,
        max_mask_pixels: int = DEFAULT_MAX_MASK_PIXELS,
        embedding_reader: "Optional[EmbeddingReader]" = None,
        max_lookup_area_km2: float = DEFAULT_MAX_LOOKUP_AREA_KM2,
        http=None,
    ) -> None:
        super().__init__(http=http)
        self.registry: ModelRegistry = registry if registry is not None else ModelRegistry()
        self.backend = backend
        self.supported_formats = frozenset(fmt.lower() for fmt in supported_formats)
        self.max_dimension = max_dimension
        self.max_mask_pixels = max_mask_pixels
        self.embedding_reader = embedding_reader
        self.max_lookup_area_km2 = max_lookup_area_km2
        self.register_tool("embed_tile", self.embed_tile)
        self.register_tool("embed_asset", self.embed_asset)
        self.register_tool("embed_assets", self.embed_assets)
        self.register_tool("detect_change", self.detect_change)
        self.register_tool("detect_change_from_assets", self.detect_change_from_assets)
        self.register_tool("change_map", self.change_map)
        self.register_tool("segment", self.segment)
        self.register_tool("lookup_embeddings", self.lookup_embeddings)
        self.register_tool("available_embedding_periods", self.available_embedding_periods)

    # ------------------------------------------------------------------
    # Catalog + credential declaration (Req 2.1, 11.3, 16.1)
    # ------------------------------------------------------------------

    def catalog_entries(self) -> "List[CatalogEntry]":
        """The Pillar C capabilities this server registers (Req 2.1, 11.3).

        One :class:`~geo_common.models.CatalogEntry` per exposed tool -
        ``embed_tile``, ``detect_change``, and ``segment`` - each naming this
        server as ``provider_server`` (Req 11.3) and recording the entry name,
        pillar, capability description, and ``Openness_Tier`` required by the
        Resource Catalog (Req 2.1). The geospatial foundation models bundled
        here (Clay, Prithvi-EO-2.0, SatCLIP, SAMGeo) are openly licensed, so
        each entry's tier is :attr:`OpennessTier.OPEN`.
        """
        return [
            CatalogEntry(
                name="embed_tile",
                pillar=self.pillar,
                capability_description=(
                    "Embed a tile (<=1024x1024) with one geospatial foundation "
                    "model (Clay, Prithvi-EO-2.0, SatCLIP, ...), returning a "
                    "model-dimensioned vector. The default backend is a "
                    "deterministic stand-in with NO semantic structure, so it "
                    "cannot rank change severity (wire a real-weight backend for "
                    "that); the 'backend' and 'structure_only' fields record "
                    "provenance. For real published Clay v1.5 vectors use "
                    "lookup_embeddings."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                installed=True,
            ),
            CatalogEntry(
                name="embed_asset",
                pillar=self.pillar,
                capability_description=(
                    "Embed a COG window server-side: read only the overlapping "
                    "tiles of a raster href (s3://http(s)) over an optional "
                    "bbox + bands by byte range, build the tile, and embed it "
                    "with one model. Removes inline pixel plumbing. Same backend "
                    "provenance caveats as embed_tile."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                installed=True,
            ),
            CatalogEntry(
                name="embed_assets",
                pillar=self.pillar,
                capability_description=(
                    "Embed a window read across SEPARATE single-band COGs (one "
                    "per band, in the model's band order) — the multi-band "
                    "counterpart to embed_asset for catalogs like Earth Search "
                    "where each band is its own .tif. Reads band 1 of each over "
                    "a shared bbox by byte range, checks grid alignment, stacks, "
                    "and embeds. Same backend provenance caveats as embed_tile."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                installed=True,
            ),
            CatalogEntry(
                name="detect_change",
                pillar=self.pillar,
                capability_description=(
                    "Change detection between two equal-dimension embeddings "
                    "of the same area, returning a scalar change measure "
                    "normalized to [0.0, 1.0]. NOTE: only meaningful when the "
                    "embeddings come from a real-weight backend; with the "
                    "deterministic stand-in any two differing tiles score ~0.5 "
                    "and two structure-only (no-pixel) tiles score exactly 0.0, "
                    "so a value is not a calibrated measurement there."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                installed=True,
            ),
            CatalogEntry(
                name="detect_change_from_assets",
                pillar=self.pillar,
                capability_description=(
                    "Change between two COG hrefs over the same window+bands in "
                    "one call: read+embed each server-side, then compare. "
                    "Returns the [0,1] measure with backend provenance and a "
                    "'caveat' whenever it is not calibrated (deterministic "
                    "stand-in or structure-only read)."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                installed=True,
            ),
            CatalogEntry(
                name="change_map",
                pillar=self.pillar,
                capability_description=(
                    "Per-tile change GRID between two co-registered COG hrefs "
                    "over an AOI: tiles the window, reads+embeds both dates per "
                    "tile server-side, and reduces each to a [0,1] change "
                    "measure. Only meaningful with a real-weight backend — under "
                    "the deterministic stand-in the grid is ~0.5 noise, so it "
                    "sets calibrated=false + a caveat (or refuses when "
                    "require_real_backend). Tile count is bounded; route large "
                    "AOIs to aws-geo-compute."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                installed=True,
            ),
            CatalogEntry(
                name="segment",
                pillar=self.pillar,
                capability_description=(
                    "SAMGeo-style segmentation of an imagery tile into a "
                    "per-pixel mask, optionally seeded by point/box prompts. "
                    "The default backend is a deterministic local stand-in "
                    "(pluggable real-weight backends); the result's 'backend' "
                    "field records which produced it."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                installed=True,
            ),
            CatalogEntry(
                name="lookup_embeddings",
                pillar=self.pillar,
                capability_description=(
                    "Retrieve precomputed open Clay v1.5 (1024-d) Sentinel-2 "
                    "embeddings for a bounding box from the LGND / Source "
                    "Cooperative Open Data dataset (CC-BY 4.0, anonymous S3); "
                    "bbox area and result count are bounded."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                installed=True,
            ),
            CatalogEntry(
                name="available_embedding_periods",
                pillar=self.pillar,
                capability_description=(
                    "List the months ('YYYY-MM') for which published Clay v1.5 "
                    "embeddings exist, so a before/after outside them is known "
                    "to be unusable before calling lookup_embeddings."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                installed=True,
            ),
        ]

    def required_credentials(self) -> "List[CredentialSpec]":
        """The ``mcp.json`` keys this server reads, with classification (Req 16.1).

        Declares the single Optional ``HF_TOKEN`` (Hugging Face) credential used
        only for gated model weights. Because it is
        :attr:`CredentialClassification.OPTIONAL`, an absent value never blocks
        startup (Req 16.5): openly licensed models load without it.
        """
        return [
            CredentialSpec(
                source=HF_TOKEN_SOURCE,
                mcp_json_key=HF_TOKEN_KEY,
                classification=CredentialClassification.OPTIONAL,
            ),
            CredentialSpec(
                source=EMBED_ENDPOINT_KEY_SOURCE,
                mcp_json_key=EMBED_ENDPOINT_KEY,
                classification=CredentialClassification.OPTIONAL,
            ),
        ]

    async def embed_tile(self, *, tile: RasterTile, model: str) -> EmbeddingResult:
        """Embed an imagery tile with exactly one selected model (Req 9.1/9.2/9.7).

        Selects ``model`` from this server's configurable registry, validates
        the tile (``<= 1024 x 1024`` in a supported format) before any
        embedding work, and returns a model-determined
        :class:`EmbeddingResult`. Empty, oversized, or unsupported tiles, and
        unknown models, raise an ``Error_Taxonomy`` ``ValidationError`` and
        produce no embedding.
        """
        return _embed_tile(
            tile,
            model,
            registry=self.registry,
            backend=self.backend,
            supported_formats=self.supported_formats,
            max_dimension=self.max_dimension,
        )


    async def embed_asset(
        self,
        *,
        raster_href: str,
        model: str,
        window_bbox: Optional[List[float]] = None,
        bands: Optional[List[int]] = None,
        latlon: Optional[List[float]] = None,
        acquired: Optional[str] = None,
    ) -> EmbeddingResult:
        """Read a COG window by byte range and embed it server-side (Req 9.1).

        Reads only the tiles overlapping ``window_bbox`` (in the asset's CRS;
        omit for the whole asset) for the selected ``bands`` (default ``[1]``),
        builds the tile, and embeds it with ``model`` — removing the need to
        inline pixels. Same validation and backend-provenance contract as
        ``embed_tile``.
        """
        return await _embed_asset(
            raster_href=raster_href,
            model=model,
            window_bbox=window_bbox,
            bands=tuple(bands) if bands else (1,),
            latlon=latlon,
            acquired=acquired,
            registry=self.registry,
            backend=self.backend,
            http=self.http,
            supported_formats=self.supported_formats,
            max_dimension=self.max_dimension,
        )

    async def embed_assets(
        self,
        *,
        assets: List[str],
        model: str,
        window_bbox: Optional[List[float]] = None,
        latlon: Optional[List[float]] = None,
        acquired: Optional[str] = None,
    ) -> EmbeddingResult:
        """Embed a window across separate single-band COGs, one per band.

        ``assets`` is an ordered list of single-band COG hrefs in the model's
        band order (e.g. Sentinel-2 bands as separate ``.tif`` files on Earth
        Search). Band 1 of each is read over ``window_bbox`` (assets' CRS; omit
        for the whole scene), the bands are stacked in order into one tile, and
        embedded with ``model``. The assets must share one pixel grid over the
        window; a mismatch raises a ``ValidationError``.
        """
        return await _embed_assets(
            assets=assets,
            model=model,
            window_bbox=window_bbox,
            latlon=latlon,
            acquired=acquired,
            registry=self.registry,
            backend=self.backend,
            http=self.http,
            supported_formats=self.supported_formats,
            max_dimension=self.max_dimension,
        )

    async def detect_change_from_assets(
        self,
        *,
        raster_href_a: str,
        raster_href_b: str,
        model: str,
        window_bbox: Optional[List[float]] = None,
        bands: Optional[List[int]] = None,
    ) -> AssetChangeResult:
        """Change between two COG hrefs over the same window, in one call.

        Reads+embeds the same ``window_bbox``/``bands`` of both assets, then
        returns the ``detect_change`` measure with backend provenance and a
        ``caveat`` when the score is not calibrated (deterministic stand-in or a
        structure-only read).
        """
        return await _detect_change_from_assets(
            raster_href_a=raster_href_a,
            raster_href_b=raster_href_b,
            model=model,
            window_bbox=window_bbox,
            bands=tuple(bands) if bands else (1,),
            registry=self.registry,
            backend=self.backend,
            http=self.http,
            supported_formats=self.supported_formats,
            max_dimension=self.max_dimension,
        )

    async def change_map(
        self,
        *,
        raster_href_a: str,
        raster_href_b: str,
        model: str,
        aoi_bbox: List[float],
        tile_size: int = DEFAULT_TILE_SIZE,
        bands: Optional[List[int]] = None,
        zones: Optional[FeatureCollection] = None,
        require_real_backend: bool = False,
        max_tiles: int = DEFAULT_MAX_TILES,
    ) -> ChangeMapResult:
        """Per-tile change grid between two co-registered COGs over an AOI.

        Tiles ``aoi_bbox`` (in the assets' CRS) into ``tile_size``-pixel tiles,
        reads+embeds both dates per tile server-side, and reduces each to a
        ``[0,1]`` change measure. Under the deterministic stand-in backend the
        grid is uncalibrated (~0.5 noise): the result carries ``calibrated=false``
        and a ``caveat``, or the call refuses when ``require_real_backend`` is
        set. The tile count is bounded by ``max_tiles``. When ``zones`` (a
        GeoJSON FeatureCollection in the assets' CRS) is supplied, the grid is
        also reduced per zone (mean/max change over the tiles each zone contains).
        """
        return await _change_map(
            raster_href_a=raster_href_a,
            raster_href_b=raster_href_b,
            model=model,
            aoi_bbox=aoi_bbox,
            tile_size=tile_size,
            bands=tuple(bands) if bands else (1,),
            zones=zones,
            require_real_backend=require_real_backend,
            max_tiles=max_tiles,
            registry=self.registry,
            backend=self.backend,
            http=self.http,
            supported_formats=self.supported_formats,
            max_dimension=self.max_dimension,
        )

    async def detect_change(
        self, *, embedding_a: List[float], embedding_b: List[float]
    ) -> float:
        """Change measure in ``[0.0, 1.0]`` for two same-area embeddings (Req 9.5/9.9).

        Requires equal dimensionality; a mismatch raises an ``Error_Taxonomy``
        ``ValidationError`` and produces no measure (Requirement 9.9). Higher
        values indicate greater change, and two identical embeddings yield the
        minimum value ``0.0`` (Requirement 9.5).
        """
        return _detect_change(embedding_a, embedding_b)

    async def segment(
        self, *, tile: RasterTile, prompts: Optional[List[Dict]] = None
    ) -> SegmentationMask:
        """SAMGeo-style segmentation of a validated tile (Pillar C, Req 9).

        Validates the tile (and any prompt coordinates) before segmenting, so
        empty / oversized / unsupported tiles raise an ``Error_Taxonomy``
        ``ValidationError`` and produce no mask. Returns a per-pixel
        :class:`SegmentationMask`; with ``prompts`` each prompt seeds a region,
        otherwise the tile is partitioned automatically by intensity.
        """
        return _segment(
            tile,
            prompts,
            model=SAMGEO_MODEL_NAME,
            supported_formats=self.supported_formats,
            max_dimension=self.max_dimension,
            max_mask_pixels=self.max_mask_pixels,
        )


    async def lookup_embeddings(
        self,
        *,
        bbox: Sequence[float],
        start: Optional[str] = None,
        end: Optional[str] = None,
        product: str = "aggregated",
        limit: int = DEFAULT_LOOKUP_LIMIT,
    ) -> List[EmbeddingRecord]:
        """Retrieve published open Clay v1.5 embeddings for ``bbox`` (bounded).

        Looks up precomputed 1024-d Clay v1.5 Sentinel-2 embeddings from the
        open LGND / Source Cooperative dataset (CC-BY 4.0, anonymous S3) for the
        given bounding box and optional ``[start, end]`` date range. ``product``
        is ``"aggregated"`` (one record per MajorTOM cell per month) or
        ``"scene"`` (per Sentinel-2 scene). The bbox area is gated and the
        number of returned records is capped at ``limit`` (default 50) so the
        response stays small; the dataset only publishes June 2024 and June
        2025, so a range outside those returns no records.
        """
        return _lookup_embeddings(
            bbox=bbox,
            start=start,
            end=end,
            product=product,
            limit=limit,
            reader=self.embedding_reader,
            max_area_km2=self.max_lookup_area_km2,
        )

    async def available_embedding_periods(self) -> List[str]:
        """List the months (``"YYYY-MM"``) with published Clay v1.5 embeddings.

        The open Clay v1.5 dataset only publishes a few monthly partitions, so
        the real-embedding ``lookup_embeddings`` path is unusable outside them.
        Calling this first lets an agent see the coverage (currently June 2024
        and June 2025) instead of inferring it from an empty lookup result.
        """
        return _available_periods()


def main() -> None:
    """Console entry point: serve geo-foundation-models over MCP stdio.

    Runs the startup credential guard, then serves this server's registered
    tools over stdin/stdout via the shared geo-common MCP runtime until the
    client disconnects.

    If a remote embedding endpoint is configured in the environment
    (``GEO_FM_EMBED_ENDPOINT_URL`` or ``GEO_FM_SAGEMAKER_ENDPOINT``), a real-
    weight backend is wired in so ``embed_*`` and ``change_map`` produce
    calibrated results; otherwise the honest deterministic stand-in backend is
    used. Extra models declared in ``GEO_FM_EXTRA_MODELS`` are registered so a
    bring-your-own model's dimension is selectable without code.
    """
    GeoFoundationModelsServer(
        registry=registry_from_env(), backend=backend_from_env()
    ).run()

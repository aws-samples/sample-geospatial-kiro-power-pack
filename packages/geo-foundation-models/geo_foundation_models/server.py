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
from geo_common.server import BaseGeoServer

from geo_foundation_models.change import detect_change as _detect_change
from geo_foundation_models.clay_embeddings import (
    DEFAULT_LOOKUP_LIMIT,
    DEFAULT_MAX_LOOKUP_AREA_KM2,
    EmbeddingReader,
    lookup_embeddings as _lookup_embeddings,
)
from geo_foundation_models.embedding import (
    DEFAULT_SUPPORTED_FORMATS,
    EmbeddingBackend,
    ModelRegistry,
    embed_tile as _embed_tile,
)
from geo_foundation_models.segmentation import (
    DEFAULT_MAX_MASK_PIXELS,
    SAMGEO_MODEL_NAME,
    segment as _segment,
)
from geo_foundation_models.models import (
    MAX_TILE_DIMENSION,
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


class GeoFoundationModelsServer(BaseGeoServer):
    """Foundation_Model_Server exposing geospatial embedding tools (Req 9).

    Holds the configurable :class:`ModelRegistry` (Clay, Prithvi-EO-2.0,
    SatCLIP and any registered additions - Requirement 9.2) and the embedding
    backend, and registers ``embed_tile`` as an MCP tool.
    """

    pillar = "C"
    server_name = "geo-foundation-models"
    version = "0.2.0"

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
        self.register_tool("detect_change", self.detect_change)
        self.register_tool("segment", self.segment)
        self.register_tool("lookup_embeddings", self.lookup_embeddings)

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
                    "Embed an imagery tile (<=1024x1024) with one selected "
                    "geospatial foundation model (Clay, Prithvi-EO-2.0, "
                    "SatCLIP, ...), returning a model-dimensioned vector. The "
                    "default backend is a deterministic local stand-in "
                    "(pluggable real-weight backends); the result's 'backend' "
                    "field records which produced it. For real published Clay "
                    "v1.5 vectors use lookup_embeddings."
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
                    "normalized to [0.0, 1.0]."
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
            )
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


def main() -> None:
    """Console entry point: serve geo-foundation-models over MCP stdio.

    Runs the startup credential guard, then serves this server's registered
    tools over stdin/stdout via the shared geo-common MCP runtime until the
    client disconnects.
    """
    GeoFoundationModelsServer().run()

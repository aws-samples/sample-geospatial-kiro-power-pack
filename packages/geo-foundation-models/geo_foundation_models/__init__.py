"""geo-foundation-models: Pillar C (MVP) GeoAI MCP server.

Exposes geospatial foundation-model embedding capabilities. Task 7.1 delivers
``embed_tile`` (model selection + tile validation + model-determined
embedding) and the embedding data models; task 7.2 adds ``detect_change``
(normalized embedding change measure) and ``segment`` (SAMGeo-style
segmentation). Catalog registration follows in a later task.
"""

from __future__ import annotations

from geo_foundation_models.embedding import (
    DEFAULT_MODELS,
    DEFAULT_SUPPORTED_FORMATS,
    DeterministicLocalBackend,
    EmbeddingBackend,
    ModelRegistry,
    embed_tile,
    extra_models_from_env,
    registry_from_env,
    validate_tile,
)
from geo_foundation_models.change import detect_change
from geo_foundation_models.change_map import change_map
from geo_foundation_models.asset_embedding import (
    detect_change_from_assets,
    embed_asset,
    embed_assets,
)
from geo_foundation_models.remote_backend import RemoteEndpointBackend, backend_from_env
from geo_foundation_models.local_backend import LocalCallableBackend
from geo_foundation_models.segmentation import SAMGEO_MODEL_NAME, segment
from geo_foundation_models.clay_embeddings import (
    CLAY_V15_DIMENSION,
    CLAY_V15_MODEL_NAME,
    available_periods,
    lookup_embeddings,
)
from geo_foundation_models.models import (
    MAX_TILE_DIMENSION,
    AssetChangeResult,
    ChangeMapCell,
    ChangeMapResult,
    ZoneChange,
    EmbeddingMetadata,
    EmbeddingRecord,
    EmbeddingResult,
    ModelSpec,
    RasterTile,
    SegmentationMask,
)
from geo_foundation_models.server import GeoFoundationModelsServer

__all__ = [
    # Data models (task 7.1)
    "RasterTile",
    "ModelSpec",
    "EmbeddingResult",
    "AssetChangeResult",
    "ChangeMapCell",
    "ChangeMapResult",
    "ZoneChange",
    "EmbeddingMetadata",
    "EmbeddingRecord",
    "SegmentationMask",
    "MAX_TILE_DIMENSION",
    # Embedding core
    "DEFAULT_MODELS",
    "DEFAULT_SUPPORTED_FORMATS",
    "ModelRegistry",
    "EmbeddingBackend",
    "DeterministicLocalBackend",
    "validate_tile",
    "embed_tile",
    "extra_models_from_env",
    "registry_from_env",
    # GeoAI tools (task 7.2)
    "detect_change",
    "segment",
    "SAMGEO_MODEL_NAME",
    # Read-then-embed bridges
    "embed_asset",
    "embed_assets",
    "detect_change_from_assets",
    "change_map",
    # Real-weight backend seam (remote inference endpoint + local in-process)
    "RemoteEndpointBackend",
    "LocalCallableBackend",
    "backend_from_env",
    # Open Clay v1.5 embedding lookup (LGND / Source Cooperative)
    "lookup_embeddings",
    "available_periods",
    "CLAY_V15_DIMENSION",
    "CLAY_V15_MODEL_NAME",
    # Server
    "GeoFoundationModelsServer",
]

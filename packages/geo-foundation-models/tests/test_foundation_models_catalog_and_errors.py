"""Catalog/credential registration and error-mapping tests for ``geo-foundation-models``.

Covers task 7.4 of the Geospatial Power Pack spec:

* Req 2.1 / 11.3 - :meth:`GeoFoundationModelsServer.catalog_entries` registers
  one Resource Catalog entry per exposed tool (``embed_tile``,
  ``detect_change``, ``segment``), each naming ``geo-foundation-models`` as the
  ``provider_server`` (Req 11.3) and carrying name, pillar, capability
  description, and ``Openness_Tier`` (Req 2.1). The bundled foundation models
  (Clay, Prithvi-EO-2.0, SatCLIP, SAMGeo) are openly licensed, so every entry's
  tier is :attr:`OpennessTier.OPEN`.
* Req 16.1 / 16.5 - :meth:`GeoFoundationModelsServer.required_credentials`
  declares the single Optional ``HF_TOKEN`` key, so the startup credential
  guard never blocks (openly licensed weights load without it).
* Req 11.2 / 11.5 - every library/transport/upstream failure maps onto exactly
  one ``Error_Taxonomy`` category. ``embed_tile`` / ``detect_change`` /
  ``segment`` raise ``ValidationError`` (taxonomy ``validation``) on bad input
  with no result produced, and the inherited :meth:`BaseGeoServer.map_error`
  routes transport, HTTP-status, and unmapped library errors onto the taxonomy
  (design Property 8 - mapping totality).
"""

from __future__ import annotations

import httpx
import pytest

from geo_common.errors import (
    AuthenticationError,
    ErrorCategory,
    GeoError,
    NetworkError,
    UpstreamError,
    ValidationError,
)
from geo_common.models import (
    CredentialClassification,
    OpennessTier,
)

from geo_foundation_models.models import RasterTile
from geo_foundation_models.server import (
    HF_TOKEN_KEY,
    GeoFoundationModelsServer,
)

EXPECTED_TOOLS = {
    "embed_tile",
    "embed_asset",
    "detect_change",
    "detect_change_from_assets",
    "segment",
    "lookup_embeddings",
    "available_embedding_periods",
}


def _tile(width: int = 8, height: int = 8, bands: int = 1, fmt: str = "GTiff", data=None) -> RasterTile:
    return RasterTile(width=width, height=height, bands=bands, format=fmt, data=data)


# --- Catalog registration (Req 2.1 / 11.3) --------------------------------


def test_catalog_entries_cover_every_tool() -> None:
    """Req 2.1 / 11.3: one catalog entry per exposed tool, attributed correctly."""
    server = GeoFoundationModelsServer()
    entries = server.catalog_entries()

    assert {entry.name for entry in entries} == EXPECTED_TOOLS
    # The catalog mirrors the registered MCP tools (no orphan/missing entries).
    assert {entry.name for entry in entries} == set(server.tool_names())


def test_catalog_entries_are_open_pillar_c_and_self_provided() -> None:
    """Req 2.1 / 11.3: each entry is Pillar C, OPEN, and names this server."""
    server = GeoFoundationModelsServer()

    for entry in server.catalog_entries():
        assert entry.pillar == "C"
        assert entry.provider_server == "geo-foundation-models"
        assert entry.openness_tier is OpennessTier.OPEN
        # Resource Catalog requires a non-empty, bounded capability description.
        assert 1 <= len(entry.capability_description) <= 500


# --- Credential registration (Req 16.1 / 16.5) ----------------------------


def test_required_credentials_declare_optional_hf_token() -> None:
    """Req 16.1: the single HF_TOKEN credential is declared and Optional."""
    server = GeoFoundationModelsServer()
    specs = server.required_credentials()

    assert {spec.mcp_json_key for spec in specs} == {HF_TOKEN_KEY}
    assert all(
        spec.classification is CredentialClassification.OPTIONAL for spec in specs
    )


def test_server_starts_with_no_credentials_configured() -> None:
    """Req 16.5: only Optional credentials -> startup is never blocked."""
    server = GeoFoundationModelsServer()
    server.start(configured_keys=[])  # nothing configured
    assert server.started is True


# --- Validation errors map onto the taxonomy (Req 11.2) -------------------


async def test_embed_tile_unknown_model_is_validation_error() -> None:
    """Req 11.2: an unknown model selection maps onto VALIDATION, no embedding."""
    server = GeoFoundationModelsServer()
    with pytest.raises(ValidationError) as exc_info:
        await server.embed_tile(tile=_tile(), model="NoSuchModel")
    assert exc_info.value.category is ErrorCategory.VALIDATION


async def test_embed_tile_oversized_tile_is_validation_error() -> None:
    """Req 11.2: an oversized tile maps onto VALIDATION, no embedding."""
    server = GeoFoundationModelsServer()
    with pytest.raises(ValidationError) as exc_info:
        await server.embed_tile(tile=_tile(width=2048, height=2048), model="Clay")
    assert exc_info.value.category is ErrorCategory.VALIDATION


async def test_detect_change_dimension_mismatch_is_validation_error() -> None:
    """Req 11.2: a dimensionality mismatch maps onto VALIDATION, no measure."""
    server = GeoFoundationModelsServer()
    with pytest.raises(ValidationError) as exc_info:
        await server.detect_change(embedding_a=[1.0, 2.0], embedding_b=[1.0, 2.0, 3.0])
    assert exc_info.value.category is ErrorCategory.VALIDATION


async def test_segment_out_of_bounds_prompt_is_validation_error() -> None:
    """Req 11.2: an out-of-bounds prompt maps onto VALIDATION, no mask."""
    server = GeoFoundationModelsServer()
    with pytest.raises(ValidationError) as exc_info:
        await server.segment(
            tile=_tile(width=8, height=8),
            prompts=[{"type": "point", "x": 100, "y": 100}],
        )
    assert exc_info.value.category is ErrorCategory.VALIDATION


# --- map_error covers the whole taxonomy (Req 11.2 / 11.5; Property 8) -----


def test_map_error_passes_through_existing_geoerror_unchanged() -> None:
    """Property 8: an already-classified GeoError keeps its single category."""
    server = GeoFoundationModelsServer()
    original = ValidationError("bad tile", source="geo-foundation-models")
    mapped = server.map_error(original, source="geo-foundation-models")
    assert mapped is original
    assert mapped.category is ErrorCategory.VALIDATION


def test_map_error_routes_unmapped_library_error_to_upstream() -> None:
    """Req 11.5: an unmapped library error -> UPSTREAM, original retained."""
    server = GeoFoundationModelsServer()
    mapped = server.map_error(
        RuntimeError("model weights failed to load"),
        source="geo-foundation-models",
    )
    assert isinstance(mapped, UpstreamError)
    assert mapped.category is ErrorCategory.UPSTREAM
    assert mapped.original is not None
    assert "model weights failed to load" in mapped.original


def test_map_error_routes_transport_failure_to_network() -> None:
    """Req 5.4 / 5.9: a transport failure (e.g. gated-weights fetch) -> NETWORK."""
    server = GeoFoundationModelsServer()
    request = httpx.Request("GET", "https://huggingface.co/model")
    mapped = server.map_error(
        httpx.ConnectError("connection refused", request=request),
        source="geo-foundation-models",
    )
    assert isinstance(mapped, NetworkError)
    assert mapped.category is ErrorCategory.NETWORK


def test_map_error_maps_http_401_to_authentication() -> None:
    """Req 11.2: a 401 (e.g. gated HF weights) maps onto AUTHENTICATION."""
    server = GeoFoundationModelsServer()
    request = httpx.Request("GET", "https://huggingface.co/model")
    response = httpx.Response(401, request=request)
    exc = httpx.HTTPStatusError("unauthorized", request=request, response=response)
    mapped = server.map_error(exc, source="geo-foundation-models")
    assert isinstance(mapped, AuthenticationError)
    assert mapped.category is ErrorCategory.AUTHENTICATION


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("bad value"),
        KeyError("missing"),
        RuntimeError("boom"),
        httpx.ConnectError(
            "refused", request=httpx.Request("GET", "https://example.org")
        ),
    ],
)
def test_map_error_always_yields_exactly_one_taxonomy_category(exc: Exception) -> None:
    """Property 8: map_error is total - every error gets exactly one category."""
    server = GeoFoundationModelsServer()
    mapped = server.map_error(exc, source="geo-foundation-models")
    assert isinstance(mapped, GeoError)
    assert isinstance(mapped.category, ErrorCategory)

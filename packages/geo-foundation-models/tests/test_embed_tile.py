"""Unit tests for ``embed_tile`` and model selection (Req 9.1, 9.2, 9.7).

These example-based tests cover the task 7.1 contract:

* a valid tile yields a model-determined embedding within the size limit (9.1);
* exactly one model is selectable from the configurable set including Clay,
  Prithvi-EO-2.0, and SatCLIP, and an unknown model is rejected (9.2);
* empty / oversized / unsupported tiles are rejected with a validation error
  and produce no embedding (9.7).
"""

from __future__ import annotations

import pytest

from geo_common.errors import ErrorCategory, ValidationError

from geo_foundation_models.embedding import (
    DEFAULT_MODELS,
    ModelRegistry,
    embed_tile,
)
from geo_foundation_models.models import EmbeddingResult, ModelSpec, RasterTile


def _tile(width: int = 64, height: int = 64, bands: int = 3, fmt: str = "GTiff", data=None) -> RasterTile:
    return RasterTile(width=width, height=height, bands=bands, format=fmt, data=data)


# --- Requirement 9.1: model-determined embedding for a valid tile ---------

def test_embed_tile_returns_model_dimension():
    result = embed_tile(_tile(), "Clay")
    assert isinstance(result, EmbeddingResult)
    assert result.model == "Clay"
    assert result.dimension == DEFAULT_MODELS["Clay"].dimension
    assert len(result.vector) == result.dimension


def test_embed_tile_records_deterministic_backend_provenance():
    """The default embedding self-describes as the deterministic stand-in.

    Guards the honesty contract: a caller can tell a stand-in vector apart from
    real model-weight inference via the result's ``backend`` field.
    """
    result = embed_tile(_tile(), "Clay")
    assert result.backend == "deterministic-local"


def test_embed_tile_is_deterministic_and_content_dependent():
    a1 = embed_tile(_tile(data=[0.1] * (64 * 64 * 3)), "Clay")
    a2 = embed_tile(_tile(data=[0.1] * (64 * 64 * 3)), "Clay")
    b = embed_tile(_tile(data=[0.9] * (64 * 64 * 3)), "Clay")
    assert a1.vector == a2.vector  # deterministic
    assert a1.vector != b.vector  # content-dependent


def test_max_size_tile_is_accepted():
    result = embed_tile(_tile(width=1024, height=1024, bands=1), "SatCLIP")
    assert result.dimension == DEFAULT_MODELS["SatCLIP"].dimension


# --- Requirement 9.2: exactly one model from the configurable set ---------

@pytest.mark.parametrize("model", ["Clay", "Prithvi-EO-2.0", "SatCLIP"])
def test_required_models_are_selectable(model):
    result = embed_tile(_tile(), model)
    assert result.model == model
    assert result.dimension == DEFAULT_MODELS[model].dimension


@pytest.mark.parametrize(
    "given, canonical",
    [("clay", "Clay"), ("CLAY", "Clay"), ("satclip", "SatCLIP"), ("prithvi-eo-2.0", "Prithvi-EO-2.0")],
)
def test_model_name_lookup_is_case_insensitive(given, canonical):
    """A case-variant model name resolves to the canonical registered spec."""
    result = embed_tile(_tile(), given)
    assert result.model == canonical
    assert result.dimension == DEFAULT_MODELS[canonical].dimension


def test_unknown_model_is_rejected():
    with pytest.raises(ValidationError) as exc:
        embed_tile(_tile(), "NotAModel")
    assert exc.value.category is ErrorCategory.VALIDATION


def test_registry_is_configurable():
    reg = ModelRegistry()
    reg.register(ModelSpec(name="MyModel", dimension=42))
    result = embed_tile(_tile(), "MyModel", registry=reg)
    assert result.dimension == 42
    reg.unregister("Clay")
    with pytest.raises(ValidationError):
        embed_tile(_tile(), "Clay", registry=reg)


# --- Requirement 9.7: empty / oversized / unsupported tiles rejected ------

@pytest.mark.parametrize(
    "tile",
    [
        _tile(width=0, height=64),          # empty: no width
        _tile(width=64, height=0),          # empty: no height
        _tile(width=64, height=64, bands=0),  # empty: no bands
    ],
)
def test_empty_tile_rejected(tile):
    with pytest.raises(ValidationError) as exc:
        embed_tile(tile, "Clay")
    assert exc.value.category is ErrorCategory.VALIDATION


def test_empty_inlined_data_rejected():
    with pytest.raises(ValidationError):
        embed_tile(_tile(data=[]), "Clay")


@pytest.mark.parametrize(
    "tile",
    [
        _tile(width=1025, height=64),
        _tile(width=64, height=2000),
        _tile(width=4096, height=4096),
    ],
)
def test_oversized_tile_rejected(tile):
    with pytest.raises(ValidationError) as exc:
        embed_tile(tile, "Clay")
    assert exc.value.category is ErrorCategory.VALIDATION


def test_unsupported_format_rejected():
    with pytest.raises(ValidationError) as exc:
        embed_tile(_tile(fmt="bmp"), "Clay")
    assert exc.value.category is ErrorCategory.VALIDATION


def test_mismatched_data_length_rejected():
    with pytest.raises(ValidationError):
        embed_tile(_tile(width=10, height=10, bands=1, data=[0.0] * 7), "Clay")


def test_invalid_input_produces_no_embedding():
    # The validation error path must not return a partial EmbeddingResult.
    try:
        embed_tile(_tile(width=0), "Clay")
    except ValidationError as exc:
        assert exc.category is ErrorCategory.VALIDATION
    else:  # pragma: no cover
        pytest.fail("expected ValidationError for empty tile")


# --- Req 9.7: a malformed/uncoerced tile is a clean ValidationError --------
# Regression: when the MCP runtime cannot coerce a JSON tile argument into a
# RasterTile (e.g. the client sends a base64 image string, or a dict whose
# fields are wrong), the raw value reaches the tool. It must surface as an
# Error_Taxonomy ValidationError rather than an opaque AttributeError.


@pytest.mark.parametrize(
    "bad_tile",
    [
        "iVBORw0KGgoAAAANSUhEUg==",            # a base64 image string
        {"image": "data:image/png;base64,..."},  # wrong-shaped dict
        {"width": 64, "height": 64},            # dict missing required 'format'
        12345,                                   # not a tile at all
        None,                                    # missing
    ],
)
def test_malformed_tile_raises_clean_validation_error(bad_tile):
    with pytest.raises(ValidationError) as exc:
        embed_tile(bad_tile, "Clay")
    assert exc.value.category is ErrorCategory.VALIDATION


def test_dict_tile_is_accepted_when_well_formed():
    # A correctly shaped dict (as the runtime would pass through unchanged on a
    # coercion miss) still embeds, proving the guard only rejects bad input.
    tile = {"width": 2, "height": 2, "bands": 1, "format": "numpy", "data": [1.0, 2.0, 3.0, 4.0]}
    result = embed_tile(tile, "Clay")
    assert result.model == "Clay"
    assert len(result.vector) == result.dimension


@pytest.mark.asyncio
async def test_server_embed_tile_tool():
    from geo_foundation_models.server import GeoFoundationModelsServer

    server = GeoFoundationModelsServer()
    assert "embed_tile" in server.tool_names()
    result = await server.embed_tile(tile=_tile(), model="Prithvi-EO-2.0")
    assert result.model == "Prithvi-EO-2.0"
    assert len(result.vector) == result.dimension

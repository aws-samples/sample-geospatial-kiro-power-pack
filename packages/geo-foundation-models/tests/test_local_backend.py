"""Tests for the in-process LocalCallableBackend (real-weight, no endpoint).

Exercised without any model install: the embedding callable is injected (or
loaded by dotted path from this module), so these assert the payload contract,
the shared vector-coercion, the error normalization, provenance, and the
environment factory precedence — plus a randomized round-trip property.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geo_common.errors import ValidationError

from geo_foundation_models.embedding import embed_tile
from geo_foundation_models.local_backend import LocalCallableBackend, load_callable
from geo_foundation_models.models import ModelSpec, RasterTile
from geo_foundation_models.remote_backend import backend_from_env

_CLAY = ModelSpec(name="Clay", dimension=768)


def _tile(width=4, height=4, bands=1):
    n = width * height * bands
    return RasterTile(
        width=width, height=height, bands=bands, format="GTiff",
        data=[float(i % 5) for i in range(n)], dtype="uint16",
    )


# A module-level callable so load_callable / the env factory can import it.
def sample_embed(payload):
    """Deterministic content-dependent vector of the requested dimension."""
    dim = payload["dimension"]
    data = payload.get("data") or [0.0]
    base = sum(data) % 7
    return [float((base + i) % 11) for i in range(dim)]


def _echo(vec):
    return lambda payload: vec


# --- configuration --------------------------------------------------------

def test_requires_exactly_one_of_fn_or_path() -> None:
    with pytest.raises(ValidationError):
        LocalCallableBackend()  # neither
    with pytest.raises(ValidationError):
        LocalCallableBackend(embed_fn=_echo([1.0]), dotted_path="x:y")  # both


def test_backend_id_defaults_and_is_not_standin() -> None:
    injected = LocalCallableBackend(embed_fn=_echo([1.0]))
    assert injected.backend_id == "local-callable"
    # A guaranteed-importable stdlib callable stands in for a real model fn.
    by_path = LocalCallableBackend(dotted_path="math:sqrt")
    assert by_path.backend_id == "local:math.sqrt"
    assert injected.backend_id != "deterministic-local"


# --- payload + coercion contract ------------------------------------------

def test_embed_passes_payload_and_returns_vector() -> None:
    captured = {}

    def fn(payload):
        captured.update(payload)
        return [0.0] * 767 + [1.0]

    backend = LocalCallableBackend(embed_fn=fn)
    out = backend.embed(_tile(), _CLAY)
    assert out == [0.0] * 767 + [1.0]
    assert captured["model"] == "Clay" and captured["dimension"] == 768
    assert len(captured["data"]) == 16


@pytest.mark.parametrize(
    "returned",
    [
        [1.0, 2.0, 3.0],
        (1.0, 2.0, 3.0),
        [[1.0, 2.0, 3.0]],
        {"vector": [1.0, 2.0, 3.0]},
        {"embedding": [1.0, 2.0, 3.0]},
    ],
)
def test_return_shape_flexibility(returned) -> None:
    backend = LocalCallableBackend(embed_fn=_echo(returned))
    assert backend.embed(_tile(), ModelSpec(name="M", dimension=3)) == [1.0, 2.0, 3.0]


def test_embed_tile_records_local_provenance() -> None:
    backend = LocalCallableBackend(embed_fn=sample_embed, backend_id="clay-local")
    result = embed_tile(_tile(), "Clay", backend=backend)
    assert result.backend == "clay-local"
    assert result.dimension == 768 and len(result.vector) == 768


def test_callable_exception_is_validation_error() -> None:
    def boom(payload):
        raise RuntimeError("model not loaded")

    backend = LocalCallableBackend(embed_fn=boom)
    with pytest.raises(ValidationError) as exc:
        backend.embed(_tile(), _CLAY)
    assert exc.value.detail.get("backend") == "local-callable"


def test_wrong_dimension_from_local_is_rejected_by_embed_tile() -> None:
    backend = LocalCallableBackend(embed_fn=_echo([1.0, 2.0]))  # 2-d, Clay wants 768
    with pytest.raises(ValidationError):
        embed_tile(_tile(), "Clay", backend=backend)


# --- load_callable --------------------------------------------------------

def test_load_callable_resolves_module_attr() -> None:
    import math

    assert load_callable("math:sqrt") is math.sqrt
    # Dotted (no colon) form also works.
    assert load_callable("math.sqrt") is math.sqrt


@pytest.mark.parametrize("bad", ["", "nomodule", "no.such.module:fn", "math:missing"])
def test_load_callable_bad_path_is_validation_error(bad) -> None:
    with pytest.raises(ValidationError):
        load_callable(bad)


def test_load_callable_non_callable_target_is_validation_error() -> None:
    with pytest.raises(ValidationError):
        load_callable("math:pi")  # a value, not callable


# --- environment factory precedence ---------------------------------------

def test_backend_from_env_selects_local_callable() -> None:
    backend = backend_from_env(env={
        "GEO_FM_EMBED_LOCAL_CALLABLE": "math:sqrt",
    })
    assert isinstance(backend, LocalCallableBackend)
    assert backend.backend_id.startswith("local:")


def test_local_callable_takes_precedence_over_remote() -> None:
    backend = backend_from_env(env={
        "GEO_FM_EMBED_LOCAL_CALLABLE": "math:sqrt",
        "GEO_FM_SAGEMAKER_ENDPOINT": "ep",
        "GEO_FM_EMBED_ENDPOINT_URL": "https://x",
    })
    assert isinstance(backend, LocalCallableBackend)


# --- property: local round-trip -------------------------------------------

@settings(deadline=None)
@given(
    width=st.integers(min_value=1, max_value=12),
    height=st.integers(min_value=1, max_value=12),
    bands=st.integers(min_value=1, max_value=3),
    dim=st.integers(min_value=1, max_value=48),
)
def test_local_embed_roundtrip(width, height, bands, dim) -> None:
    """Feature: geospatial-power-pack, in-process local embedding backend.

    For any tile/dim, the local backend returns exactly the callable's vector
    and always passes a correctly-sized payload.
    """
    target = [float(i) - dim / 2 for i in range(dim)]

    def fn(payload):
        assert len(payload["data"]) == payload["width"] * payload["height"] * payload["bands"]
        return target

    backend = LocalCallableBackend(embed_fn=fn)
    out = backend.embed(_tile(width, height, bands), ModelSpec(name="M", dimension=dim))
    assert out == target

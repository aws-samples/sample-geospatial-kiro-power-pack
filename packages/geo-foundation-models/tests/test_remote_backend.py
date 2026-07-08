"""Tests for the real-weight RemoteEndpointBackend seam (hermetic).

The backend is exercised without any network or AWS access: the ``invoke``
hook fully replaces the transport, so these assert the payload/response
contract, the response-shape flexibility, the error mapping onto the
``Error_Taxonomy``, and the environment-config factory. A separate test drives
the real generic-HTTPS code path by monkeypatching ``httpx.post``.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geo_common.errors import UpstreamError, ValidationError

from geo_foundation_models.embedding import ModelRegistry, embed_tile
from geo_foundation_models.models import ModelSpec, RasterTile
from geo_foundation_models.remote_backend import (
    RemoteEndpointBackend,
    backend_from_env,
)

_CLAY = ModelSpec(name="Clay", dimension=768)


def _tile(width=4, height=4, bands=1):
    n = width * height * bands
    return RasterTile(
        width=width, height=height, bands=bands, format="GTiff",
        data=[float(i % 7) for i in range(n)], dtype="uint16",
    )


def _vec(dim=768):
    return [0.0] * (dim - 1) + [1.0]


# --- configuration --------------------------------------------------------

def test_requires_exactly_one_transport() -> None:
    with pytest.raises(ValidationError):
        RemoteEndpointBackend()  # neither
    with pytest.raises(ValidationError):
        RemoteEndpointBackend(
            endpoint_url="https://x/invocations", sagemaker_endpoint="ep"
        )  # both


def test_backend_id_defaults() -> None:
    https = RemoteEndpointBackend(endpoint_url="https://x/invocations")
    assert https.backend_id == "remote-endpoint"
    sm = RemoteEndpointBackend(sagemaker_endpoint="clay-ep")
    assert sm.backend_id == "sagemaker:clay-ep"
    custom = RemoteEndpointBackend(endpoint_url="https://x", backend_id="clay-v1.5")
    assert custom.backend_id == "clay-v1.5"
    # Crucially, a real backend is never the deterministic stand-in id.
    assert https.backend_id != "deterministic-local"


# --- payload + response contract ------------------------------------------

def test_embed_sends_expected_payload_and_returns_vector() -> None:
    captured = {}

    def fake_invoke(payload):
        captured.update(payload)
        return json.dumps({"vector": _vec(768)})

    backend = RemoteEndpointBackend(endpoint_url="https://x", invoke=fake_invoke)
    tile = _tile()
    vector = backend.embed(tile, _CLAY)

    assert vector == _vec(768)
    assert captured["model"] == "Clay"
    assert captured["dimension"] == 768
    assert captured["width"] == 4 and captured["height"] == 4 and captured["bands"] == 1
    assert len(captured["data"]) == 16  # width*height*bands
    # Enrichment fields default to None when the tile carries no space/time.
    assert captured["latlon"] is None and captured["acquired"] is None


def test_payload_carries_latlon_and_acquired_when_present() -> None:
    captured = {}
    backend = RemoteEndpointBackend(
        endpoint_url="https://x",
        invoke=lambda p: (captured.update(p), json.dumps({"vector": _vec(768)}))[1],
    )
    tile = RasterTile(
        width=4, height=4, bands=1, format="GTiff",
        data=[float(i) for i in range(16)], dtype="uint16",
        latlon=(37.77, -122.42), acquired="2024-06-14T18:30:00Z",
    )
    backend.embed(tile, _CLAY)
    assert captured["latlon"] == [37.77, -122.42]
    assert captured["acquired"] == "2024-06-14T18:30:00Z"


@pytest.mark.parametrize(
    "response",
    [
        {"vector": _vec()},
        {"embedding": _vec()},
        {"embeddings": [_vec()]},
        {"predictions": [_vec()]},
        _vec(),          # bare array
        [_vec()],        # single batched row
    ],
)
def test_response_shape_flexibility(response) -> None:
    backend = RemoteEndpointBackend(
        endpoint_url="https://x", invoke=lambda p: json.dumps(response)
    )
    assert backend.embed(_tile(), _CLAY) == _vec()


def test_embed_tile_records_remote_backend_provenance() -> None:
    backend = RemoteEndpointBackend(
        endpoint_url="https://x", backend_id="clay-v1.5",
        invoke=lambda p: json.dumps({"vector": _vec(768)}),
    )
    result = embed_tile(_tile(), "Clay", backend=backend)
    assert result.backend == "clay-v1.5"
    assert result.dimension == 768 and len(result.vector) == 768


def test_embed_tile_rejects_wrong_dimension_from_endpoint() -> None:
    # Endpoint returns a 3-d vector but Clay expects 768 -> embed_tile guards it.
    backend = RemoteEndpointBackend(
        endpoint_url="https://x", invoke=lambda p: json.dumps({"vector": [1, 2, 3]})
    )
    with pytest.raises(ValidationError):
        embed_tile(_tile(), "Clay", backend=backend)


# --- error mapping --------------------------------------------------------

@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        json.dumps({"nope": [1, 2]}),
        json.dumps({"vector": []}),
        json.dumps({"vector": ["a", "b"]}),
        json.dumps({"vector": [float("inf"), 1.0]}),
    ],
)
def test_malformed_response_is_upstream_error(bad) -> None:
    backend = RemoteEndpointBackend(endpoint_url="https://x", invoke=lambda p: bad)
    with pytest.raises(UpstreamError):
        backend.embed(_tile(), _CLAY)


def test_https_transport_path_with_monkeypatched_post(monkeypatch) -> None:
    """Drive the real _call_https path (header auth + status handling)."""
    import httpx

    seen = {}

    class _Resp:
        status_code = 200
        text = json.dumps({"vector": _vec(768)})

    def fake_post(url, json=None, headers=None, timeout=None):
        seen["url"] = url
        seen["headers"] = headers
        return _Resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    backend = RemoteEndpointBackend(
        endpoint_url="https://host/invocations", api_key="secret",
    )
    vector = backend.embed(_tile(), _CLAY)
    assert vector == _vec(768)
    assert seen["url"] == "https://host/invocations"
    assert seen["headers"]["Authorization"] == "Bearer secret"


def test_https_transport_maps_http_error(monkeypatch) -> None:
    import httpx

    class _Resp:
        status_code = 502
        text = ""

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp())
    backend = RemoteEndpointBackend(endpoint_url="https://host/invocations")
    with pytest.raises(UpstreamError):
        backend.embed(_tile(), _CLAY)


# --- environment factory --------------------------------------------------

def test_backend_from_env_none_when_unset() -> None:
    assert backend_from_env(env={}) is None


def test_backend_from_env_https() -> None:
    backend = backend_from_env(env={
        "GEO_FM_EMBED_ENDPOINT_URL": "https://host/invocations",
        "GEO_FM_EMBED_API_KEY": "k",
        "GEO_FM_EMBED_BACKEND_ID": "clay-v1.5",
    })
    assert isinstance(backend, RemoteEndpointBackend)
    assert backend.endpoint_url == "https://host/invocations"
    assert backend.api_key == "k"
    assert backend.backend_id == "clay-v1.5"


def test_backend_from_env_sagemaker_takes_precedence() -> None:
    backend = backend_from_env(env={
        "GEO_FM_SAGEMAKER_ENDPOINT": "clay-ep",
        "GEO_FM_EMBED_ENDPOINT_URL": "https://host/invocations",
        "GEO_FM_EMBED_REGION": "us-west-2",
    })
    assert backend.sagemaker_endpoint == "clay-ep"
    assert backend.endpoint_url is None
    assert backend.region == "us-west-2"


# --- property: parsing/payload invariants ---------------------------------

@settings(deadline=None)
@given(
    width=st.integers(min_value=1, max_value=16),
    height=st.integers(min_value=1, max_value=16),
    bands=st.integers(min_value=1, max_value=3),
    dim=st.integers(min_value=1, max_value=64),
)
def test_embed_roundtrips_endpoint_vector(width, height, bands, dim) -> None:
    """For any tile/dim, embed returns exactly the endpoint's vector, once.

    Feature: geospatial-power-pack, real-weight remote embedding backend.
    """
    calls = {"n": 0}
    target = [float(i) - dim / 2 for i in range(dim)]

    def fake_invoke(payload):
        calls["n"] += 1
        # The payload always carries a correctly-sized flat pixel buffer.
        assert len(payload["data"]) == payload["width"] * payload["height"] * payload["bands"]
        return json.dumps({"vector": target})

    backend = RemoteEndpointBackend(endpoint_url="https://x", invoke=fake_invoke)
    spec = ModelSpec(name="M", dimension=dim)
    out = backend.embed(_tile(width, height, bands), spec)
    assert out == target
    assert calls["n"] == 1

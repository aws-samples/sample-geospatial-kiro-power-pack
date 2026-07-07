"""Live self-test for a configured real-weight embedding backend (opt-in).

This is the one-command "is my model wired correctly?" check. It runs against
whatever backend the environment configures — a local callable
(``GEO_FM_EMBED_LOCAL_CALLABLE``), a generic HTTPS endpoint
(``GEO_FM_EMBED_ENDPOINT_URL``), or a SageMaker endpoint
(``GEO_FM_SAGEMAKER_ENDPOINT``) — and asserts the properties that make an
embedding backend usable for change detection, without any network of its own
(it embeds small synthetic tiles).

Run it (skipped by default)::

    RUN_LIVE_EMBED=1 \
    GEO_FM_EMBED_LOCAL_CALLABLE=template_embed:embed \
    GEO_FM_EXTRA_MODELS=MyModel:512 GEO_FM_LIVE_MODEL=MyModel \
    PYTHONPATH=examples/custom-embedding-backend \
    .venv/bin/python -m pytest packages/geo-foundation-models/tests/test_live_backend.py -q

For real Clay v1.5, point the callable at examples/clay-local-backend/clay_embed.py,
set GEO_FM_LIVE_MODEL=Clay-v1.5 and GEO_FM_LIVE_BANDS to the sensor's band count.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_EMBED") != "1",
    reason="set RUN_LIVE_EMBED=1 (and configure a backend) to run the live self-test",
)


def _tile(bands: int, *, fill):
    from geo_foundation_models.models import RasterTile

    n = 16 * 16 * bands
    return RasterTile(
        width=16, height=16, bands=bands, format="GTiff",
        data=[float(fill(i)) for i in range(n)], dtype="uint16",
    )


def test_configured_backend_is_calibrated_deterministic_and_content_sensitive():
    """The configured real backend must be calibrated, stable, and discriminating."""
    from geo_foundation_models.change import detect_change
    from geo_foundation_models.embedding import embed_tile, registry_from_env
    from geo_foundation_models.remote_backend import backend_from_env

    backend = backend_from_env()
    if backend is None:
        pytest.skip(
            "no real backend configured — set GEO_FM_EMBED_LOCAL_CALLABLE, "
            "GEO_FM_EMBED_ENDPOINT_URL, or GEO_FM_SAGEMAKER_ENDPOINT"
        )

    registry = registry_from_env()
    model = os.environ.get("GEO_FM_LIVE_MODEL", "Clay-v1.5")
    bands = int(os.environ.get("GEO_FM_LIVE_BANDS", "4"))
    spec = registry.get(model)

    tile_a = _tile(bands, fill=lambda i: i % 97)
    tile_b = _tile(bands, fill=lambda i: (i * 7 + 13) % 97)

    r_a = embed_tile(tile_a, model, backend=backend, registry=registry)
    r_a2 = embed_tile(tile_a, model, backend=backend, registry=registry)
    r_b = embed_tile(tile_b, model, backend=backend, registry=registry)

    # Calibrated: a real backend, not the honest stand-in.
    assert r_a.backend != "deterministic-local", r_a.backend
    # Declared dimension matches the backend's actual output (embed_tile guards it).
    assert r_a.dimension == spec.dimension == len(r_a.vector)
    # Stable: identical input -> ~identical embedding -> change ≈ 0. A near-zero
    # tolerance (not exact) accommodates the ~1e-13 run-to-run float noise real
    # torch models exhibit; the stand-in/template are bit-exact and pass too.
    self_change = detect_change(r_a.vector, r_a2.vector)
    assert self_change < 1e-6, self_change
    # Content-sensitive: different input -> a clearly non-zero, in-range change.
    change = detect_change(r_a.vector, r_b.vector)
    assert 1e-6 < change <= 1.0, change

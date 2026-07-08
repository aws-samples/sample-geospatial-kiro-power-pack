"""Honesty / foot-gun guards for the deterministic stand-in (Batch B #3/#4/#6).

These pin the behaviors that keep a stand-in embedding from being mistaken for
real inference:

* structure-only tiles (no pixels) are flagged, and two of the same shape yield
  the IDENTICAL embedding -> detect_change == 0.0 (the silent "no change"
  foot-gun);
* with the deterministic backend, two *different* tiles score ~0.5 (the space
  has no semantic structure, so it cannot rank change severity);
* the published-embedding periods are queryable so an out-of-range before/after
  is known to be unusable up front.
"""

from __future__ import annotations

import random

import pytest

from geo_foundation_models.change import detect_change
from geo_foundation_models.clay_embeddings import AVAILABLE_PERIODS, available_periods
from geo_foundation_models.embedding import embed_tile
from geo_foundation_models.models import RasterTile


def _tile(data=None, width=8, height=8, bands=1):
    return RasterTile(width=width, height=height, bands=bands, format="GTiff", data=data)


def test_structure_only_tile_is_flagged() -> None:
    result = embed_tile(_tile(data=None), model="Clay")
    assert result.structure_only is True


def test_tile_with_pixels_is_not_structure_only() -> None:
    result = embed_tile(_tile(data=[float(i % 7) for i in range(64)]), model="Clay")
    assert result.structure_only is False


def test_two_structure_only_tiles_collide_and_read_as_zero_change() -> None:
    """The foot-gun: no pixels -> identical embedding -> detect_change 0.0."""
    a = embed_tile(_tile(data=None), model="Clay")
    b = embed_tile(_tile(data=None), model="Clay")
    assert a.structure_only and b.structure_only
    assert a.vector == b.vector
    assert detect_change(a.vector, b.vector) == 0.0


@pytest.mark.property
def test_deterministic_backend_cannot_rank_severity() -> None:
    """Different tiles cluster near 0.5 — the stand-in cannot rank change."""
    rng = random.Random(7)
    scores = []
    for _ in range(50):
        da = [float(rng.randint(0, 255)) for _ in range(64)]
        db = [float(rng.randint(0, 255)) for _ in range(64)]
        ea = embed_tile(_tile(data=da), model="Clay")
        eb = embed_tile(_tile(data=db), model="Clay")
        scores.append(detect_change(ea.vector, eb.vector))
    # High-dimensional random unit vectors are near-orthogonal -> change ~0.5.
    assert all(0.30 < s < 0.70 for s in scores)
    assert abs(sum(scores) / len(scores) - 0.5) < 0.05


def test_available_periods_matches_dataset_partitions() -> None:
    periods = available_periods()
    assert periods == [f"{y:04d}-{m:02d}" for (y, m) in AVAILABLE_PERIODS]
    # Documents the real-embedding coverage gap (no 2026, etc.).
    assert "2026-06" not in periods

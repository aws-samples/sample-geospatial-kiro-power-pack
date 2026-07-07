"""Tests for the read-then-embed bridges: embed_asset / detect_change_from_assets.

They build deterministic in-memory COGs with ``geo_common.testing`` and read
them through a recording byte-range reader, so no network is needed. Coverage:
a full-asset and a windowed embed, band-range validation, non-overlapping
window rejection, and change-from-assets provenance/caveats.
"""

from __future__ import annotations

import random

import pytest

from geo_common.errors import ValidationError
from geo_common.testing import RecordingByteRangeReader, build_cog

from geo_foundation_models.asset_embedding import detect_change_from_assets, embed_asset

_GT = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)  # 4x4 asset covering x/y in [0, 4]


def _cog(values, dtype="uint16", **kw):
    return build_cog(
        width=4, height=4, tile_width=2, tile_height=2,
        bands=[values], dtype=dtype, geotransform=_GT, **kw,
    )


async def test_embed_asset_reads_and_embeds_full_asset() -> None:
    values = [i for i in range(16)]
    result = await embed_asset(
        raster_href="s3://amzn-s3-demo-bucket/scene.tif",
        model="Clay",
        reader=RecordingByteRangeReader(_cog(values)),
    )
    assert result.model == "Clay"
    assert len(result.vector) == result.dimension
    assert result.backend == "deterministic-local"
    # Real pixels were read, so the embedding is not structure-only.
    assert result.structure_only is False


async def test_embed_asset_windowed_read() -> None:
    values = [i for i in range(16)]
    result = await embed_asset(
        raster_href="s3://amzn-s3-demo-bucket/scene.tif",
        model="Clay",
        window_bbox=(0.0, 2.0, 2.0, 4.0),  # top-left 2x2
        reader=RecordingByteRangeReader(_cog(values)),
    )
    assert len(result.vector) == result.dimension


async def test_embed_asset_rejects_out_of_range_band() -> None:
    with pytest.raises(ValidationError) as exc:
        await embed_asset(
            raster_href="s3://amzn-s3-demo-bucket/scene.tif",
            model="Clay",
            bands=[2],  # single-band asset
            reader=RecordingByteRangeReader(_cog(list(range(16)))),
        )
    assert exc.value.detail.get("parameter") == "bands"


async def test_embed_asset_rejects_non_overlapping_window() -> None:
    with pytest.raises(ValidationError) as exc:
        await embed_asset(
            raster_href="s3://amzn-s3-demo-bucket/scene.tif",
            model="Clay",
            window_bbox=(100.0, 100.0, 101.0, 101.0),
            reader=RecordingByteRangeReader(_cog(list(range(16)))),
        )
    assert exc.value.detail.get("parameter") == "window_bbox"


async def test_detect_change_from_identical_assets_is_zero_with_caveat() -> None:
    values = [i for i in range(16)]
    result = await detect_change_from_assets(
        raster_href_a="s3://amzn-s3-demo-bucket/a.tif",
        raster_href_b="s3://amzn-s3-demo-bucket/b.tif",
        model="Clay",
        readers={
            "a": RecordingByteRangeReader(_cog(values)),
            "b": RecordingByteRangeReader(_cog(values)),
        },
    )
    assert result.change == 0.0
    assert result.structure_only is False
    # Under the stand-in backend the score is not calibrated -> caveat present.
    assert result.caveat is not None and "calibrated" in result.caveat.lower()


async def test_detect_change_from_different_assets_scores_midrange() -> None:
    rng = random.Random(3)
    a = [rng.randint(0, 255) for _ in range(16)]
    b = [rng.randint(0, 255) for _ in range(16)]
    result = await detect_change_from_assets(
        raster_href_a="s3://amzn-s3-demo-bucket/a.tif",
        raster_href_b="s3://amzn-s3-demo-bucket/b.tif",
        model="Clay",
        readers={
            "a": RecordingByteRangeReader(_cog(a)),
            "b": RecordingByteRangeReader(_cog(b)),
        },
    )
    # Deterministic stand-in: differing tiles are near-orthogonal -> ~0.5.
    assert 0.30 < result.change < 0.70
    assert result.caveat is not None

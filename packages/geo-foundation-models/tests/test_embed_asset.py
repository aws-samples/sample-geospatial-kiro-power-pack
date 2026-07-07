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

from geo_foundation_models.asset_embedding import (
    detect_change_from_assets,
    embed_asset,
    embed_assets,
)

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


def _cog_grid(width, height, values, dtype="uint16"):
    return build_cog(
        width=width, height=height, tile_width=2, tile_height=2,
        bands=[values], dtype=dtype, geotransform=_GT,
    )


async def test_embed_assets_stacks_separate_single_band_cogs() -> None:
    # Three separate single-band 4x4 COGs -> one 3-band tile embedded together.
    b1 = _cog(list(range(16)))
    b2 = _cog([v + 100 for v in range(16)])
    b3 = _cog([v * 2 for v in range(16)])
    result = await embed_assets(
        assets=["s3://b/B1.tif", "s3://b/B2.tif", "s3://b/B3.tif"],
        model="Clay",
        readers=[
            RecordingByteRangeReader(b1),
            RecordingByteRangeReader(b2),
            RecordingByteRangeReader(b3),
        ],
    )
    assert result.model == "Clay"
    assert len(result.vector) == result.dimension
    assert result.structure_only is False


async def test_embed_asset_forwards_latlon_and_acquired_to_backend() -> None:
    from geo_foundation_models.embedding import EmbeddingBackend

    seen = {}

    class _CapturingBackend(EmbeddingBackend):
        backend_id = "test-capture"

        def embed(self, tile, spec):
            seen["latlon"] = tile.latlon
            seen["acquired"] = tile.acquired
            return [0.0] * spec.dimension

    await embed_asset(
        raster_href="s3://b/scene.tif", model="Clay",
        latlon=(37.77, -122.42), acquired="2024-06-14T18:30:00Z",
        backend=_CapturingBackend(),
        reader=RecordingByteRangeReader(_cog(list(range(16)))),
    )
    assert seen["latlon"] == (37.77, -122.42)
    assert seen["acquired"] == "2024-06-14T18:30:00Z"


async def test_embed_assets_windowed() -> None:
    result = await embed_assets(
        assets=["s3://b/B1.tif", "s3://b/B2.tif"],
        model="Clay",
        window_bbox=(0.0, 2.0, 2.0, 4.0),  # top-left 2x2
        readers=[
            RecordingByteRangeReader(_cog(list(range(16)))),
            RecordingByteRangeReader(_cog([v + 5 for v in range(16)])),
        ],
    )
    assert len(result.vector) == result.dimension


async def test_embed_assets_rejects_misaligned_grids() -> None:
    with pytest.raises(ValidationError) as exc:
        await embed_assets(
            assets=["s3://b/B1.tif", "s3://b/B2.tif"],
            model="Clay",
            readers=[
                RecordingByteRangeReader(_cog_grid(4, 4, list(range(16)))),
                RecordingByteRangeReader(_cog_grid(6, 6, list(range(36)))),
            ],
        )
    assert exc.value.detail.get("parameter") == "assets"


async def test_embed_assets_rejects_empty_assets() -> None:
    with pytest.raises(ValidationError) as exc:
        await embed_assets(assets=[], model="Clay", readers=[])
    assert exc.value.detail.get("parameter") == "assets"


async def test_embed_assets_rejects_non_overlapping_window() -> None:
    with pytest.raises(ValidationError) as exc:
        await embed_assets(
            assets=["s3://b/B1.tif"],
            model="Clay",
            window_bbox=(100.0, 100.0, 101.0, 101.0),
            readers=[RecordingByteRangeReader(_cog(list(range(16))))],
        )
    assert exc.value.detail.get("parameter") == "window_bbox"


async def test_embed_assets_matches_multiband_embed_asset() -> None:
    """Stacking 2 single-band COGs must equal reading a 2-band COG (same pixels)."""
    vals1 = list(range(16))
    vals2 = [v + 50 for v in range(16)]
    multiband = build_cog(
        width=4, height=4, tile_width=2, tile_height=2,
        bands=[vals1, vals2], dtype="uint16", geotransform=_GT,
    )
    from_assets = await embed_assets(
        assets=["s3://b/B1.tif", "s3://b/B2.tif"],
        model="Clay",
        readers=[
            RecordingByteRangeReader(_cog(vals1)),
            RecordingByteRangeReader(_cog(vals2)),
        ],
    )
    from_multi = await embed_asset(
        raster_href="s3://b/multi.tif", model="Clay", bands=[1, 2],
        reader=RecordingByteRangeReader(multiband),
    )
    assert from_assets.vector == from_multi.vector


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


async def test_detect_change_from_assets_per_band_lists_identical_is_zero() -> None:
    """Per-band asset-list mode: identical dates score exactly 0.0, one call."""
    b1 = [i for i in range(16)]
    b2 = [i + 100 for i in range(16)]
    result = await detect_change_from_assets(
        model="Clay",
        assets_a=["s3://b/a_B1.tif", "s3://b/a_B2.tif"],
        assets_b=["s3://b/b_B1.tif", "s3://b/b_B2.tif"],
        readers_a=[RecordingByteRangeReader(_cog(b1)), RecordingByteRangeReader(_cog(b2))],
        readers_b=[RecordingByteRangeReader(_cog(b1)), RecordingByteRangeReader(_cog(b2))],
    )
    assert result.change == 0.0
    assert result.structure_only is False
    assert result.caveat is not None  # stand-in backend -> not calibrated


async def test_detect_change_from_assets_per_band_lists_matches_single_href() -> None:
    """Per-band lists give the same score as the equivalent multi-band COGs."""
    a1, a2 = list(range(16)), [v + 10 for v in range(16)]
    b1, b2 = [v + 3 for v in range(16)], [v + 40 for v in range(16)]
    multi_a = build_cog(width=4, height=4, tile_width=2, tile_height=2,
                        bands=[a1, a2], dtype="uint16", geotransform=_GT)
    multi_b = build_cog(width=4, height=4, tile_width=2, tile_height=2,
                        bands=[b1, b2], dtype="uint16", geotransform=_GT)
    from_lists = await detect_change_from_assets(
        model="Clay",
        assets_a=["s3://b/a1.tif", "s3://b/a2.tif"],
        assets_b=["s3://b/b1.tif", "s3://b/b2.tif"],
        readers_a=[RecordingByteRangeReader(_cog(a1)), RecordingByteRangeReader(_cog(a2))],
        readers_b=[RecordingByteRangeReader(_cog(b1)), RecordingByteRangeReader(_cog(b2))],
    )
    from_single = await detect_change_from_assets(
        model="Clay", raster_href_a="s3://b/ma.tif", raster_href_b="s3://b/mb.tif",
        bands=[1, 2],
        readers={"a": RecordingByteRangeReader(multi_a), "b": RecordingByteRangeReader(multi_b)},
    )
    assert from_lists.change == from_single.change


async def test_detect_change_from_assets_rejects_mixed_modes() -> None:
    with pytest.raises(ValidationError):
        await detect_change_from_assets(
            model="Clay", raster_href_a="s3://b/a.tif",
            assets_b=["s3://b/b1.tif"],
        )


async def test_detect_change_from_assets_requires_two_dates() -> None:
    with pytest.raises(ValidationError):
        await detect_change_from_assets(model="Clay")
    with pytest.raises(ValidationError) as exc:
        await detect_change_from_assets(model="Clay", assets_a=["s3://b/a1.tif"])
    assert exc.value.detail.get("parameter") == "assets_b"


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

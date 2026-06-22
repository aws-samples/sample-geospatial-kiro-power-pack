"""Tests for the open Clay v1.5 embedding lookup (``lookup_embeddings``).

The selection / partition-pruning / bounding logic is exercised with an
injected mock reader so no network or ``pyarrow`` S3 I/O is required. A single
opt-in live test (``RUN_LIVE_CLAY=1``) reads a couple of real records from the
open dataset to confirm the default S3 reader path end to end.
"""

from __future__ import annotations

import os

import pytest

from geo_common.errors import ErrorCategory, ValidationError

from geo_foundation_models.clay_embeddings import (
    AVAILABLE_PERIODS,
    CLAY_V15_DIMENSION,
    CLAY_V15_MODEL_NAME,
    gzd_partitions,
    lookup_embeddings,
    periods_in_range,
)
from geo_foundation_models.models import EmbeddingRecord

# A small bbox in San Francisco (UTM zone 10, band S -> GZD "10S").
SF_BBOX = (-122.45, 37.74, -122.39, 37.80)


def _row(i: int) -> dict:
    return {
        "chips_id": "chip-%d" % i,
        "cell_id": "cell-%d" % i,
        "stac_item_id": "S2_item_%d" % i,
        "collection": "sentinel-2-l2a",
        "datetime": "2024-06-15T18:00:00Z",
        "embedding": [0.01 * i] * CLAY_V15_DIMENSION,
        "bbox": {"xmin": -122.44, "ymin": 37.75, "xmax": -122.43, "ymax": 37.76},
    }


def _mock_reader(rows, *, calls=None):
    def reader(*, product, bbox, periods, gzds, limit):
        if calls is not None:
            calls.append({"product": product, "periods": list(periods), "gzds": list(gzds), "limit": limit})
        return iter(rows)

    return reader


# --- partition helpers ------------------------------------------------------


def test_gzd_partitions_cover_bbox():
    gzds = gzd_partitions(SF_BBOX)
    assert "10S" in gzds


def test_gzd_partitions_span_multiple_zones_and_bands():
    # A bbox crossing the zone 10/11 boundary (-120 lon) and two lat bands.
    gzds = gzd_partitions((-120.5, 39.5, -119.5, 40.5))
    assert any(g.startswith("10") for g in gzds)
    assert any(g.startswith("11") for g in gzds)


def test_periods_in_range_filters_to_available():
    assert periods_in_range("2024-01-01", "2024-12-31") == [(2024, 6)]
    assert periods_in_range(None, None) == list(AVAILABLE_PERIODS)
    # A range with no published data yields nothing.
    assert periods_in_range("2023-01-01", "2023-12-31") == []


def test_periods_in_range_rejects_start_after_end():
    with pytest.raises(ValidationError) as exc:
        periods_in_range("2025-06-01", "2024-06-01")
    assert exc.value.category is ErrorCategory.VALIDATION


# --- lookup_embeddings: records + bounding ---------------------------------


def test_lookup_builds_clay_records():
    records = lookup_embeddings(bbox=SF_BBOX, reader=_mock_reader([_row(1), _row(2)]))
    assert all(isinstance(r, EmbeddingRecord) for r in records)
    assert [r.id for r in records] == ["chip-1", "chip-2"]
    r = records[0]
    assert r.model == CLAY_V15_MODEL_NAME
    assert r.dimension == CLAY_V15_DIMENSION
    assert len(r.vector) == CLAY_V15_DIMENSION
    assert r.metadata.source_asset == "S2_item_1"
    assert r.metadata.bbox == (-122.44, 37.75, -122.43, 37.76)


def test_lookup_enforces_limit():
    rows = [_row(i) for i in range(100)]
    records = lookup_embeddings(bbox=SF_BBOX, limit=5, reader=_mock_reader(rows))
    assert len(records) == 5


def test_lookup_scene_product_passes_gzds_to_reader():
    calls: list = []
    lookup_embeddings(
        bbox=SF_BBOX, product="scene", reader=_mock_reader([_row(1)], calls=calls)
    )
    assert calls and "10S" in calls[0]["gzds"]


def test_lookup_aggregated_product_passes_no_gzds():
    calls: list = []
    lookup_embeddings(
        bbox=SF_BBOX, product="aggregated", reader=_mock_reader([_row(1)], calls=calls)
    )
    assert calls and calls[0]["gzds"] == []


def test_lookup_out_of_range_returns_empty_without_calling_reader():
    calls: list = []
    records = lookup_embeddings(
        bbox=SF_BBOX,
        start="2023-01-01",
        end="2023-12-31",
        reader=_mock_reader([_row(1)], calls=calls),
    )
    assert records == []
    assert calls == []  # no period in range -> reader never invoked


# --- lookup_embeddings: validation -----------------------------------------


def test_lookup_rejects_unknown_product():
    with pytest.raises(ValidationError) as exc:
        lookup_embeddings(bbox=SF_BBOX, product="bogus", reader=_mock_reader([]))
    assert exc.value.category is ErrorCategory.VALIDATION


def test_lookup_rejects_nonpositive_limit():
    with pytest.raises(ValidationError):
        lookup_embeddings(bbox=SF_BBOX, limit=0, reader=_mock_reader([]))


@pytest.mark.parametrize(
    "bad_bbox",
    [
        (-122.45, 37.74, -122.39),          # too few ordinates
        (200.0, 37.0, 201.0, 38.0),         # longitude out of range
        (-122.45, 37.80, -122.39, 37.74),   # inverted latitude
    ],
)
def test_lookup_rejects_malformed_bbox(bad_bbox):
    with pytest.raises(ValidationError) as exc:
        lookup_embeddings(bbox=bad_bbox, reader=_mock_reader([]))
    assert exc.value.category is ErrorCategory.VALIDATION


def test_lookup_rejects_oversized_bbox():
    # A 20x20 degree box is far larger than the 2,500 km² default cap.
    with pytest.raises(ValidationError) as exc:
        lookup_embeddings(bbox=(-10.0, 0.0, 10.0, 20.0), reader=_mock_reader([]))
    assert exc.value.category is ErrorCategory.VALIDATION
    assert exc.value.detail["parameter"] == "bbox"


# --- server tool ------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_registers_lookup_embeddings_tool():
    from geo_foundation_models.server import GeoFoundationModelsServer

    server = GeoFoundationModelsServer(embedding_reader=_mock_reader([_row(1), _row(2)]))
    assert "lookup_embeddings" in server.tool_names()
    records = await server.lookup_embeddings(bbox=SF_BBOX, limit=10)
    assert len(records) == 2
    assert records[0].model == CLAY_V15_MODEL_NAME


# --- opt-in live smoke (skipped unless RUN_LIVE_CLAY=1) --------------------


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_CLAY") != "1",
    reason="set RUN_LIVE_CLAY=1 to read real records from the open S3 dataset",
)
def test_live_lookup_returns_real_records():
    # Restrict to a single published period to minimize the data scanned.
    records = lookup_embeddings(
        bbox=SF_BBOX,
        product="scene",
        start="2024-06-01",
        end="2024-06-30",
        limit=2,
    )
    assert 1 <= len(records) <= 2
    assert all(r.dimension == CLAY_V15_DIMENSION for r in records)
    assert all(len(r.vector) == CLAY_V15_DIMENSION for r in records)

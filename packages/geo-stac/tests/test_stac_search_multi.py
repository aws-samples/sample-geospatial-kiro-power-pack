"""Unit tests for federated ``stac_search_multi`` (geo-stac).

Drives the multi-source search through a fully wired :class:`GeoStacServer`
whose shared :class:`HttpClient` is backed by an ``httpx.MockTransport`` that
routes each STAC endpoint host to a canned response - including a failing
source - so the merge, de-duplication, and partial/provenance semantics are
exercised without real network I/O.
"""

from __future__ import annotations

import httpx
import pytest

from geo_common.errors import ValidationError
from geo_common.http import HttpClient
from geo_common.retry import RetryPolicy

from geo_stac.search import KNOWN_STAC_ENDPOINTS, StacSearchResult
from geo_stac.server import GeoStacServer

VALID_BBOX = (-122.6, 37.6, -122.3, 37.9)
VALID_RANGE = ("2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z")


def _feature(item_id: str) -> dict:
    return {
        "id": item_id,
        "bbox": [-122.6, 37.6, -122.3, 37.9],
        "properties": {"datetime": "2024-01-15T00:00:00Z"},
        "assets": {"red": {"href": "https://example.com/%s_red.tif" % item_id}},
    }


def _host_of(url: str) -> str:
    return httpx.URL(url).host


def _server(handler):
    client = HttpClient(RetryPolicy(max_attempts=1), transport=httpx.MockTransport(handler))
    return GeoStacServer(http=client)


async def test_multi_source_merges_dedupes_and_reports_partial():
    es_host = _host_of(KNOWN_STAC_ENDPOINTS["earth-search"])
    pc_host = _host_of(KNOWN_STAC_ENDPOINTS["planetary-computer"])
    cmr_host = _host_of(KNOWN_STAC_ENDPOINTS["cmr-stac"])

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == es_host:
            return httpx.Response(200, json={"features": [_feature("A"), _feature("B")]})
        if host == pc_host:
            # Returns B (a duplicate of Earth Search) + a new C.
            return httpx.Response(200, json={"features": [_feature("B"), _feature("C")]})
        if host == cmr_host:
            return httpx.Response(503)  # this source is unavailable
        # Copernicus / USGS: empty result sets.
        return httpx.Response(200, json={"features": []})

    server = _server(handler)
    try:
        result = await server.stac_search_multi(bbox=VALID_BBOX, datetime_range=VALID_RANGE)
    finally:
        await server.aclose()

    assert isinstance(result, StacSearchResult)
    # A, B, C — B de-duplicated across Earth Search + Planetary Computer.
    assert sorted(item.id for item in result.items) == ["A", "B", "C"]
    # A source failed -> partial, and provenance records every source's outcome.
    assert result.partial is True
    by_name = {s.name: s for s in result.sources}
    # Round-robin: Earth Search places A first (round 1), Planetary Computer
    # places B; in round 2 Earth Search's B is a duplicate (skipped) and
    # Planetary Computer places C. So ES contributes 1 (A) and PC contributes
    # 2 (B, C) — the dedup credit goes to whichever source places the item.
    assert by_name["earth-search"].status == "ok" and by_name["earth-search"].count == 1
    assert by_name["planetary-computer"].status == "ok" and by_name["planetary-computer"].count == 2
    assert by_name["cmr-stac"].status == "error"
    assert by_name["cmr-stac"].category in {"upstream", "network", "rate-limit"}


async def test_multi_source_round_robin_represents_each_source_under_small_limit():
    """Round-robin: under a small limit, each responding catalog is represented
    rather than the first catalog (in source order) filling the whole quota."""
    es_host = _host_of(KNOWN_STAC_ENDPOINTS["earth-search"])
    pc_host = _host_of(KNOWN_STAC_ENDPOINTS["planetary-computer"])

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == es_host:
            return httpx.Response(
                200, json={"features": [_feature("es-%d" % i) for i in range(3)]}
            )
        if host == pc_host:
            return httpx.Response(
                200, json={"features": [_feature("pc-%d" % i) for i in range(3)]}
            )
        return httpx.Response(200, json={"features": []})

    server = _server(handler)
    try:
        result = await server.stac_search_multi(
            bbox=VALID_BBOX, datetime_range=VALID_RANGE, limit=4
        )
    finally:
        await server.aclose()

    ids = [item.id for item in result.items]
    assert len(ids) == 4
    # Both catalogs contribute 2 each (interleaved), not 3 ES + 1 PC.
    assert sum(i.startswith("es-") for i in ids) == 2
    assert sum(i.startswith("pc-") for i in ids) == 2
    # Interleaved order: es, pc, es, pc.
    assert ids == ["es-0", "pc-0", "es-1", "pc-1"]
    by_name = {s.name: s for s in result.sources}
    assert by_name["earth-search"].count == 2
    assert by_name["planetary-computer"].count == 2


async def test_scene_dedupe_collapses_same_granule_across_catalogs():
    """Default scene mode: one physical granule published by two catalogs under
    different ids (same acquisition instant + MGRS tile) collapses to one item;
    ``dedupe="id"`` keeps both copies."""
    es_host = _host_of(KNOWN_STAC_ENDPOINTS["earth-search"])
    pc_host = _host_of(KNOWN_STAC_ENDPOINTS["planetary-computer"])

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == es_host:
            return httpx.Response(200, json={"features": [
                {"id": "S2B_10SEG_20240129_1_L2A", "bbox": [-122.6, 37.6, -122.3, 37.9],
                 "properties": {"datetime": "2024-01-29T18:56:29Z", "s2:mgrs_tile": "10SEG"},
                 "assets": {}},
            ]})
        if host == pc_host:
            # Same granule, different id scheme + sub-second datetime formatting.
            return httpx.Response(200, json={"features": [
                {"id": "S2B_MSIL2A_20240129T185629_R113_T10SEG_x",
                 "bbox": [-122.6, 37.6, -122.3, 37.9],
                 "properties": {"datetime": "2024-01-29T18:56:29.024Z", "s2:mgrs_tile": "10SEG"},
                 "assets": {}},
            ]})
        return httpx.Response(200, json={"features": []})

    server = _server(handler)
    try:
        scene = await server.stac_search_multi(bbox=VALID_BBOX, datetime_range=VALID_RANGE)
        by_id = await server.stac_search_multi(
            bbox=VALID_BBOX, datetime_range=VALID_RANGE, dedupe="id"
        )
    finally:
        await server.aclose()

    assert len(scene.items) == 1   # same physical scene collapsed
    assert len(by_id.items) == 2   # both catalog copies kept


async def test_scene_dedupe_keeps_distinct_tiles_from_same_overpass():
    """Two adjacent tiles from one overpass share a datetime but are distinct
    scenes - scene dedup must keep both (the over-merge trap)."""
    es_host = _host_of(KNOWN_STAC_ENDPOINTS["earth-search"])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == es_host:
            dt = "2024-01-29T18:56:29Z"
            return httpx.Response(200, json={"features": [
                {"id": "S2B_10SEG_20240129_1_L2A", "bbox": [-122.6, 37.6, -122.3, 37.9],
                 "properties": {"datetime": dt, "s2:mgrs_tile": "10SEG"}, "assets": {}},
                {"id": "S2B_10SEH_20240129_0_L2A", "bbox": [-122.6, 37.6, -122.3, 37.9],
                 "properties": {"datetime": dt, "s2:mgrs_tile": "10SEH"}, "assets": {}},
            ]})
        return httpx.Response(200, json={"features": []})

    server = _server(handler)
    try:
        result = await server.stac_search_multi(bbox=VALID_BBOX, datetime_range=VALID_RANGE)
    finally:
        await server.aclose()
    assert {i.id for i in result.items} == {
        "S2B_10SEG_20240129_1_L2A", "S2B_10SEH_20240129_0_L2A"
    }


async def test_scene_dedupe_extracts_tile_from_id_without_tile_property():
    """With no tile property, the tile is recovered from the id, so cross-catalog
    granule collapse still works."""
    es_host = _host_of(KNOWN_STAC_ENDPOINTS["earth-search"])
    pc_host = _host_of(KNOWN_STAC_ENDPOINTS["planetary-computer"])

    def handler(request: httpx.Request) -> httpx.Response:
        dt = "2024-01-29T18:56:29Z"
        if request.url.host == es_host:
            return httpx.Response(200, json={"features": [
                {"id": "S2B_10SEG_20240129_1_L2A", "bbox": [-122.6, 37.6, -122.3, 37.9],
                 "properties": {"datetime": dt}, "assets": {}}
            ]})
        if request.url.host == pc_host:
            return httpx.Response(200, json={"features": [
                {"id": "S2B_MSIL2A_20240129T185629_R113_T10SEG_x",
                 "bbox": [-122.6, 37.6, -122.3, 37.9],
                 "properties": {"datetime": dt}, "assets": {}}
            ]})
        return httpx.Response(200, json={"features": []})

    server = _server(handler)
    try:
        result = await server.stac_search_multi(bbox=VALID_BBOX, datetime_range=VALID_RANGE)
    finally:
        await server.aclose()
    assert len(result.items) == 1  # tile 10SEG recovered from both ids


async def test_multi_source_unknown_dedupe_is_validation_error():
    server = _server(lambda r: httpx.Response(200, json={"features": []}))
    try:
        with pytest.raises(ValidationError) as exc_info:
            await server.stac_search_multi(
                bbox=VALID_BBOX, datetime_range=VALID_RANGE, dedupe="bogus"
            )
    finally:
        await server.aclose()
    assert exc_info.value.detail.get("parameter") == "dedupe"


async def test_multi_source_not_partial_when_all_ok():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"features": []})

    server = _server(handler)
    try:
        result = await server.stac_search_multi(bbox=VALID_BBOX, datetime_range=VALID_RANGE)
    finally:
        await server.aclose()
    assert result.partial is False
    assert all(s.status == "ok" for s in result.sources)
    assert len(result.sources) == len(KNOWN_STAC_ENDPOINTS)


async def test_multi_source_respects_limit_across_sources():
    def handler(request: httpx.Request) -> httpx.Response:
        # Every source returns 3 distinct items (host-prefixed ids).
        host = request.url.host
        return httpx.Response(
            200, json={"features": [_feature("%s-%d" % (host, i)) for i in range(3)]}
        )

    server = _server(handler)
    try:
        result = await server.stac_search_multi(
            bbox=VALID_BBOX, datetime_range=VALID_RANGE, limit=4
        )
    finally:
        await server.aclose()
    assert len(result.items) == 4  # global cap honored across merged sources


async def test_multi_source_validates_inputs_before_fanning_out():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        calls["n"] += 1
        return httpx.Response(200, json={"features": []})

    server = _server(handler)
    try:
        with pytest.raises(ValidationError):
            await server.stac_search_multi(
                bbox=(200.0, 0.0, 201.0, 1.0), datetime_range=VALID_RANGE
            )
    finally:
        await server.aclose()
    assert calls["n"] == 0  # no source queried on invalid input


async def test_date_only_range_is_expanded_to_full_day_timestamps():
    """A bare date range is sent to STAC as full-day RFC3339 (else 0 results)."""
    import json

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["datetime"] = json.loads(request.content)["datetime"]
        return httpx.Response(200, json={"features": []})

    from geo_stac.search import stac_search

    client = HttpClient(RetryPolicy(max_attempts=1), transport=httpx.MockTransport(handler))
    try:
        await stac_search(
            bbox=VALID_BBOX, datetime_range=("2024-01-01", "2024-02-01"), http=client
        )
    finally:
        await client.aclose()
    assert captured["datetime"] == "2024-01-01T00:00:00Z/2024-02-01T23:59:59Z"

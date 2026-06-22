"""Unit tests for the iNaturalist source (geo-biodiversity).

Drives ``INaturalistSource`` and the merged ``species_occurrences`` path through
an ``httpx.MockTransport`` returning canned iNaturalist observation payloads. No
real network I/O happens.
"""

from __future__ import annotations

import httpx
import pytest

from geo_common.http import HttpClient
from geo_common.retry import RetryPolicy

from geo_biodiversity.occurrences import (
    GBIFSource,
    INaturalistSource,
    default_sources,
    species_occurrences,
)

SMALL_BBOX = (13.40, 52.50, 13.41, 52.51)
INAT_HOST = "api.inaturalist.org"

_INAT_PAYLOAD = {
    "total_results": 2,
    "results": [
        {
            "id": 1234,
            "taxon": {"name": "Vulpes vulpes"},
            "geojson": {"type": "Point", "coordinates": [13.405, 52.505]},
            "observed_on": "2024-05-01",
        },
        {
            "id": 5678,
            "taxon": {"name": "Turdus merula"},
            "location": "52.506,13.406",
            "observed_on": "2024-05-02",
        },
    ],
}


def _client(handler, **kwargs):
    return HttpClient(RetryPolicy(max_attempts=1), transport=httpx.MockTransport(handler), **kwargs)


def test_inaturalist_params_include_bbox_corners_and_geo():
    params = INaturalistSource().build_params(SMALL_BBOX, None, 100)
    assert params["nelat"] == 52.51 and params["swlat"] == 52.50
    assert params["nelng"] == 13.41 and params["swlng"] == 13.40
    assert params["geo"] == "true"
    assert params["per_page"] == 100


def test_inaturalist_params_cap_per_page_and_add_taxon():
    params = INaturalistSource().build_params(SMALL_BBOX, "Aves", 9999)
    assert params["per_page"] == 200  # capped at the iNaturalist page max
    assert params["taxon_name"] == "Aves"


async def test_inaturalist_fetch_parses_observations():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == INAT_HOST
        return httpx.Response(200, json=_INAT_PAYLOAD)

    client = _client(handler)
    try:
        records = await species_occurrences(
            bbox=SMALL_BBOX, http=client, sources=[INaturalistSource()]
        )
    finally:
        await client.aclose()

    assert len(records) == 2
    assert records[0]["source"] == "inaturalist"
    assert records[0]["scientific_name"] == "Vulpes vulpes"
    assert records[0]["longitude"] == pytest.approx(13.405)
    assert records[0]["latitude"] == pytest.approx(52.505)
    # Second record's coords come from the "lat,lng" location string.
    assert records[1]["longitude"] == pytest.approx(13.406)
    assert records[1]["latitude"] == pytest.approx(52.506)


async def test_inaturalist_forwards_bearer_token_when_set():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"results": []})

    client = _client(handler)
    try:
        await species_occurrences(
            bbox=SMALL_BBOX, http=client, sources=[INaturalistSource(token="secret-token")]
        )
    finally:
        await client.aclose()
    assert seen["auth"] == "Bearer secret-token"


def test_default_sources_include_gbif_and_inaturalist():
    names = [s.name for s in default_sources()]
    assert names == ["gbif", "inaturalist"]
    # The optional token is forwarded to the iNaturalist source.
    sourced = default_sources(inaturalist_token="tok")
    inat = [s for s in sourced if isinstance(s, INaturalistSource)][0]
    assert inat.token == "tok"

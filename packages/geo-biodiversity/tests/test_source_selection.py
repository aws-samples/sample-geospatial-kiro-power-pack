"""Source selection for ``species_occurrences`` (geo-biodiversity).

Verifies the optional ``source`` parameter: omitted -> merge the default
occurrence sources (GBIF + iNaturalist); a name -> query just that source; IUCN
is opt-in (reachable only via ``source="iucn"``, never in the default merge);
an unknown name -> validation error. Driven through an ``httpx.MockTransport``
that routes by host, so no real network I/O happens.
"""

from __future__ import annotations

import httpx
import pytest

from geo_common.errors import ErrorCategory, ValidationError
from geo_common.http import HttpClient
from geo_common.retry import RetryPolicy

from geo_biodiversity.occurrences import (
    GBIFSource,
    INaturalistSource,
    IUCNSource,
)
from geo_biodiversity.server import GeoBiodiversityServer

SMALL_BBOX = (13.40, 52.50, 13.41, 52.51)

GBIF_HOST = "api.gbif.org"
INAT_HOST = "api.inaturalist.org"
IUCN_HOST = "api.iucnredlist.org"

_GBIF_PAYLOAD = {
    "results": [
        {
            "key": 1,
            "scientificName": "Vulpes vulpes",
            "decimalLongitude": 13.405,
            "decimalLatitude": 52.505,
            "eventDate": "2024-05-01",
        }
    ]
}
_INAT_PAYLOAD = {
    "results": [
        {
            "id": 99,
            "taxon": {"name": "Turdus merula"},
            "geojson": {"type": "Point", "coordinates": [13.406, 52.506]},
            "observed_on": "2024-05-02",
        }
    ]
}
_IUCN_PAYLOAD = {
    "taxon": {"sis_id": 15951, "scientific_name": "Panthera leo"},
    "assessments": [{"assessment_id": 1, "red_list_category_code": "VU"}],
}


def _server(hosts, *, with_iucn=False):
    """A server whose sources route to a host-recording mock transport."""

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        hosts.append(host)
        if host == GBIF_HOST:
            return httpx.Response(200, json=_GBIF_PAYLOAD)
        if host == INAT_HOST:
            return httpx.Response(200, json=_INAT_PAYLOAD)
        if host == IUCN_HOST:
            return httpx.Response(200, json=_IUCN_PAYLOAD)
        return httpx.Response(404, json={})

    client = HttpClient(RetryPolicy(max_attempts=1), transport=httpx.MockTransport(handler))
    sources = [GBIFSource(), INaturalistSource()]
    if with_iucn:
        sources.append(IUCNSource(token="secret-token"))
    return GeoBiodiversityServer(sources=sources, http=client)


async def test_default_merges_gbif_and_inaturalist():
    hosts: list[str] = []
    server = _server(hosts)
    try:
        records = await server.species_occurrences(bbox=SMALL_BBOX, limit=100)
    finally:
        await server.aclose()
    assert GBIF_HOST in hosts and INAT_HOST in hosts
    assert {r["source"] for r in records} == {"gbif", "inaturalist"}


async def test_source_gbif_queries_only_gbif():
    hosts: list[str] = []
    server = _server(hosts)
    try:
        records = await server.species_occurrences(
            bbox=SMALL_BBOX, limit=100, source="gbif"
        )
    finally:
        await server.aclose()
    assert hosts == [GBIF_HOST]  # only GBIF contacted
    assert all(r["source"] == "gbif" for r in records)


async def test_source_inaturalist_queries_only_inaturalist():
    hosts: list[str] = []
    server = _server(hosts)
    try:
        records = await server.species_occurrences(
            bbox=SMALL_BBOX, limit=100, source="inaturalist"
        )
    finally:
        await server.aclose()
    assert hosts == [INAT_HOST]
    assert all(r["source"] == "inaturalist" for r in records)


async def test_iucn_is_opt_in_not_in_default_merge():
    """Even when IUCN is configured, a default (taxon) query never queries it."""
    hosts: list[str] = []
    server = _server(hosts, with_iucn=True)
    try:
        records = await server.species_occurrences(
            bbox=SMALL_BBOX, taxon="Panthera leo", limit=100
        )
    finally:
        await server.aclose()
    assert IUCN_HOST not in hosts
    assert "iucn" not in {r["source"] for r in records}


async def test_source_iucn_reaches_iucn():
    hosts: list[str] = []
    server = _server(hosts, with_iucn=True)
    try:
        records = await server.species_occurrences(
            bbox=SMALL_BBOX, taxon="Panthera leo", limit=100, source="iucn"
        )
    finally:
        await server.aclose()
    assert hosts == [IUCN_HOST]
    assert records and records[0]["source"] == "iucn"
    assert records[0]["properties"]["category"] == "VU"


async def test_unknown_source_is_validation_error():
    hosts: list[str] = []
    server = _server(hosts)
    try:
        with pytest.raises(ValidationError) as exc_info:
            await server.species_occurrences(
                bbox=SMALL_BBOX, limit=100, source="does-not-exist"
            )
    finally:
        await server.aclose()
    assert exc_info.value.category is ErrorCategory.VALIDATION
    assert exc_info.value.detail["parameter"] == "source"
    assert hosts == []  # nothing queried


async def test_source_iucn_without_taxon_is_validation_error():
    """`source="iucn"` with no taxon -> a clear validation error (IUCN is
    name-based and ignores the bbox), raised before any request."""
    hosts: list[str] = []
    server = _server(hosts, with_iucn=True)
    try:
        with pytest.raises(ValidationError) as exc_info:
            await server.species_occurrences(bbox=SMALL_BBOX, source="iucn")
    finally:
        await server.aclose()
    assert exc_info.value.detail["parameter"] == "taxon"
    assert hosts == []  # rejected before any query


async def test_source_iucn_without_token_is_authentication_error():
    """``source="iucn"`` with IUCN present but no token -> a clean
    AuthenticationError naming the key (mirroring the NOAA CDO pattern),
    raised before any request."""
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        hosts.append(request.url.host)
        return httpx.Response(200, json={})

    client = HttpClient(RetryPolicy(max_attempts=1), transport=httpx.MockTransport(handler))
    # IUCN configured as a source but with no token.
    server = GeoBiodiversityServer(
        sources=[GBIFSource(), INaturalistSource(), IUCNSource(token=None)],
        http=client,
    )
    from geo_common.errors import AuthenticationError

    try:
        with pytest.raises(AuthenticationError) as exc_info:
            await server.species_occurrences(
                bbox=SMALL_BBOX, taxon="Panthera leo", source="iucn"
            )
    finally:
        await server.aclose()
    assert exc_info.value.detail.get("mcp_json_key") == "IUCN_TOKEN"
    assert hosts == []  # guard fires before any request

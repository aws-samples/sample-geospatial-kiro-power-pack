"""Tests for geo-stac ``list_collections`` (catalog discovery), mock-transport.

Exercises the collection-listing path deterministically (no network): parsing
of id/title/description/extent, the substring ``query`` filter, the ``limit``
cap + ``truncated`` flag, pagination via ``rel="next"``, and the up-front
validation guards.
"""

from __future__ import annotations

import httpx
import pytest

from geo_common.errors import ValidationError
from geo_common.http import HttpClient

from geo_stac.search import CollectionList, list_collections
from geo_stac.server import GeoStacServer


def _coll(cid, title=None, bbox=None, temporal=None, keywords=None):
    entry = {"id": cid}
    if title:
        entry["title"] = title
    if keywords:
        entry["keywords"] = keywords
    extent = {}
    if bbox is not None:
        extent["spatial"] = {"bbox": [list(bbox)]}
    if temporal is not None:
        extent["temporal"] = {"interval": [list(temporal)]}
    if extent:
        entry["extent"] = extent
    return entry


def _client(handler):
    return HttpClient(transport=httpx.MockTransport(handler))


async def test_lists_and_parses_collections() -> None:
    payload = {
        "collections": [
            _coll("sentinel-2-l2a", "Sentinel-2 L2A",
                  bbox=[-180, -90, 180, 90],
                  temporal=["2015-06-27T00:00:00Z", None],
                  keywords=["sentinel", "msi"]),
            _coll("landsat-c2-l2", "Landsat C2 L2"),
        ]
    }
    client = _client(lambda r: httpx.Response(200, json=payload))
    try:
        result = await list_collections(catalog="earth-search", http=client)
    finally:
        await client.aclose()

    assert isinstance(result, CollectionList)
    assert result.catalog == "earth-search"
    assert result.returned == 2 and result.truncated is False
    s2 = next(c for c in result.collections if c.id == "sentinel-2-l2a")
    assert s2.title == "Sentinel-2 L2A"
    assert s2.bbox == (-180.0, -90.0, 180.0, 90.0)
    assert s2.temporal_extent == ("2015-06-27T00:00:00Z", None)
    assert "sentinel" in s2.keywords


async def test_query_filters_by_substring() -> None:
    payload = {"collections": [
        _coll("sentinel-2-l2a", "Sentinel-2 L2A", keywords=["sentinel"]),
        _coll("landsat-c2-l2", "Landsat C2 L2"),
        _coll("cop-dem-glo-30", "Copernicus DEM"),
    ]}
    client = _client(lambda r: httpx.Response(200, json=payload))
    try:
        result = await list_collections(catalog="earth-search", query="sentinel", http=client)
    finally:
        await client.aclose()
    assert [c.id for c in result.collections] == ["sentinel-2-l2a"]


async def test_limit_caps_and_sets_truncated() -> None:
    payload = {"collections": [_coll(f"c{i}") for i in range(10)]}
    client = _client(lambda r: httpx.Response(200, json=payload))
    try:
        result = await list_collections(catalog="planetary-computer", limit=3, http=client)
    finally:
        await client.aclose()
    assert result.returned == 3 and result.truncated is True


async def test_follows_pagination_next_link() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "page2" in str(request.url):
            return httpx.Response(200, json={"collections": [_coll("c-b")]})
        return httpx.Response(200, json={
            "collections": [_coll("c-a")],
            "links": [{"rel": "next", "href": "https://x/collections?page2"}],
        })
    client = _client(handler)
    try:
        result = await list_collections(api_url="https://x", http=client)
    finally:
        await client.aclose()
    assert {c.id for c in result.collections} == {"c-a", "c-b"}


async def test_unknown_catalog_is_validation_error() -> None:
    with pytest.raises(ValidationError) as exc:
        await list_collections(catalog="not-a-catalog")
    assert exc.value.detail.get("parameter") == "catalog"


async def test_both_catalog_and_api_url_is_validation_error() -> None:
    with pytest.raises(ValidationError):
        await list_collections(catalog="earth-search", api_url="https://x")


async def test_bad_limit_is_validation_error() -> None:
    with pytest.raises(ValidationError) as exc:
        await list_collections(catalog="earth-search", limit=0)
    assert exc.value.detail.get("parameter") == "limit"


async def test_tool_registered_and_invocable_via_server() -> None:
    payload = {"collections": [_coll("sentinel-2-l2a", "Sentinel-2 L2A")]}
    server = GeoStacServer(http=_client(lambda r: httpx.Response(200, json=payload)))
    try:
        assert "list_collections" in server.tool_names()
        result = await server.list_collections(catalog="earth-search", query="sentinel")
    finally:
        await server.aclose()
    assert result.returned == 1 and result.collections[0].id == "sentinel-2-l2a"

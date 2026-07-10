"""Opt-in LIVE catalog-coverage test for geo-stac (RUN_LIVE_STAC=1).

Skipped by default; run against the real STAC APIs to confirm each configured
catalog actually returns data through geo-stac's own search path, with the
query shape that catalog requires:

    RUN_LIVE_STAC=1 .venv/bin/python -m pytest \
      packages/geo-stac/tests/test_stac_live.py -q

Findings baked into the cases (verified against the live APIs):
* Earth Search and USGS answer a bbox/datetime query with no collections filter.
* Planetary Computer and CMR-STAC (NASA LPCLOUD) return nothing without a
  ``collections`` filter, so one is supplied.
* Copernicus Data Space serves CLMS land-monitoring products (not raw
  Sentinel), so a global-daily CLMS collection over a wide bbox is used.
"""

from __future__ import annotations

import os

import pytest

from geo_stac.search import (
    KNOWN_STAC_ENDPOINTS,
    StacItem,
    stac_search,
    stac_search_multi,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_STAC") != "1",
    reason="set RUN_LIVE_STAC=1 to run live STAC catalog-coverage checks",
)

_SF_BBOX = (-122.6, 37.6, -122.3, 37.9)
_SF_RANGE = ("2024-06-01", "2024-06-30")
# Copernicus CLMS burned-area is global/daily; a wide bbox + a few days finds fires.
_WIDE_BBOX = (-20.0, -40.0, 55.0, 40.0)  # Africa
_WIDE_RANGE = ("2024-06-01", "2024-06-10")

# (catalog, bbox, range, collections) — the query each catalog needs to return data.
_CASES = [
    ("earth-search", _SF_BBOX, _SF_RANGE, None),
    ("usgs", _SF_BBOX, _SF_RANGE, None),
    ("planetary-computer", _SF_BBOX, _SF_RANGE, ["sentinel-2-l2a"]),
    ("cmr-stac", _SF_BBOX, _SF_RANGE, ["HLSS30_2.0"]),
    ("copernicus", _WIDE_BBOX, _WIDE_RANGE, ["clms_ba_global_300m_daily_v3_cog"]),
]


@pytest.mark.parametrize("name,bbox,drange,collections", _CASES)
async def test_catalog_returns_data(name, bbox, drange, collections) -> None:
    """Each configured catalog returns usable items for its expected query."""
    items = await stac_search(
        bbox=bbox, datetime_range=drange, collections=collections, limit=5,
        api_url=KNOWN_STAC_ENDPOINTS[name],
    )
    assert items, f"{name} returned no items for its expected query"
    for it in items:
        assert isinstance(it, StacItem)
        assert it.id and len(it.bbox) == 4 and it.datetime


@pytest.mark.parametrize("name", list(KNOWN_STAC_ENDPOINTS))
async def test_list_collections_live(name) -> None:
    """Every catalog reports its collections (discovery of what to request)."""
    from geo_stac.search import list_collections

    result = await list_collections(catalog=name, limit=200)
    assert result.returned > 0, f"{name} reported no collections"
    assert all(c.id for c in result.collections)


async def test_list_collections_query_filter_live() -> None:
    """A 'sentinel' query narrows Planetary Computer's ~135 collections."""
    from geo_stac.search import list_collections

    all_pc = await list_collections(catalog="planetary-computer", limit=200)
    s2 = await list_collections(catalog="planetary-computer", query="sentinel", limit=200)
    assert 0 < s2.returned < all_pc.returned
    assert all("sentinel" in (c.id + " " + (c.title or "")).lower() or
               any("sentinel" in k.lower() for k in c.keywords) for c in s2.collections)


async def test_federated_search_spans_multiple_catalogs() -> None:
    """A federated Sentinel-2 search returns items from >1 catalog (ES + PC)."""
    result = await stac_search_multi(
        bbox=_SF_BBOX, datetime_range=_SF_RANGE,
        collections=["sentinel-2-l2a"], limit=20,
    )
    contributing = [s for s in result.sources if s.status == "ok" and s.count > 0]
    assert len(contributing) >= 2, (
        "expected >=2 catalogs to contribute for sentinel-2-l2a, got "
        f"{[(s.name, s.count) for s in result.sources]}"
    )
    assert result.items

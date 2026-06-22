"""Property test for the Resource Catalog search/filters (Requirement 2.2-2.4).

**Feature: geospatial-power-pack, Property 5: Catalog search and filters are
sound and complete**

*For any* catalog and any query (keyword and/or openness-tier and/or pillar
filter), the returned set is exactly the set of entries that match every
supplied constraint: the keyword appears as a case-insensitive substring of the
entry's name, capability description, or pillar, and the entry's openness tier
and pillar equal any supplied tier/pillar filter.

Validates: Requirements 2.2, 2.3, 2.4

The test pairs ``ResourceCatalog.search`` against an independent reference
predicate (the oracle) over the same registered entries: soundness (every
returned entry matches) and completeness (every matching entry is returned) are
both covered by asserting set equality between the catalog's result and the
oracle's selection. Generators draw names/descriptions/pillars and keywords
from a small shared alphabet so substring hits and misses both occur often,
giving the ">=100 cases" search real signal rather than near-always-empty
results.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pytest
from hypothesis import given
from hypothesis import strategies as st

from geo_common.models import CatalogEntry, OpennessTier
from kiro_geospatial.catalog import CatalogQuery, CatalogResult, ResourceCatalog


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #
# A small shared alphabet so generated keywords frequently appear (and
# frequently do not appear) as substrings of generated entry fields. The space
# character is included so multi-token-ish content and keywords are exercised.
_ALPHABET = "abAB "

# Pillar values: a small set drawn from the design's pillar vocabulary plus a
# couple of case/format variants so case-insensitive substring matching (over
# pillar) is distinguished from exact pillar-equality filtering.
_PILLARS = ["A", "B", "C", "a", "expansion", "peer"]

_TIERS = list(OpennessTier)

_entry_strategy = st.builds(
    CatalogEntry,
    name=st.text(alphabet=_ALPHABET, min_size=1, max_size=12),
    pillar=st.sampled_from(_PILLARS),
    capability_description=st.text(alphabet=_ALPHABET, min_size=1, max_size=24),
    openness_tier=st.sampled_from(_TIERS),
    provider_server=st.text(alphabet="xyz", min_size=1, max_size=5),
    installed=st.booleans(),
)

# Keyword: absent, or a 1-8 char string from the shared alphabet (within the
# 1-200 valid range, so CatalogQuery construction always succeeds here).
_keyword_strategy = st.one_of(
    st.none(),
    st.text(alphabet=_ALPHABET, min_size=1, max_size=8),
)

_tier_filter_strategy = st.one_of(st.none(), st.sampled_from(_TIERS))
_pillar_filter_strategy = st.one_of(st.none(), st.sampled_from(_PILLARS))


# --------------------------------------------------------------------------- #
# Oracle + helpers
# --------------------------------------------------------------------------- #
def _oracle_matches(
    entry: CatalogEntry,
    keyword: Optional[str],
    tier: Optional[OpennessTier],
    pillar: Optional[str],
) -> bool:
    """Independent reference predicate for "entry satisfies every constraint"."""
    if keyword is not None:
        k = keyword.lower()
        if (
            k not in entry.name.lower()
            and k not in entry.capability_description.lower()
            and k not in entry.pillar.lower()
        ):
            return False
    if tier is not None and entry.openness_tier != tier:
        return False
    if pillar is not None and entry.pillar != pillar:
        return False
    return True


def _key(entry: CatalogEntry) -> Tuple:
    """A hashable/comparable identity for an entry, for multiset comparison."""
    return (
        entry.name,
        entry.pillar,
        entry.capability_description,
        entry.openness_tier.value,
        entry.provider_server,
        entry.installed,
        entry.install_command,
    )


def _sorted_keys(entries: List[CatalogEntry]) -> List[Tuple]:
    return sorted(_key(e) for e in entries)


# --------------------------------------------------------------------------- #
# Property 5: search/filters are sound and complete
# --------------------------------------------------------------------------- #
@pytest.mark.property
@given(
    entries=st.lists(_entry_strategy, max_size=15),
    keyword=_keyword_strategy,
    tier=_tier_filter_strategy,
    pillar=_pillar_filter_strategy,
)
def test_catalog_search_is_sound_and_complete(
    entries: List[CatalogEntry],
    keyword: Optional[str],
    tier: Optional[OpennessTier],
    pillar: Optional[str],
) -> None:
    """The result is exactly the entries matching every supplied constraint."""
    catalog = ResourceCatalog()
    catalog.register(entries)

    query = CatalogQuery(keyword=keyword, openness_tier=tier, pillar=pillar)
    result: CatalogResult = catalog.search(query)

    expected = [e for e in entries if _oracle_matches(e, keyword, tier, pillar)]

    # Set/multiset equality of result vs oracle covers both directions:
    #  - soundness: nothing returned that the oracle rejects;
    #  - completeness: nothing the oracle accepts is missing.
    assert _sorted_keys(result.entries) == _sorted_keys(expected)

    # Soundness, asserted directly per returned entry (Requirements 2.2-2.4).
    for entry in result.entries:
        assert _oracle_matches(entry, keyword, tier, pillar)

    # Completeness, asserted directly per matching entry.
    returned = _sorted_keys(result.entries)
    for entry in expected:
        assert _key(entry) in returned

    # The explanatory no-match message appears exactly when nothing matched
    # (Requirement 2.5), and never crowds out real results.
    if expected:
        assert result.message is None
    else:
        assert result.entries == []
        assert result.message is not None

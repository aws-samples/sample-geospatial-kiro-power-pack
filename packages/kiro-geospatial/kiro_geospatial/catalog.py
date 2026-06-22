"""The Power Hub's searchable Resource Catalog (Requirement 2).

The :class:`ResourceCatalog` is an in-memory index of every capability the
Power Pack can reach. Each capability is a :class:`~geo_common.models.CatalogEntry`
- the same shared shape servers register through ``BaseGeoServer.catalog_entries``
and the Orchestration Router resolves against - carrying the entry name, pillar,
capability description, :class:`~geo_common.models.OpennessTier`, providing
server, install state, and (for not-installed providers) an install command
(Requirements 2.1, 2.6, 11.3).

Search (Requirement 2.2-2.8):

* a **keyword** of 1-200 characters matches an entry when it appears as a
  case-insensitive substring of the entry's ``name``, ``capability_description``,
  or ``pillar`` (Requirement 2.2);
* an **openness-tier** filter keeps only entries of that tier (Requirement 2.3);
* a **pillar** filter keeps only entries belonging to that pillar
  (Requirement 2.4);
* the filters combine: a result entry must satisfy *every* supplied constraint;
* a query matching nothing returns an empty set plus an explanatory message
  (Requirement 2.5);
* a keyword that is empty or exceeds 200 characters is rejected before any
  search runs, with a validation error stating the valid length range
  (Requirement 2.7); :class:`CatalogQuery` enforces the same bound at
  construction;
* results come from an in-memory structure, returning well within 2 seconds
  (Requirement 2.8).

Python 3.9 compatibility: this module uses ``from __future__ import
annotations`` together with ``typing.Optional``/``typing.List`` so the pydantic
models resolve their annotations on 3.9+ (matching :mod:`geo_common.models`).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

# Reuse the shared catalog shape and openness tier so the hub, the servers that
# register capabilities, and the Orchestration Router all speak one vocabulary.
from geo_common.models import CatalogEntry, OpennessTier
from geo_common.errors import ValidationError

__all__ = [
    "CatalogEntry",
    "OpennessTier",
    "CatalogQuery",
    "CatalogResult",
    "ResourceCatalog",
    "KEYWORD_MIN_LENGTH",
    "KEYWORD_MAX_LENGTH",
    "NO_MATCH_MESSAGE",
]

# Valid keyword length range (Requirement 2.2 / 2.7).
KEYWORD_MIN_LENGTH = 1
KEYWORD_MAX_LENGTH = 200

NO_MATCH_MESSAGE = "No matching capabilities were found."


class CatalogQuery(BaseModel):
    """A Resource Catalog query (Requirements 2.2, 2.3, 2.4, 2.7).

    Any of the three constraints may be supplied independently or together:

    * ``keyword`` - a 1-200 character case-insensitive substring matched against
      an entry's name, capability description, or pillar (Requirement 2.2). The
      length bound is enforced here at construction so an out-of-range keyword
      is rejected before any search runs (Requirement 2.7).
    * ``openness_tier`` - keep only entries of this tier (Requirement 2.3).
    * ``pillar`` - keep only entries belonging to this pillar (Requirement 2.4).

    A query with no constraints (all three ``None``) matches every entry.
    """

    keyword: Optional[str] = Field(
        default=None,
        min_length=KEYWORD_MIN_LENGTH,
        max_length=KEYWORD_MAX_LENGTH,
    )
    openness_tier: Optional[OpennessTier] = None
    pillar: Optional[str] = None


class CatalogResult(BaseModel):
    """The outcome of a catalog search (Requirements 2.2-2.5).

    ``entries`` is exactly the set of entries satisfying every supplied
    constraint. ``message`` is populated only when ``entries`` is empty, stating
    that no matching capabilities were found (Requirement 2.5).
    """

    entries: List[CatalogEntry]
    message: Optional[str] = None


class ResourceCatalog:
    """An in-memory, searchable index of catalog entries (Requirement 2).

    Entries are registered through :meth:`register` and queried through
    :meth:`search`. Search is sound and complete: it returns precisely the
    entries that match every supplied constraint of the query (Property 5).
    """

    def __init__(self, entries: Optional[List[CatalogEntry]] = None) -> None:
        self._entries: List[CatalogEntry] = []
        if entries:
            self.register(entries)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, entries: List[CatalogEntry]) -> None:
        """Add ``entries`` to the catalog index (Requirements 2.1, 11.3)."""
        self._entries.extend(entries)

    @property
    def entries(self) -> List[CatalogEntry]:
        """A copy of all registered entries, in registration order."""
        return list(self._entries)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: CatalogQuery) -> CatalogResult:
        """Return the entries matching every supplied constraint (Property 5).

        Implements Requirements 2.2 (case-insensitive substring over
        name/description/pillar), 2.3 (openness-tier equality), 2.4 (pillar
        equality), 2.5 (empty set + message on no match), and 2.7 (reject an
        out-of-range keyword before searching). The match is the conjunction of
        every supplied constraint; an absent constraint imposes nothing.
        """
        keyword = query.keyword
        # Defensive re-validation of the keyword length (Requirement 2.7): the
        # query model already enforces this, but a caller could mutate the
        # field, so the search surface guards too, raising the taxonomy's
        # validation error stating the valid range.
        if keyword is not None and not (
            KEYWORD_MIN_LENGTH <= len(keyword) <= KEYWORD_MAX_LENGTH
        ):
            raise ValidationError(
                "Keyword query length is out of range: a keyword must be "
                f"{KEYWORD_MIN_LENGTH} to {KEYWORD_MAX_LENGTH} characters.",
                detail={
                    "min_length": KEYWORD_MIN_LENGTH,
                    "max_length": KEYWORD_MAX_LENGTH,
                    "actual_length": len(keyword),
                },
            )

        keyword_lower = keyword.lower() if keyword is not None else None

        matches = [
            entry
            for entry in self._entries
            if self._entry_matches(entry, keyword_lower, query.openness_tier, query.pillar)
        ]

        if not matches:
            return CatalogResult(entries=[], message=NO_MATCH_MESSAGE)
        return CatalogResult(entries=matches)

    @staticmethod
    def _entry_matches(
        entry: CatalogEntry,
        keyword_lower: Optional[str],
        openness_tier: Optional[OpennessTier],
        pillar: Optional[str],
    ) -> bool:
        """True when ``entry`` satisfies every supplied constraint."""
        # Keyword: case-insensitive substring of name | description | pillar.
        if keyword_lower is not None:
            haystacks = (
                entry.name.lower(),
                entry.capability_description.lower(),
                entry.pillar.lower(),
            )
            if not any(keyword_lower in field for field in haystacks):
                return False

        # Openness-tier filter: equality (Requirement 2.3).
        if openness_tier is not None and entry.openness_tier != openness_tier:
            return False

        # Pillar filter: equality (Requirement 2.4).
        if pillar is not None and entry.pillar != pillar:
            return False

        return True

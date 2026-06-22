"""Lightweight shared models for the geo-common base contract.

``BaseGeoServer`` (see :mod:`geo_common.server`) needs two small,
protocol-friendly data types in its method signatures:

* :class:`CredentialSpec` - an ``mcp.json`` configuration key a server needs,
  plus its :class:`CredentialClassification` (Required / Optional /
  License-Needed). The startup credential guard relies on the classification
  to decide whether an absent key must block startup (Requirements 16.4-16.6).
* :class:`CatalogEntry` - a capability a server registers in the Power Hub's
  Resource Catalog (Requirements 2.1, 11.3).

These are intentionally **minimal**. The Power Hub defines richer versions of
``CatalogEntry``/``CredentialSpec``/``OpennessTier`` (see design.md "Power
Hub"); the fields here are a compatible subset so the two can later be aligned
without breaking the base contract. Keeping them in ``geo-common`` means every
server - and the hub - can share one definition rather than duplicating it.

Python 3.9 compatibility: this module uses ``from __future__ import
annotations`` together with ``typing.Optional`` so the pydantic models resolve
their annotations on 3.9+.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

__all__ = [
    "OpennessTier",
    "CredentialClassification",
    "CredentialSpec",
    "CatalogEntry",
]


class OpennessTier(str, Enum):
    """How open a source is (Requirement 2.1).

    A ``str`` enum so the value serializes directly to its wire string in JSON
    and pydantic output.
    """

    OPEN = "Open"
    FREE_TIER = "Free-Tier"
    PROPRIETARY = "Proprietary/Licensed"


class CredentialClassification(str, Enum):
    """The configuration requirement of a source credential (Requirement 3.2).

    Only :attr:`REQUIRED` credentials block a server from starting when absent
    (Requirement 16.6); :attr:`OPTIONAL` credentials never do (Requirement
    16.5). :attr:`LICENSE_NEEDED` marks a credential that also carries a
    licensing obligation (Requirements 3.7, 10.4).
    """

    REQUIRED = "Required"
    OPTIONAL = "Optional"
    LICENSE_NEEDED = "License-Needed"


class CredentialSpec(BaseModel):
    """An ``mcp.json`` key a server needs, plus how it is classified.

    Carries **no** secret-bearing field (Requirement 3.8): it describes *which*
    key is needed and *how* it is classified, never the secret value itself.
    """

    source: str = Field(min_length=1)
    mcp_json_key: str = Field(min_length=1)
    classification: CredentialClassification
    license_reference: Optional[str] = None  # echoed for License-Needed (Req 3.7)


class CatalogEntry(BaseModel):
    """A capability a server registers in the Resource Catalog (Req 2.1, 11.3).

    A minimal, hub-compatible subset of the Power Hub's catalog entry. The
    providing server names itself via ``provider_server`` so wrapped external
    capabilities are still attributed to the wrapping server (Requirement
    11.3).
    """

    name: str = Field(min_length=1, max_length=100)
    pillar: str = Field(min_length=1)
    capability_description: str = Field(min_length=1, max_length=500)
    openness_tier: OpennessTier
    provider_server: str = Field(min_length=1)
    installed: bool = False
    install_command: Optional[str] = None  # present when provider not installed (Req 2.6)

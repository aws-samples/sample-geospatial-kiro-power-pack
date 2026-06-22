"""Data models for the ``geo-commercial-imagery`` server (Credentialed, expansion).

``geo-commercial-imagery`` fronts Sentinel Hub (which in turn fronts the Planet,
Maxar, and Airbus commercial constellations) and exposes two tools:

* ``search_commercial`` returns a list of :class:`StacItem` describing the
  commercial scenes matching a spatial + temporal query (design.md
  "Credentialed / Proprietary modules").
* ``order_scene`` returns an :class:`OrderConfirmation` describing the placed
  order for a commercial scene.

The :class:`StacItem` here is intentionally shape-compatible with
``geo-stac``'s ``StacItem`` (this package depends only on ``geo-common``, not on
``geo-stac``, so the type is redeclared rather than imported), so a commercial
search result flows through the rest of the Power Pack like any other STAC item.

Python 3.10+ (matching ``pyproject.toml``): uses ``from __future__ import
annotations`` together with ``typing`` generics so the pydantic models resolve
their annotations cleanly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

__all__ = ["BBox", "StacItem", "OrderConfirmation"]

#: A geographic bounding box as ``(west, south, east, north)`` in EPSG:4326
#: decimal degrees - the order used throughout the connector and by GeoJSON's
#: ``bbox`` member (RFC 7946 §5).
BBox = Tuple[float, float, float, float]


class StacItem(BaseModel):
    """A commercial STAC item returned by ``search_commercial``.

    Shape-compatible with ``geo-stac``'s ``StacItem``: each item carries its
    asset references (``assets``) and its spatial (``bbox``) and temporal
    (``datetime``) metadata, plus the originating ``collection`` (the Sentinel
    Hub collection / constellation, e.g. PlanetScope, WorldView, Pléiades) and
    its STAC ``properties`` for downstream selection.
    """

    id: str
    #: ``(west, south, east, north)`` in EPSG:4326 decimal degrees.
    bbox: BBox
    #: ISO-8601 timestamp (or interval start) describing the item's time.
    datetime: str
    #: The Sentinel Hub collection / constellation the item came from.
    collection: Optional[str] = None
    #: STAC asset references keyed by asset name.
    assets: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    #: STAC item properties.
    properties: Dict[str, Any] = Field(default_factory=dict)


class OrderConfirmation(BaseModel):
    """Confirmation of a commercial-scene order created via ``order_scene``.

    Returned when a Third-Party Data Import (TPDI) order is accepted by Sentinel
    Hub. ``order_id`` is the provider-assigned order identifier; ``provider`` is
    the TPDI provider the order was placed with (``PLANET`` / ``MAXAR`` /
    ``AIRBUS``); ``scene_ids`` echoes the requested product ids; ``status`` is
    the order's lifecycle state (e.g. ``CREATED``, ``RUNNING``, ``DONE``).

    Creating an order does not spend account quota; *confirming* it does.
    ``confirmed`` records whether the confirm step was issued. ``sqkm`` is the
    ordered area in square kilometres (the quota cost) when Sentinel Hub
    reports it; ``collection`` is the destination BYOC collection id; and
    ``created`` is the ISO-8601 creation timestamp when supplied. Never carries
    credential material.
    """

    order_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    scene_ids: List[str] = Field(min_length=1)
    status: str = Field(min_length=1)
    confirmed: bool = False
    sqkm: Optional[float] = None
    collection: Optional[str] = None
    created: Optional[str] = None

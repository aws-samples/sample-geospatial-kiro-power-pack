"""Catalog/credential registration consistency for ``geo-raster`` (task 14.11).

Cross-cutting registration sweep for the Pillar B expansion server
``geo-raster``. These tests pin the server's in-code registration against the
single source of truth on each side - the running
:meth:`~geo_raster.server.GeoRasterServer.catalog_entries` /
:meth:`~geo_raster.server.GeoRasterServer.required_credentials` and the
repo-root ``bundle-manifest.json`` - and assert the two agree:

* the capability registers a Resource Catalog entry naming ``geo-raster`` as
  provider, at an open tier, with the ``uvx`` install command for the
  not-installed case (Requirements 2.1, 2.6);
* the server's in-code :meth:`required_credentials` agrees exactly with the
  ``geo-raster`` credential block in ``bundle-manifest.json`` (Requirement 16.1);
* the server is declared as a Pillar B expansion module (Requirement 2.1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from geo_common.models import CredentialClassification, OpennessTier

from geo_raster.server import INSTALL_COMMAND, GeoRasterServer

#: Repo root: tests/ -> geo-raster/ -> packages/ -> <repo root>.
_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "bundle-manifest.json"

_OPEN_TIERS = {OpennessTier.OPEN, OpennessTier.FREE_TIER}


def _manifest_entry() -> Dict[str, Any]:
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    servers = {s["name"]: s for s in manifest["servers"]}
    assert "geo-raster" in servers, (
        "geo-raster must be declared in bundle-manifest.json"
    )
    return servers["geo-raster"]


def test_catalog_entries_register_geo_raster_as_open_provider() -> None:
    """Req 2.1 / 2.6: the capabilities register Open, naming geo-raster."""
    entries = GeoRasterServer().catalog_entries()
    assert {e.name for e in entries} == {"zonal_statistics", "read_window", "band_math"}
    for entry in entries:
        assert entry.pillar == "B"
        assert entry.provider_server == "geo-raster"
        assert entry.openness_tier in _OPEN_TIERS
        assert 1 <= len(entry.name) <= 100
        assert 1 <= len(entry.capability_description) <= 500
        assert entry.install_command == INSTALL_COMMAND == "uvx geo-raster"


def test_required_credentials_match_manifest() -> None:
    """Req 16.1: in-code credential specs agree exactly with the manifest."""
    specs = GeoRasterServer().required_credentials()
    spec_pairs = {(s.mcp_json_key, s.classification.value) for s in specs}
    manifest_pairs = {
        (c["key"], c["classification"]) for c in _manifest_entry()["credentials"]
    }
    assert spec_pairs == manifest_pairs
    assert all(
        s.classification is CredentialClassification.OPTIONAL for s in specs
    )
    # Optional credentials never block startup (Req 16.5).
    server = GeoRasterServer()
    server.start(configured_keys=[])
    assert server.started is True


def test_manifest_declares_pillar_b_expansion_with_matching_install() -> None:
    """Req 2.1 / 2.6: manifest declares geo-raster as a Pillar B expansion."""
    entry = _manifest_entry()
    assert entry["pillar"] == "B"
    assert entry["status"] == "Expansion"
    assert entry["uvx"] == INSTALL_COMMAND == "uvx geo-raster"

"""Catalog/credential registration consistency for ``geo-index`` (task 14.11).

Cross-cutting registration sweep for the Pillar B expansion server
``geo-index``. These tests pin the server's in-code registration against the
single source of truth on each side - the running
:meth:`~geo_index.server.GeoIndexServer.catalog_entries` /
:meth:`~geo_index.server.GeoIndexServer.required_credentials` and the repo-root
``bundle-manifest.json`` - and assert the two agree:

* the capability registers a Resource Catalog entry naming ``geo-index`` as
  provider at an open tier (Requirement 2.1);
* the server is local and credential-free, so it declares no credentials and
  the ``geo-index`` manifest credential block is likewise empty (Req 16.1, 16.5);
* the server is declared as a Pillar B expansion module (Requirement 2.1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from geo_common.models import OpennessTier

from geo_index.server import GeoIndexServer

#: Repo root: tests/ -> geo-index/ -> packages/ -> <repo root>.
_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "bundle-manifest.json"

_OPEN_TIERS = {OpennessTier.OPEN, OpennessTier.FREE_TIER}

_INSTALL_COMMAND = "uvx geo-index"


def _manifest_entry() -> Dict[str, Any]:
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    servers = {s["name"]: s for s in manifest["servers"]}
    assert "geo-index" in servers, (
        "geo-index must be declared in bundle-manifest.json"
    )
    return servers["geo-index"]


def test_catalog_entries_register_geo_index_as_open_provider() -> None:
    """Req 2.1: the capability registers Open, naming geo-index."""
    entries = GeoIndexServer().catalog_entries()
    assert {e.name for e in entries} == {"index_cell"}
    for entry in entries:
        assert entry.pillar == "B"
        assert entry.provider_server == "geo-index"
        assert entry.openness_tier in _OPEN_TIERS
        assert 1 <= len(entry.name) <= 100
        assert 1 <= len(entry.capability_description) <= 500
        # Local, always-available capability: installed, so no install command
        # is surfaced (Req 2.6 applies only to not-installed providers).
        assert entry.installed is True
        assert entry.install_command is None


def test_required_credentials_match_manifest() -> None:
    """Req 16.1 / 16.5: geo-index is credential-free and the manifest agrees."""
    specs = GeoIndexServer().required_credentials()
    spec_pairs = {(s.mcp_json_key, s.classification.value) for s in specs}
    manifest_pairs = {
        (c["key"], c["classification"]) for c in _manifest_entry()["credentials"]
    }
    assert spec_pairs == manifest_pairs == set()
    # Credential-free: starts with no configured credentials.
    server = GeoIndexServer()
    server.start(configured_keys=[])
    assert server.started is True


def test_manifest_declares_pillar_b_expansion_with_matching_install() -> None:
    """Req 2.1: manifest declares geo-index as a Pillar B expansion."""
    entry = _manifest_entry()
    assert entry["pillar"] == "B"
    assert entry["status"] == "Expansion"
    assert entry["uvx"] == _INSTALL_COMMAND

"""Catalog/credential registration consistency for ``geo-pointcloud`` (task 14.11).

Cross-cutting registration sweep for the Pillar B expansion server
``geo-pointcloud``. These tests pin the server's in-code registration against
the single source of truth on each side - the running
:meth:`~geo_pointcloud.server.GeoPointcloudServer.catalog_entries` /
:meth:`~geo_pointcloud.server.GeoPointcloudServer.required_credentials` and the
repo-root ``bundle-manifest.json`` - and assert the two agree:

* every capability registers a Resource Catalog entry naming ``geo-pointcloud``
  as provider, at an open tier, with the ``uvx`` install command for the
  not-installed case (Requirements 2.1, 2.6);
* the server's in-code :meth:`required_credentials` agrees exactly with the
  ``geo-pointcloud`` credential block in ``bundle-manifest.json`` (Req 16.1);
* the server is declared as a Pillar B expansion module (Requirement 2.1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from geo_common.models import CredentialClassification, OpennessTier

from geo_pointcloud.server import INSTALL_COMMAND, GeoPointcloudServer

#: Repo root: tests/ -> geo-pointcloud/ -> packages/ -> <repo root>.
_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "bundle-manifest.json"

_OPEN_TIERS = {OpennessTier.OPEN, OpennessTier.FREE_TIER}


def _manifest_entry() -> Dict[str, Any]:
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    servers = {s["name"]: s for s in manifest["servers"]}
    assert "geo-pointcloud" in servers, (
        "geo-pointcloud must be declared in bundle-manifest.json"
    )
    return servers["geo-pointcloud"]


def test_catalog_entries_register_geo_pointcloud_as_open_provider() -> None:
    """Req 2.1 / 2.6: each capability registers Open, naming geo-pointcloud."""
    entries = GeoPointcloudServer().catalog_entries()
    assert {e.name for e in entries} == {"read_pointcloud", "write_pointcloud"}
    for entry in entries:
        assert entry.pillar == "B"
        assert entry.provider_server == "geo-pointcloud"
        assert entry.openness_tier in _OPEN_TIERS
        assert 1 <= len(entry.name) <= 100
        assert 1 <= len(entry.capability_description) <= 500
        assert entry.install_command == INSTALL_COMMAND == "uvx geo-pointcloud"


def test_required_credentials_match_manifest() -> None:
    """Req 16.1: in-code credential specs agree exactly with the manifest."""
    specs = GeoPointcloudServer().required_credentials()
    spec_pairs = {(s.mcp_json_key, s.classification.value) for s in specs}
    manifest_pairs = {
        (c["key"], c["classification"]) for c in _manifest_entry()["credentials"]
    }
    assert spec_pairs == manifest_pairs
    assert all(
        s.classification is CredentialClassification.OPTIONAL for s in specs
    )
    # Optional credentials never block startup (Req 16.5).
    server = GeoPointcloudServer()
    server.start(configured_keys=[])
    assert server.started is True


def test_manifest_declares_pillar_b_expansion_with_matching_install() -> None:
    """Req 2.1 / 2.6: manifest declares geo-pointcloud as a Pillar B expansion."""
    entry = _manifest_entry()
    assert entry["pillar"] == "B"
    assert entry["status"] == "Expansion"
    assert entry["uvx"] == INSTALL_COMMAND == "uvx geo-pointcloud"

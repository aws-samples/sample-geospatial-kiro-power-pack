"""Catalog/credential registration consistency for ``geo-query`` (task 14.11).

Cross-cutting registration sweep for the Pillar B expansion server
``geo-query``. These tests pin the server's in-code registration against the
single source of truth on each side - the running
:meth:`~geo_query.server.GeoQueryServer.catalog_entries` /
:meth:`~geo_query.server.GeoQueryServer.required_credentials` and the repo-root
``bundle-manifest.json`` - and assert the two agree:

* each engine registers a Resource Catalog entry naming ``geo-query`` as
  provider at its openness tier, with the ``uvx`` install command for the
  not-installed case (Requirements 2.1, 2.6);
* the server's in-code :meth:`required_credentials` agrees exactly with the
  ``geo-query`` credential block in ``bundle-manifest.json`` (Requirement 16.1);
* the server is declared as a Pillar B expansion module (Requirement 2.1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from geo_common.models import CredentialClassification, OpennessTier

from geo_query.server import INSTALL_COMMAND, GeoQueryServer

#: Repo root: tests/ -> geo-query/ -> packages/ -> <repo root>.
_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "bundle-manifest.json"

_OPEN_TIERS = {OpennessTier.OPEN, OpennessTier.FREE_TIER}


def _manifest_entry() -> Dict[str, Any]:
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    servers = {s["name"]: s for s in manifest["servers"]}
    assert "geo-query" in servers, (
        "geo-query must be declared in bundle-manifest.json"
    )
    return servers["geo-query"]


def test_catalog_entries_register_geo_query_per_engine() -> None:
    """Req 2.1 / 2.6: one entry per engine, naming geo-query at its tier."""
    entries = {e.name: e for e in GeoQueryServer().catalog_entries()}
    assert set(entries) == {"spatial_sql:duckdb", "spatial_sql:athena"}
    for entry in entries.values():
        assert entry.pillar == "B"
        assert entry.provider_server == "geo-query"
        assert entry.openness_tier in _OPEN_TIERS
        assert 1 <= len(entry.name) <= 100
        assert 1 <= len(entry.capability_description) <= 500
        # No engine is configured on a default server -> install command shown.
        assert entry.install_command == INSTALL_COMMAND == "uvx geo-query"
    assert entries["spatial_sql:duckdb"].openness_tier is OpennessTier.OPEN
    assert entries["spatial_sql:athena"].openness_tier is OpennessTier.FREE_TIER


def test_required_credentials_match_manifest() -> None:
    """Req 16.1: in-code credential specs agree exactly with the manifest."""
    specs = GeoQueryServer().required_credentials()
    spec_pairs = {(s.mcp_json_key, s.classification.value) for s in specs}
    manifest_pairs = {
        (c["key"], c["classification"]) for c in _manifest_entry()["credentials"]
    }
    assert spec_pairs == manifest_pairs
    assert all(
        s.classification is CredentialClassification.OPTIONAL for s in specs
    )
    # Optional credentials never block startup (Req 16.5).
    server = GeoQueryServer()
    server.start(configured_keys=[])
    assert server.started is True


def test_manifest_declares_pillar_b_expansion_with_matching_install() -> None:
    """Req 2.1 / 2.6: manifest declares geo-query as a Pillar B expansion."""
    entry = _manifest_entry()
    assert entry["pillar"] == "B"
    assert entry["status"] == "Expansion"
    assert entry["uvx"] == INSTALL_COMMAND == "uvx geo-query"

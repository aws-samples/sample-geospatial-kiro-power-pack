"""Manifest/code credential-consistency test for ``geo-warehouse``.

The server's in-code credential declarations (``required_credentials()``, driven
by ``SUPPORTED_ENGINES``) and the ``geo-warehouse`` block in the repo-root
``bundle-manifest.json`` must agree exactly on credential keys and
classifications. This test pins that agreement so the two sources cannot
silently drift (the drift that previously left the manifest declaring
``SNOWFLAKE_ACCOUNT``/``USER``/``PASSWORD`` while the engine used a single
``SNOWFLAKE_CONNECTION`` key).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from geo_common.models import CredentialClassification

from geo_warehouse.server import GeoWarehouseServer

_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "bundle-manifest.json"


@pytest.fixture(scope="module")
def warehouse_entry() -> Dict[str, Any]:
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    servers = {s["name"]: s for s in manifest["servers"]}
    assert "geo-warehouse" in servers
    return servers["geo-warehouse"]


def test_required_credentials_match_manifest(warehouse_entry: Dict[str, Any]) -> None:
    """Every (key, classification) the server declares appears in the manifest."""
    specs = GeoWarehouseServer().required_credentials()
    code_pairs = {(s.mcp_json_key, s.classification.value) for s in specs}
    manifest_pairs = {
        (c["key"], c["classification"]) for c in warehouse_entry["credentials"]
    }
    assert code_pairs == manifest_pairs


def test_all_warehouse_credentials_are_license_needed(
    warehouse_entry: Dict[str, Any],
) -> None:
    specs = GeoWarehouseServer().required_credentials()
    assert specs, "geo-warehouse must declare its proprietary engine credentials"
    assert all(
        s.classification is CredentialClassification.LICENSE_NEEDED for s in specs
    )
    assert all(
        c["classification"] == "License-Needed" for c in warehouse_entry["credentials"]
    )


def test_server_starts_without_credentials() -> None:
    """License-Needed keys never block startup (Req 16.5)."""
    server = GeoWarehouseServer()
    server.start(configured_keys=[])
    assert server.started is True

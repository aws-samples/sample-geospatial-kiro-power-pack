"""Catalog/credential registration tests for ``geo-embedding-search`` (task 15.4).

Pins the registration half of the Pillar C expansion server
``geo-embedding-search`` against its two sources of truth:

* Req 2.1 / 11.3 - :meth:`GeoEmbeddingSearchServer.catalog_entries` registers one
  Resource Catalog entry per exposed tool (``store_embedding``,
  ``search_embeddings``), each naming ``geo-embedding-search`` as the
  ``provider_server`` (Req 11.3) and carrying name, pillar, capability
  description, and ``Openness_Tier`` (Req 2.1). The bundled vector stores
  (OpenSearch, LanceDB) are openly licensed, so every entry's tier is
  :attr:`OpennessTier.OPEN`.
* Req 16.1 / 16.5 - :meth:`GeoEmbeddingSearchServer.required_credentials` declares
  the three Optional OpenSearch keys, so the startup credential guard never
  blocks (the default local store works without them).
* Req 16.1 - the in-code credential specs agree exactly with the
  ``geo-embedding-search`` entry in the repo-root ``bundle-manifest.json``, so the
  manifest and the server can never silently drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from geo_common.models import CredentialClassification, OpennessTier

from geo_embedding_search.server import (
    INSTALL_COMMAND,
    OPENSEARCH_PASSWORD_KEY,
    OPENSEARCH_URL_KEY,
    OPENSEARCH_USERNAME_KEY,
    GeoEmbeddingSearchServer,
)

_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "bundle-manifest.json"

EXPECTED_TOOLS = {"store_embedding", "search_embeddings"}
EXPECTED_KEYS = {
    OPENSEARCH_URL_KEY,
    OPENSEARCH_USERNAME_KEY,
    OPENSEARCH_PASSWORD_KEY,
}


def _manifest_entry() -> Dict[str, Any]:
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    servers = {s["name"]: s for s in manifest["servers"]}
    assert "geo-embedding-search" in servers
    return servers["geo-embedding-search"]


# --- Req 2.1 / 11.3: catalog registration ----------------------------------


def test_catalog_entries_cover_every_tool() -> None:
    """Req 2.1 / 11.3: one catalog entry per exposed tool, no orphan/missing."""
    server = GeoEmbeddingSearchServer()
    entries = server.catalog_entries()

    assert {entry.name for entry in entries} == EXPECTED_TOOLS
    assert {entry.name for entry in entries} == set(server.tool_names())


def test_catalog_entries_are_open_pillar_c_and_self_provided() -> None:
    """Req 2.1 / 11.3: each entry is Pillar C, OPEN, and names this server."""
    server = GeoEmbeddingSearchServer()

    for entry in server.catalog_entries():
        assert entry.pillar == "C"
        assert entry.provider_server == "geo-embedding-search"
        assert entry.openness_tier is OpennessTier.OPEN
        assert 1 <= len(entry.capability_description) <= 500


# --- Req 16.1 / 16.5: credential registration -------------------------------


def test_required_credentials_declare_optional_opensearch_keys() -> None:
    """Req 16.1: the three OpenSearch keys are declared and all Optional."""
    server = GeoEmbeddingSearchServer()
    specs = server.required_credentials()

    assert {spec.mcp_json_key for spec in specs} == EXPECTED_KEYS
    assert all(
        spec.classification is CredentialClassification.OPTIONAL for spec in specs
    )


def test_server_starts_with_no_credentials_configured() -> None:
    """Req 16.5: only Optional credentials -> startup is never blocked."""
    server = GeoEmbeddingSearchServer()
    server.start(configured_keys=[])
    assert server.started is True


def test_required_credentials_match_manifest() -> None:
    """Req 16.1: in-code credential specs agree exactly with bundle-manifest.json."""
    specs = GeoEmbeddingSearchServer().required_credentials()
    spec_pairs = {(s.mcp_json_key, s.classification.value) for s in specs}
    manifest_pairs = {
        (c["key"], c["classification"]) for c in _manifest_entry()["credentials"]
    }
    assert spec_pairs == manifest_pairs


def test_manifest_install_command_matches_server() -> None:
    """Req 2.6: the manifest ``uvx`` command matches the server constant."""
    assert _manifest_entry()["uvx"] == INSTALL_COMMAND == "uvx geo-embedding-search"

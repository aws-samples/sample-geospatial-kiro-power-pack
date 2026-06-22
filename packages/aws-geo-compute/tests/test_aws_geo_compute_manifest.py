"""Manifest/code credential-consistency test for ``aws-geo-compute``.

The server's ``required_credentials()`` and the ``aws-geo-compute`` block in the
repo-root ``bundle-manifest.json`` must agree on credential keys and
classifications. This pins the reconciliation that replaced the invented,
manifest-absent ``AWS_GEO_COMPUTE_CONNECTION`` key with the three standard AWS
keys (all Optional) and flipped the manifest classification Required -> Optional.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from geo_common.models import CredentialClassification

from aws_geo_compute.server import AWS_CREDENTIAL_KEYS, AwsGeoComputeServer

_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "bundle-manifest.json"


@pytest.fixture(scope="module")
def compute_entry() -> Dict[str, Any]:
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    servers = {s["name"]: s for s in manifest["servers"]}
    assert "aws-geo-compute" in servers
    return servers["aws-geo-compute"]


def test_required_credentials_match_manifest(compute_entry: Dict[str, Any]) -> None:
    specs = AwsGeoComputeServer().required_credentials()
    code_pairs = {(s.mcp_json_key, s.classification.value) for s in specs}
    manifest_pairs = {
        (c["key"], c["classification"]) for c in compute_entry["credentials"]
    }
    assert code_pairs == manifest_pairs


def test_uses_standard_aws_keys_all_optional(compute_entry: Dict[str, Any]) -> None:
    specs = AwsGeoComputeServer().required_credentials()
    assert {s.mcp_json_key for s in specs} == set(AWS_CREDENTIAL_KEYS)
    assert set(AWS_CREDENTIAL_KEYS) == {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
    }
    assert all(
        s.classification is CredentialClassification.OPTIONAL for s in specs
    )
    assert all(c["classification"] == "Optional" for c in compute_entry["credentials"])


def test_server_starts_without_credentials() -> None:
    """Optional AWS keys never block startup; planning runs offline (Req 16.5)."""
    server = AwsGeoComputeServer()
    server.start(configured_keys=[])
    assert server.started is True

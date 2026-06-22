"""Catalog/registration and credential tests for ``geo-ops`` (Req 2.1, 16.5).

``geo-ops`` implements its full generic-GIS surface natively on
PyProj/Shapely/GeoPandas - including ``buffer`` and ``convex_hull``, which were
previously delegated to an external server. These tests assert that:

* every native capability (including ``buffer``/``convex_hull``) is registered
  both as an MCP tool and as a Resource_Catalog entry naming ``geo-ops`` as the
  provider (Req 2.1);
* ``geo-ops`` is open and credential-free, so it declares no credentials and
  starts without any configured keys (Req 16.5).

The per-operation behavior (happy paths and validation guards) lives in
``test_geo_ops_units.py``.
"""

from __future__ import annotations

from geo_common.models import CredentialSpec, OpennessTier

from geo_ops.server import GeoOpsServer


EXPECTED_CAPABILITIES = {
    "transform_crs",
    "validate_geometry",
    "spatial_join",
    "overlay",
    "buffer",
    "convex_hull",
}


# --- tool + catalog registration (Req 2.1) --------------------------------

def test_all_native_capabilities_are_registered_as_tools():
    """Every native capability is registered as an invocable MCP tool."""
    server = GeoOpsServer()
    assert EXPECTED_CAPABILITIES <= set(server.tools)


def test_catalog_lists_every_native_capability():
    """Req 2.1: the catalog lists each native geo-ops capability."""
    server = GeoOpsServer()
    names = {e.name for e in server.catalog_entries()}
    assert EXPECTED_CAPABILITIES <= names


def test_tools_and_catalog_agree():
    """Each registered tool has a matching catalog entry and vice versa."""
    server = GeoOpsServer()
    catalog_names = {e.name for e in server.catalog_entries()}
    assert set(server.tools) == catalog_names


def test_every_catalog_entry_is_open_and_provided_by_geo_ops():
    """Each entry names geo-ops as provider and is open tier (Req 2.1)."""
    server = GeoOpsServer()
    for entry in server.catalog_entries():
        assert entry.provider_server == "geo-ops"
        assert entry.pillar == "B"
        assert entry.openness_tier is OpennessTier.OPEN
        assert entry.installed is True
        assert entry.capability_description.strip() != ""


# --- credentials (Req 16.5; geo-ops is open/credential-free) --------------

def test_geo_ops_requires_no_credentials_and_starts():
    """geo-ops is open and credential-free, so it declares none and starts."""
    server = GeoOpsServer()
    creds = server.required_credentials()
    assert creds == []
    assert all(isinstance(c, CredentialSpec) for c in creds)
    # No Required credential => the startup guard lets it start (Req 16.5).
    server.start(configured_keys=[])
    assert server.started is True

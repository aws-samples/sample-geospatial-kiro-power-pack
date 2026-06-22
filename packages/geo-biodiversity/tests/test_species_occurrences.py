"""Tests for ``geo-biodiversity`` ``species_occurrences`` (Requirements 7.7, 7.12).

These are example-based tests driven through an ``httpx`` mock transport so the
GBIF connector is exercised without real network I/O. They cover:

* returning occurrence records with their coordinate/taxon/date metadata, and
  capping the per-response result at the configured maximum (Requirement 7.7);
* malformed-parameter rejection (bbox, limit, taxon) as an ``Error_Taxonomy``
  validation error raised before any HTTP call (Requirement 7.12);
* GBIF query building (WKT polygon geometry, hasCoordinate, scientificName);
* taxonomy-classified propagation of an upstream failure; and
* the server scaffold, catalog registration, and credential specs.
"""

from __future__ import annotations

from typing import Any, Dict, List

import httpx
import pytest

from geo_common.errors import ErrorCategory, UpstreamError, ValidationError
from geo_common.http import HttpClient
from geo_common.models import CredentialClassification, OpennessTier
from geo_common.retry import RetryPolicy

from geo_biodiversity.occurrences import GBIFSource, species_occurrences
from geo_biodiversity.server import INSTALL_COMMAND, GeoBiodiversityServer
from geo_biodiversity.validation import (
    DEFAULT_MAX_RECORDS,
    validate_and_cap_limit,
    validate_bbox,
    validate_taxon,
)

GBIF_HOST = "api.gbif.org"

# A small extent near Berlin.
SMALL_BBOX = (13.40, 52.50, 13.41, 52.51)


def _gbif_record(key: int) -> Dict[str, Any]:
    """A minimal but realistic GBIF occurrence result."""
    return {
        "key": key,
        "scientificName": "Vulpes vulpes",
        "decimalLongitude": 13.405,
        "decimalLatitude": 52.505,
        "eventDate": "2023-05-01",
        "datasetKey": "abc-123",
    }


def _gbif_payload(n: int) -> Dict[str, Any]:
    return {"count": n, "results": [_gbif_record(i) for i in range(n)]}


async def _no_sleep(_seconds: float) -> None:
    return None


def _make_handler(*, gbif=None, calls=None):
    """Build a mock-transport handler routing GBIF requests to canned data.

    iNaturalist (the second default source) returns an empty result set so
    these GBIF-focused tests see only GBIF records from the default server.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if calls is not None:
            calls.append(request)
        if host == GBIF_HOST:
            return gbif(request) if gbif else httpx.Response(
                200, json=_gbif_payload(3)
            )
        if host == "api.inaturalist.org":
            return httpx.Response(200, json={"results": []})
        return httpx.Response(404)  # pragma: no cover - unexpected host

    return handler


def _server(handler, *, max_records=None, sources=None) -> GeoBiodiversityServer:
    client = HttpClient(
        RetryPolicy(max_attempts=1),
        transport=httpx.MockTransport(handler),
        sleep=_no_sleep,
    )
    kwargs: Dict[str, Any] = {"http": client}
    if max_records is not None:
        kwargs["max_records"] = max_records
    if sources is not None:
        kwargs["sources"] = sources
    return GeoBiodiversityServer(**kwargs)


# --- Requirement 7.7: returns records with metadata ------------------------


async def test_returns_occurrence_records_with_metadata():
    server = _server(_make_handler())
    try:
        records = await server.species_occurrences(bbox=SMALL_BBOX)
    finally:
        await server.aclose()

    assert len(records) == 3
    for record in records:
        assert record["source"] == "gbif"
        assert record["scientific_name"] == "Vulpes vulpes"
        assert record["longitude"] == 13.405
        assert record["latitude"] == 52.505
        assert record["event_date"] == "2023-05-01"
        assert record["id"] is not None
        # The full source entry is retained under properties.
        assert record["properties"]["datasetKey"] == "abc-123"


async def test_empty_result_set_is_supported():
    server = _server(_make_handler(gbif=lambda r: httpx.Response(200, json=_gbif_payload(0))))
    try:
        records = await server.species_occurrences(bbox=SMALL_BBOX)
    finally:
        await server.aclose()
    assert records == []


# --- Requirement 7.7: per-response record cap ------------------------------


async def test_caps_results_at_configured_maximum():
    # The source returns more than the (small) configured maximum; the result
    # must be clamped down to the cap.
    server = _server(
        _make_handler(gbif=lambda r: httpx.Response(200, json=_gbif_payload(50))),
        max_records=10,
    )
    try:
        records = await server.species_occurrences(bbox=SMALL_BBOX, limit=10)
    finally:
        await server.aclose()
    assert len(records) == 10


async def test_oversized_limit_is_clamped_to_default_maximum():
    # A requested limit above the default 10,000 maximum is clamped, not honored.
    assert validate_and_cap_limit(DEFAULT_MAX_RECORDS + 5_000) == DEFAULT_MAX_RECORDS
    assert validate_and_cap_limit(50) == 50


# --- Requirement 7.12: malformed parameters --------------------------------


@pytest.mark.parametrize(
    "bad_bbox",
    [
        (13.40, 52.50, 13.41),  # too few ordinates
        (13.40, 52.50, 13.41, 52.51, 1.0),  # too many ordinates
        (200.0, 52.50, 200.1, 52.51),  # longitude out of range
        (13.40, 95.0, 13.41, 96.0),  # latitude out of range
        (13.41, 52.50, 13.40, 52.51),  # inverted longitude (min > max)
        (13.40, 52.51, 13.41, 52.50),  # inverted latitude (min > max)
        ("13.40", 52.50, 13.41, 52.51),  # non-numeric ordinate
        "13.40,52.50,13.41,52.51",  # a string, not a sequence of numbers
    ],
)
def test_malformed_bbox_raises_validation_error(bad_bbox):
    with pytest.raises(ValidationError) as exc:
        validate_bbox(bad_bbox)
    assert exc.value.category is ErrorCategory.VALIDATION
    assert exc.value.detail["parameter"] == "bbox"


@pytest.mark.parametrize("bad_limit", [0, -5, True, 1.5, "10"])
def test_invalid_limit_raises_validation_error(bad_limit):
    with pytest.raises(ValidationError) as exc:
        validate_and_cap_limit(bad_limit)
    assert exc.value.detail["parameter"] == "limit"


@pytest.mark.parametrize("bad_taxon", ["", "   ", 123, []])
def test_invalid_taxon_raises_validation_error(bad_taxon):
    with pytest.raises(ValidationError) as exc:
        validate_taxon(bad_taxon)
    assert exc.value.detail["parameter"] == "taxon"


def test_valid_taxon_is_stripped_and_none_is_allowed():
    assert validate_taxon(None) is None
    assert validate_taxon("  Vulpes vulpes ") == "Vulpes vulpes"


async def test_validation_happens_before_any_network_call():
    calls: List[httpx.Request] = []
    server = _server(_make_handler(calls=calls))
    try:
        with pytest.raises(ValidationError):
            await server.species_occurrences(bbox=(200.0, 0.0, 201.0, 1.0))
    finally:
        await server.aclose()
    assert calls == []


# --- GBIF query building ----------------------------------------------------


def test_gbif_params_include_polygon_and_has_coordinate():
    params = GBIFSource().build_params(SMALL_BBOX, None, DEFAULT_MAX_RECORDS)
    assert params["hasCoordinate"] == "true"
    assert params["geometry"].startswith("POLYGON((")
    # The polygon closes on its first vertex.
    assert params["geometry"].endswith("13.4 52.5))")
    # GBIF page limit caps the per-request size.
    assert params["limit"] <= 300
    assert "scientificName" not in params


def test_gbif_params_include_taxon_filter_when_present():
    params = GBIFSource().build_params(SMALL_BBOX, "Vulpes vulpes", 50)
    assert params["scientificName"] == "Vulpes vulpes"


async def test_taxon_filter_is_forwarded_to_gbif():
    captured: List[httpx.Request] = []
    server = _server(_make_handler(calls=captured))
    try:
        await server.species_occurrences(bbox=SMALL_BBOX, taxon="Vulpes vulpes")
    finally:
        await server.aclose()
    assert captured, "GBIF was queried"
    assert "scientificName=Vulpes+vulpes" in str(captured[0].url) or "scientificName=Vulpes%20vulpes" in str(captured[0].url)


# --- upstream failure propagation ------------------------------------------


async def test_upstream_failure_propagates_as_taxonomy_error():
    def gbif_500(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    server = _server(_make_handler(gbif=gbif_500))
    try:
        with pytest.raises(UpstreamError) as exc:
            await server.species_occurrences(bbox=SMALL_BBOX)
    finally:
        await server.aclose()
    assert exc.value.category is ErrorCategory.UPSTREAM


async def test_non_json_response_is_mapped_to_upstream():
    def gbif_bad(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    server = _server(_make_handler(gbif=gbif_bad))
    try:
        with pytest.raises(UpstreamError):
            await server.species_occurrences(bbox=SMALL_BBOX)
    finally:
        await server.aclose()


# --- connector-level direct usage ------------------------------------------


async def test_connector_returns_dicts():
    client = HttpClient(
        RetryPolicy(max_attempts=1),
        transport=httpx.MockTransport(_make_handler()),
        sleep=_no_sleep,
    )
    try:
        records = await species_occurrences(
            bbox=SMALL_BBOX, http=client, sources=[GBIFSource()]
        )
    finally:
        await client.aclose()
    assert all(isinstance(r, dict) for r in records)
    assert len(records) == 3


# --- server scaffold --------------------------------------------------------


def test_server_registers_species_occurrences_tool():
    server = GeoBiodiversityServer()
    assert server.server_name == "geo-biodiversity"
    assert server.pillar == "A"
    assert "species_occurrences" in server.tool_names()


# --- Requirement 2.1 / 11.3: catalog registration -------------------------


def test_catalog_entry_names_geo_biodiversity_as_open_provider():
    server = GeoBiodiversityServer()
    entries = server.catalog_entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.name == "species_occurrences"
    assert entry.pillar == "A"
    assert entry.openness_tier is OpennessTier.OPEN
    assert entry.provider_server == "geo-biodiversity"
    assert entry.install_command == INSTALL_COMMAND
    assert 1 <= len(entry.capability_description) <= 500


# --- Requirement 16.1 / 16.5: credential specs -----------------------------


def test_optional_credential_declared_and_starts():
    server = GeoBiodiversityServer()
    specs = server.required_credentials()
    keys = {spec.mcp_json_key for spec in specs}
    assert keys == {"INATURALIST_TOKEN", "IUCN_TOKEN"}
    assert all(
        spec.classification is CredentialClassification.OPTIONAL for spec in specs
    )
    # No Required credential -> startup is never blocked (Req 16.5).
    server.start(configured_keys=[])
    assert server.started is True

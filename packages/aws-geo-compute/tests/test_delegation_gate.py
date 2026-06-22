"""Unit tests for the delegation gate and server gating (task 17.1).

Covers the 60-second delegation deadline and failure mapping (Requirement
12.8) and the server's input validation, catalog/credential declaration, and
in-process-vs-delegate routing (Requirements 12.4, 12.5). All async tests run
under the repo-wide ``asyncio_mode = "auto"``.
"""

from __future__ import annotations

import asyncio

import pytest

from geo_common.errors import (
    AuthenticationError,
    ErrorCategory,
    NetworkError,
    UpstreamError,
    ValidationError,
)
from geo_common.models import CredentialClassification

from aws_geo_compute.delegation import (
    DelegationGate,
    DelegationReceipt,
    DelegationRequest,
)
from aws_geo_compute.engine_selection import GB, AccessPattern, Engine
from aws_geo_compute.server import AwsGeoComputeServer


def _request(**kwargs) -> DelegationRequest:
    base = dict(
        task_type="distributed-spatial-join",
        engine="emr-sedona",
        dataset_size_bytes=200 * GB,
        required_memory_bytes=0,
    )
    base.update(kwargs)
    return DelegationRequest(**base)


# --- Delegation gate: success ----------------------------------------------


async def test_gate_returns_accepted_receipt() -> None:
    async def acceptor(req: DelegationRequest) -> DelegationReceipt:
        return DelegationReceipt(accepted=True, job_id="job-123")

    gate = DelegationGate(acceptor)
    receipt = await gate.delegate(_request())
    assert receipt.accepted is True
    assert receipt.job_id == "job-123"


# --- Delegation gate: 60s non-acceptance -> NETWORK (Req 12.8) -------------


async def test_gate_times_out_when_not_accepted_in_time() -> None:
    async def slow_acceptor(req: DelegationRequest) -> DelegationReceipt:
        await asyncio.sleep(1.0)
        return DelegationReceipt(accepted=True, job_id="late")

    # Short deadline to exercise the timeout path deterministically.
    gate = DelegationGate(slow_acceptor, timeout_s=0.05)
    with pytest.raises(NetworkError) as exc_info:
        await gate.delegate(_request())
    assert exc_info.value.category is ErrorCategory.NETWORK
    assert "within" in str(exc_info.value)


def test_gate_default_timeout_is_60s() -> None:
    async def acceptor(req):  # pragma: no cover - not invoked
        return DelegationReceipt(accepted=True)

    assert DelegationGate(acceptor).timeout_s == 60.0


# --- Delegation gate: rejection -> UPSTREAM (Req 12.8) ---------------------


async def test_gate_maps_rejection_to_upstream_retaining_detail() -> None:
    async def rejecting(req: DelegationRequest) -> DelegationReceipt:
        return DelegationReceipt(accepted=False, detail="no capacity")

    gate = DelegationGate(rejecting)
    with pytest.raises(UpstreamError) as exc_info:
        await gate.delegate(_request())
    err = exc_info.value
    assert err.category is ErrorCategory.UPSTREAM
    assert err.original == "no capacity"


# --- Delegation gate: raised errors mapped onto the taxonomy ---------------


async def test_gate_passes_through_geoerror() -> None:
    boom = AuthenticationError("bad AWS creds", source="aws-geo-compute")

    async def raising(req: DelegationRequest) -> DelegationReceipt:
        raise boom

    gate = DelegationGate(raising)
    with pytest.raises(AuthenticationError) as exc_info:
        await gate.delegate(_request())
    assert exc_info.value is boom


async def test_gate_maps_unexpected_error_to_upstream() -> None:
    async def raising(req: DelegationRequest) -> DelegationReceipt:
        raise RuntimeError("kaboom")

    gate = DelegationGate(raising)
    with pytest.raises(UpstreamError) as exc_info:
        await gate.delegate(_request())
    err = exc_info.value
    assert err.category is ErrorCategory.UPSTREAM
    assert err.original is not None and "kaboom" in err.original


def test_gate_rejects_bad_construction() -> None:
    async def acceptor(req):  # pragma: no cover - not invoked
        return DelegationReceipt(accepted=True)

    with pytest.raises(TypeError):
        DelegationGate(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DelegationGate(acceptor, timeout_s=0)


# --- Server: plan_execution validation (Req 12.4, 12.5) --------------------


def test_plan_execution_returns_plan() -> None:
    server = AwsGeoComputeServer()
    plan = server.plan_execution(
        dataset_size_bytes=10 * GB,
        access_pattern=AccessPattern.SINGLE_USER_ANALYTICAL,
    )
    assert plan.engine is Engine.DUCKDB
    assert plan.delegate is True  # 10 GB > 5 GB


def test_plan_execution_accepts_string_access_pattern() -> None:
    server = AwsGeoComputeServer()
    plan = server.plan_execution(
        dataset_size_bytes=100,
        access_pattern="single-user-analytical",  # type: ignore[arg-type]
    )
    assert plan.engine is Engine.GEOPANDAS


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(dataset_size_bytes=-1, access_pattern=AccessPattern.AD_HOC_OVER_S3),
        dict(dataset_size_bytes=1, access_pattern="nonsense"),
        dict(
            dataset_size_bytes=1,
            access_pattern=AccessPattern.AD_HOC_OVER_S3,
            required_memory_bytes=-5,
        ),
        dict(dataset_size_bytes="big", access_pattern=AccessPattern.AD_HOC_OVER_S3),
    ],
)
def test_plan_execution_rejects_malformed_input(kwargs) -> None:
    server = AwsGeoComputeServer()
    with pytest.raises(ValidationError) as exc_info:
        server.plan_execution(**kwargs)
    assert exc_info.value.category is ErrorCategory.VALIDATION


# --- Server: execute_or_delegate routing (Req 12.4, 12.5, 12.8) ------------


async def test_execute_runs_in_process_when_under_threshold() -> None:
    server = AwsGeoComputeServer()
    outcome = await server.execute_or_delegate(
        task_type="local-op",
        dataset_size_bytes=1 * GB,
        access_pattern=AccessPattern.SINGLE_USER_ANALYTICAL,
        required_memory_bytes=1 * GB,
    )
    assert outcome.delegated is False
    assert outcome.receipt is None
    assert outcome.plan.engine is Engine.DUCKDB


async def test_execute_delegates_when_over_threshold() -> None:
    seen = {}

    async def acceptor(req: DelegationRequest) -> DelegationReceipt:
        seen["req"] = req
        return DelegationReceipt(accepted=True, job_id="job-9")

    server = AwsGeoComputeServer(delegation_acceptor=acceptor)
    outcome = await server.execute_or_delegate(
        task_type="distributed-spatial-join",
        dataset_size_bytes=200 * GB,
        access_pattern=AccessPattern.SINGLE_USER_ANALYTICAL,
    )
    assert outcome.delegated is True
    assert outcome.receipt is not None and outcome.receipt.job_id == "job-9"
    assert seen["req"].engine == "emr-sedona"


async def test_delegation_required_but_no_target_aborts() -> None:
    """Delegation required but no acceptor configured -> taxonomy error, no silent local run."""
    server = AwsGeoComputeServer()  # no delegation acceptor
    with pytest.raises(UpstreamError) as exc_info:
        await server.execute_or_delegate(
            task_type="distributed-spatial-join",
            dataset_size_bytes=200 * GB,
            access_pattern=AccessPattern.SINGLE_USER_ANALYTICAL,
        )
    assert exc_info.value.category is ErrorCategory.UPSTREAM


async def test_delegation_timeout_propagates_from_server() -> None:
    async def slow(req: DelegationRequest) -> DelegationReceipt:
        await asyncio.sleep(1.0)
        return DelegationReceipt(accepted=True)

    # Inject a short-deadline gate via a custom acceptor + monkeypatched gate.
    server = AwsGeoComputeServer(delegation_acceptor=slow)
    server._gate = DelegationGate(slow, timeout_s=0.05, error_mapper=server.map_error)
    with pytest.raises(NetworkError):
        await server.execute_or_delegate(
            task_type="distributed-spatial-join",
            dataset_size_bytes=200 * GB,
            access_pattern=AccessPattern.SINGLE_USER_ANALYTICAL,
        )


# --- Server: catalog, credentials, startup ---------------------------------


def test_server_starts_without_credentials() -> None:
    """Optional AWS credential never blocks startup (Req 16.5)."""
    server = AwsGeoComputeServer()
    server.start(configured_keys=[])
    assert server.started is True


def test_only_optional_aws_credentials_declared() -> None:
    server = AwsGeoComputeServer()
    specs = server.required_credentials()
    # The three standard AWS keys, all Optional (planning runs offline; the
    # keys are enforced at submit/status time).
    assert {s.mcp_json_key for s in specs} == {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
    }
    assert all(
        s.classification is CredentialClassification.OPTIONAL for s in specs
    )


def test_catalog_entry_registered() -> None:
    server = AwsGeoComputeServer()
    entries = server.catalog_entries()
    # Every registered tool is advertised as a catalog capability (Req 2.1).
    assert {e.name for e in entries} == {
        "plan_execution",
        "select_target",
        "submit",
        "status",
    }
    assert all(e.provider_server == "aws-geo-compute" for e in entries)


def test_catalog_matches_registered_tools() -> None:
    """The catalog and the registered tool set name the same capabilities."""
    server = AwsGeoComputeServer()
    assert {e.name for e in server.catalog_entries()} == set(server.tool_names())


def test_job_capabilities_marked_uninstalled_without_backend() -> None:
    """submit/status are advertised but flagged not-installed with no backend."""
    server = AwsGeoComputeServer()
    by_name = {e.name: e for e in server.catalog_entries()}
    assert by_name["submit"].installed is False
    assert by_name["status"].installed is False
    # select_target is pure/deterministic, so always available.
    assert by_name["select_target"].installed is True


def test_plan_execution_is_registered_tool() -> None:
    server = AwsGeoComputeServer()
    assert "plan_execution" in server.tool_names()

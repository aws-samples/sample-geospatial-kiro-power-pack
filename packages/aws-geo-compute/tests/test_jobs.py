"""Unit tests for job submission and status (task 17.3).

Covers the deterministic task-type -> target mapping and unsupported-type
rejection (Requirements 13.1, 13.6), job submission returning a unique id with
submission-failure handling (Requirements 13.2, 13.7), and status reporting for
succeeded/failed/unknown jobs (Requirements 13.3, 13.4, 13.5, 13.8). The job
backend is a controllable stub, so no AWS is required. Async tests run under the
repo-wide ``asyncio_mode = "auto"``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from geo_common.errors import (
    AuthenticationError,
    ErrorCategory,
    NetworkError,
    NotFoundError,
    UpstreamError,
    ValidationError,
)

from aws_geo_compute.jobs import (
    TASK_TARGET_MAP,
    ComputeJobRecord,
    ExecutionTarget,
    JobManager,
    JobState,
    JobStatus,
    JobStatusView,
    JobSubmission,
    select_target,
)
from aws_geo_compute.server import AwsGeoComputeServer


# --- A controllable in-memory backend --------------------------------------


class StubBackend:
    """A deterministic :class:`JobBackend` for tests.

    ``submit`` records the call and (optionally) raises or sleeps to exercise
    failure/timeout paths; ``poll`` returns the queued :class:`JobState` per id.
    """

    def __init__(
        self,
        *,
        submit_error: Optional[Exception] = None,
        submit_delay: float = 0.0,
        poll_delay: float = 0.0,
    ) -> None:
        self.submit_error = submit_error
        self.submit_delay = submit_delay
        self.poll_delay = poll_delay
        self.submitted: List[Dict[str, Any]] = []
        self.states: Dict[str, JobState] = {}
        self._default_state = JobState(status=JobStatus.QUEUED)

    async def submit(
        self, *, task_type: str, target: ExecutionTarget, payload: Dict[str, Any]
    ) -> str:
        if self.submit_delay:
            await asyncio.sleep(self.submit_delay)
        if self.submit_error is not None:
            raise self.submit_error
        self.submitted.append(
            {"task_type": task_type, "target": target, "payload": payload}
        )
        return "provider-handle-%d" % len(self.submitted)

    async def poll(
        self,
        *,
        job_id: str,
        task_type: str,
        target: ExecutionTarget,
        provider_handle: Optional[str] = None,
    ) -> JobState:
        if self.poll_delay:
            await asyncio.sleep(self.poll_delay)
        return self.states.get(job_id, self._default_state)


# --- select_target (Req 13.1, 13.6; Property 18) ---------------------------


@pytest.mark.parametrize("task_type, expected", list(TASK_TARGET_MAP.items()))
def test_select_target_maps_supported_types(
    task_type: str, expected: ExecutionTarget
) -> None:
    assert select_target(task_type) is expected


def test_select_target_is_deterministic() -> None:
    assert select_target("model-inference") is select_target("model-inference")


@pytest.mark.parametrize("bad", ["", "no-such-task", "BULK-COG-CONVERSION"])
def test_select_target_rejects_unsupported(bad: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        select_target(bad)
    assert exc_info.value.category is ErrorCategory.VALIDATION


def test_select_target_rejects_non_string() -> None:
    with pytest.raises(ValidationError):
        select_target(None)  # type: ignore[arg-type]


# --- submit (Req 13.2, 13.7) ----------------------------------------------


async def test_submit_returns_submission_with_target() -> None:
    mgr = JobManager(StubBackend())
    sub = await mgr.submit(task_type="bulk-cog-conversion", payload={"k": "v"})
    assert isinstance(sub, JobSubmission)
    assert sub.target is ExecutionTarget.BATCH
    assert sub.job_id
    record = mgr.get_record(sub.job_id)
    assert isinstance(record, ComputeJobRecord)
    assert record.status is JobStatus.QUEUED


async def test_submit_job_ids_are_unique() -> None:
    mgr = JobManager(StubBackend())
    ids = [
        (await mgr.submit(task_type="model-inference")).job_id for _ in range(50)
    ]
    assert len(set(ids)) == len(ids)


async def test_submit_unsupported_task_type_is_validation_error() -> None:
    mgr = JobManager(StubBackend())
    with pytest.raises(ValidationError):
        await mgr.submit(task_type="not-supported")


async def test_submit_failure_returns_error_and_no_job_id() -> None:
    """Submission failure -> taxonomy error and no persisted record (Req 13.7)."""
    backend = StubBackend(submit_error=RuntimeError("Batch rejected job"))
    mgr = JobManager(backend)
    with pytest.raises(UpstreamError) as exc_info:
        await mgr.submit(task_type="bulk-cog-conversion")
    assert exc_info.value.category is ErrorCategory.UPSTREAM
    assert exc_info.value.original is not None
    assert "Batch rejected job" in exc_info.value.original
    # No record persisted for the failed submission.
    assert mgr._records == {}


async def test_submit_timeout_is_failure_with_no_job_id() -> None:
    backend = StubBackend(submit_delay=1.0)
    mgr = JobManager(backend, submit_timeout_s=0.05)
    with pytest.raises(NetworkError):
        await mgr.submit(task_type="bulk-cog-conversion")
    assert mgr._records == {}


# --- status (Req 13.3, 13.4, 13.5, 13.8) -----------------------------------


async def test_status_unknown_id_is_not_found() -> None:
    mgr = JobManager(StubBackend())
    with pytest.raises(NotFoundError) as exc_info:
        await mgr.status(job_id="does-not-exist")
    assert exc_info.value.category is ErrorCategory.NOT_FOUND


async def test_status_queued_returns_view() -> None:
    backend = StubBackend()
    mgr = JobManager(backend)
    sub = await mgr.submit(task_type="windowed-raster-op")
    view = await mgr.status(job_id=sub.job_id)
    assert isinstance(view, JobStatusView)
    assert view.status is JobStatus.QUEUED
    assert view.output_s3_uri is None


async def test_status_succeeded_returns_s3_output_location() -> None:
    backend = StubBackend()
    mgr = JobManager(backend)
    sub = await mgr.submit(task_type="bulk-cog-conversion")
    backend.states[sub.job_id] = JobState(
        status=JobStatus.SUCCEEDED, output_s3_uri="s3://amzn-s3-demo-bucket/out/result.tif"
    )
    view = await mgr.status(job_id=sub.job_id)
    assert view.status is JobStatus.SUCCEEDED
    assert view.output_s3_uri == "s3://amzn-s3-demo-bucket/out/result.tif"


async def test_status_failed_returns_error_and_failure_detail() -> None:
    """Failed job -> taxonomy error carrying the failure detail (Req 13.5)."""
    backend = StubBackend()
    mgr = JobManager(backend)
    sub = await mgr.submit(task_type="distributed-spatial-join")
    backend.states[sub.job_id] = JobState(
        status=JobStatus.FAILED, failure_detail="executor OOM at stage 3"
    )
    with pytest.raises(UpstreamError) as exc_info:
        await mgr.status(job_id=sub.job_id)
    assert exc_info.value.original == "executor OOM at stage 3"
    # The record retains the failure detail.
    record = mgr.get_record(sub.job_id)
    assert record is not None and record.failure_detail == "executor OOM at stage 3"


async def test_status_timeout_is_network_error() -> None:
    backend = StubBackend(poll_delay=1.0)
    mgr = JobManager(backend, status_timeout_s=0.05)
    sub = await mgr.submit(task_type="model-inference")
    with pytest.raises(NetworkError):
        await mgr.status(job_id=sub.job_id)


def test_manager_rejects_bad_construction() -> None:
    with pytest.raises(TypeError):
        JobManager(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        JobManager(StubBackend(), submit_timeout_s=0)
    with pytest.raises(ValueError):
        JobManager(StubBackend(), status_timeout_s=0)


# --- Server wiring (Req 13) ------------------------------------------------


def test_server_select_target_available_without_backend() -> None:
    server = AwsGeoComputeServer()
    assert server.select_target("foundation-model-embed") is ExecutionTarget.SAGEMAKER


def test_server_registers_job_tools() -> None:
    server = AwsGeoComputeServer()
    for tool in ("select_target", "submit", "status"):
        assert tool in server.tool_names()


async def test_server_submit_without_backend_or_credentials_is_auth_error(
    monkeypatch,
) -> None:
    """No backend and no AWS credentials -> AuthenticationError naming a key."""
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"):
        monkeypatch.delenv(key, raising=False)
    server = AwsGeoComputeServer()  # no job backend, no creds
    with pytest.raises(AuthenticationError) as exc_info:
        await server.submit(task_type="bulk-cog-conversion")
    assert exc_info.value.category is ErrorCategory.AUTHENTICATION
    assert "AWS_ACCESS_KEY_ID" in str(exc_info.value)


async def test_server_submit_without_backend_but_with_credentials_aborts(
    monkeypatch,
) -> None:
    """AWS creds present but no backend wired -> UpstreamError (not auth)."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    server = AwsGeoComputeServer()  # creds present, but no backend
    with pytest.raises(UpstreamError):
        await server.submit(task_type="bulk-cog-conversion")


async def test_server_submit_and_status_with_backend() -> None:
    backend = StubBackend()
    server = AwsGeoComputeServer(job_backend=backend)
    sub = await server.submit(task_type="bulk-cog-conversion")
    backend.states[sub.job_id] = JobState(
        status=JobStatus.SUCCEEDED, output_s3_uri="s3://amzn-s3-demo-bucket/out.tif"
    )
    view = await server.status(job_id=sub.job_id)
    assert view.status is JobStatus.SUCCEEDED
    assert view.output_s3_uri == "s3://amzn-s3-demo-bucket/out.tif"

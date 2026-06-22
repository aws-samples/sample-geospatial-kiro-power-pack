"""Layer-1 tests for the concrete AWS Batch job backend (no boto3, no account).

The :class:`~aws_geo_compute.backends.AwsBatchJobBackend` is exercised against a
**fake Batch client** injected into the backend, so these tests verify the
``submit_job`` request shape, the Batch-status -> :class:`JobStatus` mapping, the
S3 output-location convention, the provider-handle correlation (submit ->
record -> poll), the unsupported-target guard, the config guard, the auto-wiring
gate, and the botocore-error mapping **without** installing boto3 or contacting
AWS. End-to-end verification against a real Batch queue is a manual step
documented in ``docs/testing-workflows.md``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from geo_common.errors import (
    AuthenticationError,
    ErrorCategory,
    NotFoundError,
    UpstreamError,
)

from aws_geo_compute import (
    AWS_BATCH_JOB_DEFINITION_KEY,
    AWS_BATCH_JOB_QUEUE_KEY,
    AWS_GEO_COMPUTE_OUTPUT_S3_KEY,
    AwsBatchJobBackend,
    AwsGeoComputeServer,
    ExecutionTarget,
    JobStatus,
    default_job_backend,
)
from aws_geo_compute.backends import AWS_ACCESS_KEY_ID_KEY


# --- Fake AWS Batch client -------------------------------------------------


class _FakeBatchClient:
    """A minimal stand-in for boto3's Batch client.

    ``submit_job`` records the kwargs and returns a fixed jobId; ``describe_jobs``
    returns the canned ``describe`` response (or raises ``error`` from either
    call). ``submit_error`` raises on submit to exercise error mapping.
    """

    def __init__(
        self,
        *,
        job_id: str = "batch-abc123",
        describe: Optional[Dict[str, Any]] = None,
        submit_error: Optional[Exception] = None,
        describe_error: Optional[Exception] = None,
    ) -> None:
        self._job_id = job_id
        self._describe = describe
        self._submit_error = submit_error
        self._describe_error = describe_error
        self.submitted: List[Dict[str, Any]] = []
        self.described: List[List[str]] = []

    def submit_job(self, **kwargs: Any) -> Dict[str, Any]:
        if self._submit_error is not None:
            raise self._submit_error
        self.submitted.append(kwargs)
        return {"jobId": self._job_id, "jobName": kwargs.get("jobName")}

    def describe_jobs(self, *, jobs: List[str]) -> Dict[str, Any]:
        if self._describe_error is not None:
            raise self._describe_error
        self.described.append(list(jobs))
        return self._describe or {"jobs": []}


def _backend(client: _FakeBatchClient, **kwargs: Any) -> AwsBatchJobBackend:
    return AwsBatchJobBackend(
        batch_client=client,
        job_queue="my-queue",
        job_definition="my-def:1",
        **kwargs,
    )


# --- submit request shape --------------------------------------------------


async def test_submit_builds_batch_request_and_returns_job_id() -> None:
    client = _FakeBatchClient(job_id="batch-xyz")
    backend = _backend(client)
    handle = await backend.submit(
        task_type="bulk-cog-conversion",
        target=ExecutionTarget.BATCH,
        payload={"src": "s3://amzn-s3-demo-input/a.tif"},
    )
    assert handle == "batch-xyz"
    sent = client.submitted[0]
    assert sent["jobQueue"] == "my-queue"
    assert sent["jobDefinition"] == "my-def:1"
    assert sent["jobName"].startswith("geo-bulk-cog-conversion-")
    env = {e["name"]: e["value"] for e in sent["containerOverrides"]["environment"]}
    assert env["GEO_TASK_TYPE"] == "bulk-cog-conversion"
    assert json.loads(env["GEO_PAYLOAD"]) == {"src": "s3://amzn-s3-demo-input/a.tif"}


async def test_submit_includes_output_prefix_when_configured() -> None:
    client = _FakeBatchClient()
    backend = _backend(client, output_s3_prefix="s3://amzn-s3-demo-results/run1")
    await backend.submit(
        task_type="windowed-raster-op", target=ExecutionTarget.FARGATE, payload={}
    )
    env = {
        e["name"]: e["value"]
        for e in client.submitted[0]["containerOverrides"]["environment"]
    }
    assert env["GEO_OUTPUT_S3"] == "s3://amzn-s3-demo-results/run1"


async def test_fargate_target_is_supported_via_batch() -> None:
    client = _FakeBatchClient()
    handle = await _backend(client).submit(
        task_type="windowed-raster-op", target=ExecutionTarget.FARGATE, payload={}
    )
    assert handle == "batch-abc123"


@pytest.mark.parametrize(
    "target", [ExecutionTarget.EMR_SEDONA, ExecutionTarget.SAGEMAKER]
)
async def test_unsupported_target_raises_upstream(target: ExecutionTarget) -> None:
    client = _FakeBatchClient()
    with pytest.raises(UpstreamError) as exc_info:
        await _backend(client).submit(
            task_type="distributed-spatial-join", target=target, payload={}
        )
    assert target.value in str(exc_info.value)
    assert client.submitted == []  # nothing submitted


async def test_missing_queue_config_raises_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(AWS_BATCH_JOB_QUEUE_KEY, raising=False)
    client = _FakeBatchClient()
    backend = AwsBatchJobBackend(batch_client=client, job_definition="d")
    with pytest.raises(UpstreamError) as exc_info:
        await backend.submit(
            task_type="bulk-cog-conversion", target=ExecutionTarget.BATCH, payload={}
        )
    assert AWS_BATCH_JOB_QUEUE_KEY in str(exc_info.value)


# --- poll status mapping ---------------------------------------------------


@pytest.mark.parametrize(
    "batch_status, expected",
    [
        ("SUBMITTED", JobStatus.QUEUED),
        ("PENDING", JobStatus.QUEUED),
        ("RUNNABLE", JobStatus.QUEUED),
        ("STARTING", JobStatus.RUNNING),
        ("RUNNING", JobStatus.RUNNING),
        ("SUCCEEDED", JobStatus.SUCCEEDED),
        ("FAILED", JobStatus.FAILED),
    ],
)
async def test_poll_maps_batch_status(batch_status: str, expected: JobStatus) -> None:
    client = _FakeBatchClient(
        describe={"jobs": [{"jobId": "h", "status": batch_status}]}
    )
    state = await _backend(client).poll(
        job_id="m1",
        task_type="bulk-cog-conversion",
        target=ExecutionTarget.BATCH,
        provider_handle="h",
    )
    assert state.status is expected
    assert client.described == [["h"]]


async def test_poll_succeeded_reports_output_uri() -> None:
    client = _FakeBatchClient(describe={"jobs": [{"jobId": "h", "status": "SUCCEEDED"}]})
    backend = _backend(client, output_s3_prefix="s3://amzn-s3-demo-results/run1")
    state = await backend.poll(
        job_id="m1",
        task_type="bulk-cog-conversion",
        target=ExecutionTarget.BATCH,
        provider_handle="h",
    )
    assert state.status is JobStatus.SUCCEEDED
    assert state.output_s3_uri == "s3://amzn-s3-demo-results/run1/h/"


async def test_poll_failed_carries_status_reason() -> None:
    client = _FakeBatchClient(
        describe={"jobs": [{"jobId": "h", "status": "FAILED", "statusReason": "OOM"}]}
    )
    state = await _backend(client).poll(
        job_id="m1",
        task_type="bulk-cog-conversion",
        target=ExecutionTarget.BATCH,
        provider_handle="h",
    )
    assert state.status is JobStatus.FAILED
    assert state.failure_detail == "OOM"


async def test_poll_without_handle_raises_upstream() -> None:
    state_or_err = _backend(_FakeBatchClient())
    with pytest.raises(UpstreamError):
        await state_or_err.poll(
            job_id="m1",
            task_type="bulk-cog-conversion",
            target=ExecutionTarget.BATCH,
            provider_handle=None,
        )


async def test_poll_unknown_job_is_not_found() -> None:
    client = _FakeBatchClient(describe={"jobs": []})
    with pytest.raises(NotFoundError):
        await _backend(client).poll(
            job_id="m1",
            task_type="bulk-cog-conversion",
            target=ExecutionTarget.BATCH,
            provider_handle="gone",
        )


# --- botocore error mapping ------------------------------------------------


class _ClientError(Exception):
    def __init__(self, code: str, http_status: int) -> None:
        super().__init__(code)
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": http_status},
        }


async def test_submit_access_denied_maps_to_authentication() -> None:
    client = _FakeBatchClient(submit_error=_ClientError("AccessDeniedException", 403))
    with pytest.raises(AuthenticationError) as exc_info:
        await _backend(client).submit(
            task_type="bulk-cog-conversion", target=ExecutionTarget.BATCH, payload={}
        )
    assert exc_info.value.category is ErrorCategory.AUTHENTICATION
    assert exc_info.value.detail is not None
    assert exc_info.value.detail["mcp_json_key"] == AWS_ACCESS_KEY_ID_KEY


# --- auto-wiring gate ------------------------------------------------------


def test_default_job_backend_none_without_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(AWS_ACCESS_KEY_ID_KEY, raising=False)
    assert default_job_backend() is None


def test_default_job_backend_wired_with_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(AWS_ACCESS_KEY_ID_KEY, "AKIA-test")
    backend = default_job_backend()
    assert isinstance(backend, AwsBatchJobBackend)


# --- end-to-end through the server (provider-handle correlation) -----------


async def test_server_submit_status_via_batch_backend() -> None:
    """The manager records the Batch jobId from submit and passes it to poll."""
    client = _FakeBatchClient(
        job_id="batch-corr-1",
        describe={"jobs": [{"jobId": "batch-corr-1", "status": "SUCCEEDED"}]},
    )
    backend = _backend(client, output_s3_prefix="s3://amzn-s3-demo-output")
    server = AwsGeoComputeServer(job_backend=backend)

    submission = await server.submit(task_type="bulk-cog-conversion")
    view = await server.status(job_id=submission.job_id)

    assert view.status is JobStatus.SUCCEEDED
    assert view.output_s3_uri == "s3://amzn-s3-demo-output/batch-corr-1/"
    # poll was driven by the provider handle captured at submit time.
    assert client.described == [["batch-corr-1"]]

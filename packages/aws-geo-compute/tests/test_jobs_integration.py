"""Integration tests for compute submit/status/result (task 17.6).

These exercise the full :class:`~aws_geo_compute.server.AwsGeoComputeServer`
job surface end-to-end against a *mocked AWS* backend, covering the four paths
called out by the task:

* **submit -> status -> result** - a job is submitted, polled through its
  lifecycle, and on success exposes its output location in Amazon S3
  (Requirements 13.3, 13.4).
* **unsupported task type** - a task type outside the deterministic mapping is
  rejected with an ``Error_Taxonomy`` validation error naming it (Req 13.6).
* **unknown job id** - a status query for an id that was never submitted is a
  not-found error (Requirement 13.8).
* **failed job** - a job whose backend reports ``failed`` surfaces as an
  ``Error_Taxonomy`` error carrying the failure detail (Requirement 13.5).

The primary backend (:class:`SimulatedAwsComputeBackend`) is a faithful
in-memory stand-in that mimics the relevant AWS Batch / ECS Fargate / EMR /
SageMaker behavior (a ``submit`` that mints provider job ARNs and a ``poll``
that advances ``queued -> running -> succeeded|failed`` and yields an S3 output
URI on success). A second, equivalent test is provided against **moto** (real
boto3 S3 + an S3-object-backed lifecycle); it skips automatically when moto is
not installed in the current environment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from geo_common.errors import (
    ErrorCategory,
    NotFoundError,
    UpstreamError,
    ValidationError,
)

from aws_geo_compute.jobs import (
    ExecutionTarget,
    JobState,
    JobStatus,
)
from aws_geo_compute.server import AwsGeoComputeServer

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# A faithful, mocked AWS compute backend (no real AWS required)
# ---------------------------------------------------------------------------

#: S3 bucket the simulated backend writes job outputs to.
RESULT_BUCKET = "amzn-s3-demo-geo-compute-results"


class SimulatedAwsComputeBackend:
    """An in-memory :class:`~aws_geo_compute.jobs.JobBackend` simulating AWS.

    Mirrors the externally observable behavior of the real targets without any
    network or credentials:

    * :meth:`submit` mints a provider job handle that looks like the ARN/id the
      corresponding AWS service would return (AWS Batch job id, ECS task ARN,
      EMR step id, or SageMaker job ARN) and records the submission.
    * :meth:`poll` advances each job through ``queued -> running -> terminal``
      on successive polls (keyed by the manager's job id, exactly as the real
      backend is polled). The terminal state is ``succeeded`` by default, with
      an ``s3://`` output URI; jobs whose id is registered in ``fail_detail``
      terminate as ``failed`` carrying that detail.

    ``polls_until_terminal`` controls how many polls precede the terminal
    state, so a test can observe ``queued``/``running`` before the result.
    """

    def __init__(self, *, polls_until_terminal: int = 1) -> None:
        self.polls_until_terminal = max(0, int(polls_until_terminal))
        self.submitted: List[Dict[str, Any]] = []
        #: manager job_id -> failure detail, for jobs that should fail.
        self.fail_detail: Dict[str, str] = {}
        #: manager job_id -> number of times polled so far.
        self._poll_count: Dict[str, int] = {}

    @staticmethod
    def _provider_handle(target: ExecutionTarget, seq: int) -> str:
        """Return a provider-shaped job handle for the selected target."""
        if target is ExecutionTarget.BATCH:
            return "batch-job-%08x" % seq
        if target is ExecutionTarget.FARGATE:
            return "arn:aws:ecs:us-east-1:123456789012:task/%08x" % seq
        if target is ExecutionTarget.EMR_SEDONA:
            return "emr-step-s-%08X" % seq
        # SageMaker
        return (
            "arn:aws:sagemaker:us-east-1:123456789012:processing-job/job-%08x" % seq
        )

    async def submit(
        self, *, task_type: str, target: ExecutionTarget, payload: Dict[str, Any]
    ) -> str:
        seq = len(self.submitted) + 1
        handle = self._provider_handle(target, seq)
        self.submitted.append(
            {
                "handle": handle,
                "task_type": task_type,
                "target": target,
                "payload": payload,
            }
        )
        return handle

    async def poll(
        self,
        *,
        job_id: str,
        task_type: str,
        target: ExecutionTarget,
        provider_handle: Optional[str] = None,
    ) -> JobState:
        count = self._poll_count.get(job_id, 0) + 1
        self._poll_count[job_id] = count

        if count <= self.polls_until_terminal:
            # Still working: first poll reports queued, later polls running.
            status = JobStatus.QUEUED if count == 1 else JobStatus.RUNNING
            return JobState(status=status)

        if job_id in self.fail_detail:
            return JobState(
                status=JobStatus.FAILED,
                failure_detail=self.fail_detail[job_id],
            )

        output_uri = "s3://%s/%s/%s/output.tif" % (
            RESULT_BUCKET,
            target.value,
            job_id,
        )
        return JobState(status=JobStatus.SUCCEEDED, output_s3_uri=output_uri)


@pytest.fixture
def server_with_backend():
    """An :class:`AwsGeoComputeServer` wired to the simulated AWS backend."""
    backend = SimulatedAwsComputeBackend(polls_until_terminal=1)
    server = AwsGeoComputeServer(job_backend=backend)
    return server, backend


# ---------------------------------------------------------------------------
# submit -> status -> result (Req 13.3, 13.4)
# ---------------------------------------------------------------------------


async def test_submit_status_result_happy_path(server_with_backend) -> None:
    """A submitted job advances queued -> running -> succeeded with an S3 result."""
    server, _backend = server_with_backend

    submission = await server.submit(
        task_type="bulk-cog-conversion", payload={"src": "s3://amzn-s3-demo-input/scene.tif"}
    )
    assert submission.job_id
    assert submission.target is ExecutionTarget.BATCH

    # First query: still queued (one of the four allowed statuses; Req 13.3).
    view = await server.status(job_id=submission.job_id)
    assert view.status is JobStatus.QUEUED
    assert view.output_s3_uri is None

    # Drive it to the terminal state and read the result location (Req 13.4).
    final = await server.status(job_id=submission.job_id)
    assert final.status is JobStatus.SUCCEEDED
    assert final.output_s3_uri == (
        "s3://%s/aws-batch/%s/output.tif" % (RESULT_BUCKET, submission.job_id)
    )


async def test_status_is_one_of_the_four_allowed_values(server_with_backend) -> None:
    """Every status query returns exactly one of the four lifecycle values (Req 13.3)."""
    server, _backend = server_with_backend
    submission = await server.submit(task_type="model-inference")
    for _ in range(3):
        view = await server.status(job_id=submission.job_id)
        assert view.status in {
            JobStatus.QUEUED,
            JobStatus.RUNNING,
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
        }


@pytest.mark.parametrize(
    "task_type, expected_target",
    [
        ("bulk-cog-conversion", ExecutionTarget.BATCH),
        ("windowed-raster-op", ExecutionTarget.FARGATE),
        ("distributed-spatial-join", ExecutionTarget.EMR_SEDONA),
        ("foundation-model-embed", ExecutionTarget.SAGEMAKER),
    ],
)
async def test_result_location_per_target(
    server_with_backend, task_type: str, expected_target: ExecutionTarget
) -> None:
    """Each supported task type runs on its mapped target and yields an S3 result."""
    server, _backend = server_with_backend
    submission = await server.submit(task_type=task_type)
    assert submission.target is expected_target
    # advance past the single working poll to the terminal success state.
    await server.status(job_id=submission.job_id)
    final = await server.status(job_id=submission.job_id)
    assert final.status is JobStatus.SUCCEEDED
    assert final.output_s3_uri is not None
    assert final.output_s3_uri.startswith(
        "s3://%s/%s/" % (RESULT_BUCKET, expected_target.value)
    )


# ---------------------------------------------------------------------------
# unsupported task type (Req 13.6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_task", ["", "unsupported-task", "BULK-COG-CONVERSION"])
async def test_unsupported_task_type_is_validation_error(
    server_with_backend, bad_task: str
) -> None:
    """An unsupported task type is rejected with a validation error naming it (Req 13.6)."""
    server, backend = server_with_backend
    with pytest.raises(ValidationError) as exc_info:
        await server.submit(task_type=bad_task)
    assert exc_info.value.category is ErrorCategory.VALIDATION
    # Nothing was handed to the backend for an unsupported task type.
    assert backend.submitted == []


# ---------------------------------------------------------------------------
# unknown job id (Req 13.8)
# ---------------------------------------------------------------------------


async def test_unknown_job_id_is_not_found(server_with_backend) -> None:
    """A status query for an id that was never submitted is a not-found error (Req 13.8)."""
    server, _backend = server_with_backend
    with pytest.raises(NotFoundError) as exc_info:
        await server.status(job_id="00000000000000000000000000000000")
    assert exc_info.value.category is ErrorCategory.NOT_FOUND


async def test_unknown_job_id_after_other_submissions(server_with_backend) -> None:
    """A valid submission does not make an unrelated id resolve (Req 13.8)."""
    server, _backend = server_with_backend
    await server.submit(task_type="bulk-cog-conversion")
    with pytest.raises(NotFoundError):
        await server.status(job_id="not-a-real-job-id")


# ---------------------------------------------------------------------------
# failed job (Req 13.5)
# ---------------------------------------------------------------------------


async def test_failed_job_returns_taxonomy_error_with_detail(
    server_with_backend,
) -> None:
    """A failed job surfaces as a taxonomy error carrying the failure detail (Req 13.5)."""
    server, backend = server_with_backend
    submission = await server.submit(task_type="distributed-spatial-join")
    backend.fail_detail[submission.job_id] = "EMR step failed: executor OOM at stage 3"

    # advance past the working poll, then the terminal failed poll.
    await server.status(job_id=submission.job_id)
    with pytest.raises(UpstreamError) as exc_info:
        await server.status(job_id=submission.job_id)

    assert exc_info.value.category is ErrorCategory.UPSTREAM
    assert exc_info.value.original == "EMR step failed: executor OOM at stage 3"
    # The persisted record retains the failure detail.
    record = server._require_jobs().get_record(submission.job_id)
    assert record is not None
    assert record.status is JobStatus.FAILED
    assert record.failure_detail == "EMR step failed: executor OOM at stage 3"


# ---------------------------------------------------------------------------
# Equivalent test against moto (real boto3 S3); skips if moto is unavailable
# ---------------------------------------------------------------------------


class MotoS3BackedBackend:
    """A :class:`JobBackend` whose successful jobs write output into S3 (moto).

    Faithfully simulates the result hand-off: on the terminal poll it puts an
    object into the (moto-mocked) S3 bucket and returns that object's ``s3://``
    URI, so the test can read the result back from S3 exactly as a caller would.
    """

    def __init__(self, s3_client, bucket: str) -> None:
        self._s3 = s3_client
        self._bucket = bucket
        self.submitted: List[Dict[str, Any]] = []
        self._polled: Dict[str, bool] = {}

    async def submit(
        self, *, task_type: str, target: ExecutionTarget, payload: Dict[str, Any]
    ) -> str:
        seq = len(self.submitted) + 1
        handle = "batch-job-%08x" % seq
        self.submitted.append({"handle": handle, "target": target})
        return handle

    async def poll(
        self,
        *,
        job_id: str,
        task_type: str,
        target: ExecutionTarget,
        provider_handle: Optional[str] = None,
    ) -> JobState:
        if not self._polled.get(job_id):
            self._polled[job_id] = True
            return JobState(status=JobStatus.RUNNING)
        key = "%s/%s/output.tif" % (target.value, job_id)
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=b"COG-bytes")
        return JobState(
            status=JobStatus.SUCCEEDED,
            output_s3_uri="s3://%s/%s" % (self._bucket, key),
        )


async def test_submit_status_result_against_moto_s3(aws_mock) -> None:
    """End-to-end submit/status/result with output materialized in moto S3."""
    boto3 = pytest.importorskip("boto3")
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=RESULT_BUCKET)

    backend = MotoS3BackedBackend(s3, RESULT_BUCKET)
    server = AwsGeoComputeServer(job_backend=backend)

    submission = await server.submit(task_type="bulk-cog-conversion")
    running = await server.status(job_id=submission.job_id)
    assert running.status is JobStatus.RUNNING

    final = await server.status(job_id=submission.job_id)
    assert final.status is JobStatus.SUCCEEDED
    assert final.output_s3_uri is not None

    # The advertised S3 result location actually exists and is readable.
    key = final.output_s3_uri.split("/", 3)[3]
    obj = s3.get_object(Bucket=RESULT_BUCKET, Key=key)
    assert obj["Body"].read() == b"COG-bytes"

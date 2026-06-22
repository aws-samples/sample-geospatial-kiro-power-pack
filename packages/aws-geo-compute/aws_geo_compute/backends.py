"""A concrete AWS Batch job backend for ``aws-geo-compute`` (Requirement 13).

:class:`AwsBatchJobBackend` is a concrete :class:`~aws_geo_compute.jobs.JobBackend`
that submits container jobs to **AWS Batch** and reports their status, returning
an S3 output location on success. It covers the two container-based execution
targets - :attr:`~aws_geo_compute.jobs.ExecutionTarget.BATCH` and
:attr:`~aws_geo_compute.jobs.ExecutionTarget.FARGATE` (Amazon ECS with AWS Fargate
runs through a Batch Fargate compute environment, so both use the same
``submit_job`` call). The distributed/GPU targets (Amazon EMR with Apache Sedona,
Amazon SageMaker AI) are intentionally
**not** handled here: they need very different infrastructure, so submitting one
raises an actionable ``UpstreamError`` telling the operator to inject a custom
backend for those.

boto3 is an **optional** dependency (the ``[aws]`` extra). It is imported lazily
so the rest of ``aws-geo-compute`` - whose engine-selection/planning surface
needs no AWS at all, and whose tests inject a fake client - never requires boto3.

Credential + config surface (the single ``mcp.json`` surface, the environment):
boto3 reads the standard ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` /
``AWS_REGION`` keys; the backend additionally needs ``AWS_BATCH_JOB_QUEUE`` and
``AWS_BATCH_JOB_DEFINITION`` (the Batch job queue and job definition to run), and
optionally ``AWS_GEO_COMPUTE_OUTPUT_S3`` (an ``s3://`` prefix used to report a
per-job output location). :func:`default_job_backend` wires this backend when AWS
credentials are configured, so the server's ``main()`` offers job submission once
AWS is set up; absent credentials the server denies ``submit``/``status`` with an
``AuthenticationError`` naming the AWS key.

The blocking boto3 calls are run in a worker thread (``asyncio.to_thread``) so
they neither block the event loop nor defeat the manager's submit/status
timeouts.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, Dict, Mapping, Optional

from geo_common.errors import (
    AuthenticationError,
    GeoError,
    NotFoundError,
    UpstreamError,
)

from aws_geo_compute.jobs import ExecutionTarget, JobBackend, JobState, JobStatus

__all__ = [
    "AwsBatchJobBackend",
    "default_job_backend",
    "AWS_ACCESS_KEY_ID_KEY",
    "AWS_BATCH_JOB_QUEUE_KEY",
    "AWS_BATCH_JOB_DEFINITION_KEY",
    "AWS_GEO_COMPUTE_OUTPUT_S3_KEY",
]

#: Primary AWS credential key (named in the credential-deny message).
AWS_ACCESS_KEY_ID_KEY = "AWS_ACCESS_KEY_ID"

#: The Batch job queue to submit to.
AWS_BATCH_JOB_QUEUE_KEY = "AWS_BATCH_JOB_QUEUE"

#: The Batch job definition to run.
AWS_BATCH_JOB_DEFINITION_KEY = "AWS_BATCH_JOB_DEFINITION"

#: Optional ``s3://`` prefix used to report a per-job output location.
AWS_GEO_COMPUTE_OUTPUT_S3_KEY = "AWS_GEO_COMPUTE_OUTPUT_S3"

#: The execution targets this backend can run (both via AWS Batch).
_SUPPORTED_TARGETS = (ExecutionTarget.BATCH, ExecutionTarget.FARGATE)

#: AWS Batch job states that mean "not yet running".
_QUEUED_STATES = {"SUBMITTED", "PENDING", "RUNNABLE"}
#: AWS Batch job states that mean "running".
_RUNNING_STATES = {"STARTING", "RUNNING"}


class AwsBatchJobBackend:
    """Submit and poll container jobs on AWS Batch (Requirement 13).

    A boto3 Batch client may be injected (``batch_client=...``) so the backend
    is exercisable without boto3 or a real account; otherwise a client is built
    lazily from the environment. ``job_queue`` / ``job_definition`` /
    ``output_s3_prefix`` default to their environment keys.
    """

    def __init__(
        self,
        *,
        batch_client: Optional[Any] = None,
        job_queue: Optional[str] = None,
        job_definition: Optional[str] = None,
        output_s3_prefix: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        self._batch_client = batch_client
        self._job_queue = job_queue
        self._job_definition = job_definition
        self._output_s3_prefix = output_s3_prefix
        self._region = region

    async def submit(
        self, *, task_type: str, target: ExecutionTarget, payload: Dict[str, Any]
    ) -> str:
        """Submit a Batch job and return its ``jobId`` (the provider handle).

        Rejects a target this backend does not run (EMR/SageMaker) with an
        ``UpstreamError`` naming it; a missing queue/definition raises an
        ``UpstreamError`` naming the env key. The ``task_type`` and ``payload``
        are passed to the container as environment overrides.
        """
        if target not in _SUPPORTED_TARGETS:
            raise UpstreamError(
                "the AWS Batch backend does not run %s jobs; inject a custom "
                "JobBackend for that target" % target.value,
                source="aws-geo-compute",
                detail={"target": target.value, "task_type": task_type},
            )
        client = self._batch_client or self._build_client()
        queue = self._require_config(self._job_queue, AWS_BATCH_JOB_QUEUE_KEY)
        definition = self._require_config(
            self._job_definition, AWS_BATCH_JOB_DEFINITION_KEY
        )

        job_name = "geo-%s-%s" % (
            _sanitize_job_name(task_type),
            uuid.uuid4().hex[:12],
        )
        environment = [
            {"name": "GEO_TASK_TYPE", "value": task_type},
            {"name": "GEO_PAYLOAD", "value": json.dumps(payload or {})},
        ]
        prefix = self._output_s3_prefix or os.environ.get(AWS_GEO_COMPUTE_OUTPUT_S3_KEY)
        if prefix:
            environment.append({"name": "GEO_OUTPUT_S3", "value": prefix})

        try:
            response = await asyncio.to_thread(
                client.submit_job,
                jobName=job_name,
                jobQueue=queue,
                jobDefinition=definition,
                containerOverrides={"environment": environment},
            )
        except GeoError:
            raise
        except Exception as exc:  # noqa: BLE001 - boto3/botocore error
            raise self._classify(exc) from exc

        job_id = response.get("jobId") if isinstance(response, Mapping) else None
        if not job_id:
            raise UpstreamError(
                "AWS Batch did not return a jobId for the submitted job",
                source="aws-geo-compute",
                detail={"target": target.value},
            )
        return str(job_id)

    async def poll(
        self,
        *,
        job_id: str,
        task_type: str,
        target: ExecutionTarget,
        provider_handle: Optional[str] = None,
    ) -> JobState:
        """Describe the Batch job and map its status onto a :class:`JobState`.

        Uses ``provider_handle`` (the Batch ``jobId`` returned by
        :meth:`submit`) to look the job up; a succeeded job reports its S3
        output location (from the configured output prefix), a failed job
        carries Batch's status reason.
        """
        if not provider_handle:
            raise UpstreamError(
                "no AWS Batch jobId was recorded for job %r; cannot poll status"
                % job_id,
                source="aws-geo-compute",
                detail={"job_id": job_id},
            )
        client = self._batch_client or self._build_client()
        try:
            response = await asyncio.to_thread(
                client.describe_jobs, jobs=[provider_handle]
            )
        except GeoError:
            raise
        except Exception as exc:  # noqa: BLE001 - boto3/botocore error
            raise self._classify(exc) from exc

        jobs = response.get("jobs", []) if isinstance(response, Mapping) else []
        if not jobs:
            raise NotFoundError(
                "AWS Batch has no job with id %r" % provider_handle,
                source="aws-geo-compute",
                detail={"job_id": job_id, "provider_handle": provider_handle},
            )
        return self._to_state(jobs[0], provider_handle=provider_handle)

    def _to_state(self, job: Mapping[str, Any], *, provider_handle: str) -> JobState:
        """Map an AWS Batch job description onto a :class:`JobState`."""
        status = str(job.get("status") or "").upper()
        if status in _QUEUED_STATES:
            return JobState(status=JobStatus.QUEUED)
        if status in _RUNNING_STATES:
            return JobState(status=JobStatus.RUNNING)
        if status == "SUCCEEDED":
            return JobState(
                status=JobStatus.SUCCEEDED,
                output_s3_uri=self._output_uri(provider_handle),
            )
        if status == "FAILED":
            reason = job.get("statusReason") or self._container_reason(job) or "AWS Batch job failed"
            return JobState(status=JobStatus.FAILED, failure_detail=str(reason))
        # Unknown/empty Batch status: treat conservatively as still running.
        return JobState(status=JobStatus.RUNNING)

    @staticmethod
    def _container_reason(job: Mapping[str, Any]) -> Optional[str]:
        """Pull a failure reason from the job's container block, if present."""
        container = job.get("container")
        if isinstance(container, Mapping):
            reason = container.get("reason")
            if isinstance(reason, str) and reason:
                return reason
        return None

    def _output_uri(self, provider_handle: str) -> Optional[str]:
        """Report the conventional S3 output location for a succeeded job."""
        prefix = self._output_s3_prefix or os.environ.get(AWS_GEO_COMPUTE_OUTPUT_S3_KEY)
        if not prefix:
            return None
        return "%s/%s/" % (prefix.rstrip("/"), provider_handle)

    def _require_config(self, value: Optional[str], env_key: str) -> str:
        """Resolve a required Batch config value from the arg or env, or raise."""
        resolved = value or os.environ.get(env_key)
        if not resolved or not str(resolved).strip():
            raise UpstreamError(
                "AWS Batch is not fully configured: set %s in mcp.json" % env_key,
                source="aws-geo-compute",
                detail={"missing_config": env_key},
            )
        return str(resolved).strip()

    def _build_client(self) -> Any:
        """Build a boto3 Batch client, or raise an actionable taxonomy error."""
        try:
            import boto3  # noqa: PLC0415 - optional dependency, imported lazily
        except ImportError as exc:  # pragma: no cover - exercised only w/o boto3
            raise UpstreamError(
                "the AWS Batch backend requires the 'boto3' package; install the "
                "aws-geo-compute[aws] extra to enable it",
                source="aws-geo-compute",
                original=str(exc),
            ) from exc
        region = (
            self._region
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
        )
        return boto3.client("batch", region_name=region)

    def _classify(self, exc: Exception) -> GeoError:
        """Map a boto3/botocore failure onto the ``Error_Taxonomy``."""
        name = type(exc).__name__
        message = str(exc)
        code: Optional[str] = None
        http_status: Optional[int] = None
        response = getattr(exc, "response", None)
        if isinstance(response, Mapping):
            error = response.get("Error", {})
            if isinstance(error, Mapping):
                code = error.get("Code")
            meta = response.get("ResponseMetadata", {})
            if isinstance(meta, Mapping):
                http_status = meta.get("HTTPStatusCode")

        auth_codes = {
            "AccessDenied",
            "AccessDeniedException",
            "UnrecognizedClientException",
            "InvalidSignatureException",
            "AuthFailure",
            "ExpiredTokenException",
        }
        if (
            name in ("NoCredentialsError", "PartialCredentialsError")
            or code in auth_codes
            or http_status in (401, 403)
        ):
            return AuthenticationError(
                "AWS Batch rejected the AWS credentials configured in mcp.json",
                source="aws-geo-compute",
                detail={"mcp_json_key": AWS_ACCESS_KEY_ID_KEY},
                original=message,
            )
        if code in ("ClientException",) or http_status == 400:
            return UpstreamError(
                "AWS Batch rejected the request: %s" % message,
                source="aws-geo-compute",
                detail={"code": code},
                original=message,
            )
        return UpstreamError(
            "AWS Batch operation failed: %s" % message,
            source="aws-geo-compute",
            original=message,
        )


def _sanitize_job_name(task_type: str) -> str:
    """Coerce a task type into a valid Batch jobName fragment ([A-Za-z0-9_-])."""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in task_type)
    return (cleaned[:90] or "job").strip("-") or "job"


def default_job_backend() -> "Optional[JobBackend]":
    """Return the auto-wired job backend used by the server's ``main()``.

    Gates on **credential presence**: the AWS Batch backend is wired only when
    :data:`AWS_ACCESS_KEY_ID_KEY` is configured. Absent the credential the
    server has no backend and denies ``submit``/``status`` with an
    ``AuthenticationError`` naming the key (the documented healthy refusal). The
    boto3 driver and the Batch queue/definition are resolved lazily at submit
    time, so a configured-but-incomplete deployment surfaces an actionable
    ``UpstreamError`` (install ``aws-geo-compute[aws]`` / set
    ``AWS_BATCH_JOB_QUEUE``/``AWS_BATCH_JOB_DEFINITION``).
    """
    if not os.environ.get(AWS_ACCESS_KEY_ID_KEY):
        return None
    return AwsBatchJobBackend()

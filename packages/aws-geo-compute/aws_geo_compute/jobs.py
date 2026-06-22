"""Job submission and status for the ``aws-geo-compute`` peer Power (Req 13).

Once :mod:`aws_geo_compute.engine_selection` decides a task must be delegated
and :mod:`aws_geo_compute.delegation` hands it to this peer Power, the peer must
actually **run** the job on AWS and let callers track it. This module implements
that surface (Requirement 13):

* :class:`ExecutionTarget` and :data:`TASK_TARGET_MAP` - the deterministic,
  single-valued mapping from a supported **task type** to **exactly one** AWS
  execution target (AWS Batch, Amazon ECS with AWS Fargate, Amazon EMR with
  Apache Sedona, or Amazon SageMaker AI) (Requirement 13.1; Property 18).
* :func:`select_target` - resolves a task type through that table, rejecting an
  unsupported task type with an ``Error_Taxonomy`` ``ValidationError`` that
  names it (Requirement 13.6).
* :class:`JobStatus`, :class:`JobSubmission`, :class:`JobStatusView`, and
  :class:`ComputeJobRecord` - the job state machine and the views returned to
  callers (no secret-bearing field).
* :class:`JobManager` - submits jobs (returning a job id unique among submitted
  jobs within 10s; submission failure -> taxonomy error and **no** job id) and
  reports status (within 5s; ``succeeded`` -> S3 output location; ``failed`` ->
  taxonomy error carrying the failure detail; unknown id -> not-found).

The actual AWS calls live behind the injectable :class:`JobBackend` so the
manager is exercisable without AWS (the integration test in a later task swaps in
a moto/localstack-backed backend). Python 3.10+.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol

from pydantic import BaseModel, Field

from geo_common.errors import (
    GeoError,
    NetworkError,
    NotFoundError,
    UpstreamError,
    ValidationError,
)

__all__ = [
    "ExecutionTarget",
    "TASK_TARGET_MAP",
    "JobStatus",
    "JobSubmission",
    "JobStatusView",
    "ComputeJobRecord",
    "JobState",
    "JobBackend",
    "JobManager",
    "select_target",
    "SUBMIT_TIMEOUT_S",
    "STATUS_TIMEOUT_S",
]

#: Submission must return a job id within this many seconds (Requirement 13.2,
#: "within 10 seconds"). Exceeding it is treated as a submission failure.
SUBMIT_TIMEOUT_S = 10.0

#: A status query must return within this many seconds (Requirement 13.3,
#: "within 5 seconds"). Exceeding it aborts with a ``NETWORK`` taxonomy error.
STATUS_TIMEOUT_S = 5.0


class ExecutionTarget(str, Enum):
    """An AWS execution target the peer Power can run a job on (Requirement 13.1).

    Exactly one target is selected per supported task type (Property 18). A
    ``str`` enum so the value serializes directly to its wire string.
    """

    BATCH = "aws-batch"  # AWS Batch — containerized bulk/array jobs
    FARGATE = "fargate"  # Amazon ECS with AWS Fargate — short single-container tasks
    EMR_SEDONA = "emr-sedona"  # Amazon EMR with Apache Sedona — distributed spatial SQL/joins >100GB
    SAGEMAKER = "sagemaker-ai"  # Amazon SageMaker AI — GPU model inference/training (GeoAI)


#: The deterministic, single-valued task-type -> target mapping (Requirement
#: 13.1; Property 18). The single source of truth for supported task types: a
#: task type is "supported" iff it is a key here. Keep this a plain dict so the
#: mapping is total and trivially inspectable.
TASK_TARGET_MAP: "Dict[str, ExecutionTarget]" = {
    "bulk-cog-conversion": ExecutionTarget.BATCH,
    "windowed-raster-op": ExecutionTarget.FARGATE,
    "distributed-spatial-join": ExecutionTarget.EMR_SEDONA,
    "large-zonal-statistics": ExecutionTarget.EMR_SEDONA,
    "foundation-model-embed": ExecutionTarget.SAGEMAKER,
    "model-inference": ExecutionTarget.SAGEMAKER,
}


class JobStatus(str, Enum):
    """The lifecycle status of a submitted job (Requirement 13.3).

    A status query returns exactly one of these four values. ``SUCCEEDED``
    carries an S3 output location (Requirement 13.4); ``FAILED`` carries a
    failure detail surfaced as a taxonomy error (Requirement 13.5).
    """

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobSubmission(BaseModel):
    """The receipt returned from a successful job submission (Requirement 13.2).

    ``job_id`` is unique among submitted jobs and lets the caller query status.
    """

    job_id: str = Field(min_length=1)
    target: ExecutionTarget


class JobStatusView(BaseModel):
    """The view returned from a status query for a non-failed job (Req 13.3, 13.4).

    ``status`` is exactly one of the four :class:`JobStatus` values. When
    ``status`` is :attr:`JobStatus.SUCCEEDED`, ``output_s3_uri`` carries the S3
    location of the job output (Requirement 13.4). A failed job is surfaced as a
    taxonomy error rather than a view (Requirement 13.5), so ``failure_detail``
    is retained here only for completeness.
    """

    job_id: str = Field(min_length=1)
    status: JobStatus
    output_s3_uri: Optional[str] = None
    failure_detail: Optional[str] = None


class ComputeJobRecord(BaseModel):
    """The peer Power's record of a submitted job (Requirement 13; design model).

    Carries no secret-bearing field. Updated in place as status is polled;
    ``output_s3_uri`` is populated when the job succeeds (Requirement 13.4) and
    ``failure_detail`` when it fails (Requirement 13.5).
    """

    job_id: str = Field(min_length=1)
    task_type: str = Field(min_length=1)
    target: ExecutionTarget
    status: JobStatus
    submitted_at: datetime
    provider_handle: Optional[str] = None
    output_s3_uri: Optional[str] = None
    failure_detail: Optional[str] = None


class JobState(BaseModel):
    """A point-in-time status snapshot returned by a :class:`JobBackend`.

    The backend reports the current :class:`JobStatus` plus, where applicable,
    the S3 output location (when succeeded) or the failure detail (when failed).
    The :class:`JobManager` maps this snapshot onto the caller-facing view/error.
    """

    status: JobStatus
    output_s3_uri: Optional[str] = None
    failure_detail: Optional[str] = None


class JobBackend(Protocol):
    """The injectable AWS hand-off used by :class:`JobManager`.

    Isolating the side-effecting AWS calls behind this protocol keeps the
    manager deterministic and testable without AWS. ``submit`` launches the job
    on the selected target and returns the provider's job handle (it may raise
    to signal a submission failure); ``poll`` reports the job's current state.
    """

    async def submit(
        self, *, task_type: str, target: ExecutionTarget, payload: Dict[str, Any]
    ) -> str:
        """Launch the job and return the provider job handle (may raise).

        The returned handle (e.g. an AWS Batch ``jobId`` or an EMR step id) is
        retained by the :class:`JobManager` on the job record and handed back to
        :meth:`poll` as ``provider_handle`` so a stateful backend can correlate
        a status query with the provider job it created.
        """
        ...

    async def poll(
        self,
        *,
        job_id: str,
        task_type: str,
        target: ExecutionTarget,
        provider_handle: Optional[str] = None,
    ) -> JobState:
        """Return the current :class:`JobState` for a submitted job (may raise).

        ``job_id`` is the manager-minted id; ``provider_handle`` is the handle
        this backend returned from :meth:`submit` (``None`` if it returned no
        handle), letting a real backend look the provider job up.
        """
        ...


#: Maps an arbitrary exception onto the ``Error_Taxonomy`` (mirrors
#: :meth:`~geo_common.server.BaseGeoServer.map_error`). The server injects its
#: own mapper; a default is used when the manager is constructed standalone.
ErrorMapper = Callable[..., GeoError]


def _default_error_mapper(exc: Exception, *, source: str) -> GeoError:
    """Fallback mapper: pass a ``GeoError`` through, else wrap as ``UPSTREAM``.

    An already-classified error keeps its single category; any other error
    becomes ``UPSTREAM`` while retaining the original detail (Requirement 11.5).
    """
    if isinstance(exc, GeoError):
        return exc
    detail = str(exc).strip() or type(exc).__name__
    return UpstreamError(
        "aws-geo-compute job operation failed",
        source=source,
        original=detail,
    )


def select_target(task_type: str) -> ExecutionTarget:
    """Resolve a task type to exactly one execution target (Req 13.1, 13.6).

    Looks ``task_type`` up in :data:`TASK_TARGET_MAP`. The mapping is
    deterministic and single-valued, so the same task type always returns the
    same target (Property 18). An unsupported task type (one not present in the
    table, including a non-string or empty value) is rejected with an
    ``Error_Taxonomy`` ``ValidationError`` that names the unsupported task type
    (Requirement 13.6); nothing is selected.
    """
    if not isinstance(task_type, str) or not task_type:
        raise ValidationError(
            "task_type must be a non-empty string naming a supported task type",
            source="aws-geo-compute",
            detail={
                "parameter": "task_type",
                "supported": sorted(TASK_TARGET_MAP),
            },
        )
    try:
        return TASK_TARGET_MAP[task_type]
    except KeyError:
        raise ValidationError(
            "unsupported task type: %r" % task_type,
            source="aws-geo-compute",
            detail={
                "task_type": task_type,
                "supported": sorted(TASK_TARGET_MAP),
            },
        )


class JobManager:
    """Submit jobs and report status against an injectable backend (Requirement 13).

    Owns the in-memory registry of :class:`ComputeJobRecord` keyed by the job id
    it mints, so submitted job ids are unique by construction (Property 19) and
    a status query for an unknown id is a not-found (Requirement 13.8). The
    ``submit_timeout_s``/``status_timeout_s`` and ``error_mapper`` are injectable
    so the manager is exercisable standalone; in the server the mapper is the
    server's :meth:`~geo_common.server.BaseGeoServer.map_error`.
    """

    def __init__(
        self,
        backend: JobBackend,
        *,
        submit_timeout_s: float = SUBMIT_TIMEOUT_S,
        status_timeout_s: float = STATUS_TIMEOUT_S,
        error_mapper: Optional[ErrorMapper] = None,
        source: str = "aws-geo-compute",
    ) -> None:
        if not (hasattr(backend, "submit") and hasattr(backend, "poll")):
            raise TypeError("backend must provide async submit() and poll() methods")
        if submit_timeout_s <= 0:
            raise ValueError("submit_timeout_s must be positive")
        if status_timeout_s <= 0:
            raise ValueError("status_timeout_s must be positive")
        self._backend = backend
        self._submit_timeout_s = float(submit_timeout_s)
        self._status_timeout_s = float(status_timeout_s)
        self._error_mapper = error_mapper or _default_error_mapper
        self._source = source
        self._records: "Dict[str, ComputeJobRecord]" = {}

    @property
    def submit_timeout_s(self) -> float:
        """The submission deadline in seconds (Requirement 13.2)."""
        return self._submit_timeout_s

    @property
    def status_timeout_s(self) -> float:
        """The status-query deadline in seconds (Requirement 13.3)."""
        return self._status_timeout_s

    def select_target(self, task_type: str) -> ExecutionTarget:
        """Deterministic task-type -> target mapping (Req 13.1; delegates to module func)."""
        return select_target(task_type)

    def _new_job_id(self) -> str:
        """Mint a job id unique among submitted jobs (Requirement 13.2; Property 19).

        Uses ``uuid4`` and regenerates on the (astronomically unlikely) chance of
        a collision with an already-registered id, so the returned ids are
        pairwise distinct for any sequence of submissions on this manager.
        """
        job_id = uuid.uuid4().hex
        while job_id in self._records:  # pragma: no cover - collision is improbable
            job_id = uuid.uuid4().hex
        return job_id

    async def submit(
        self, *, task_type: str, payload: Optional[Dict[str, Any]] = None
    ) -> JobSubmission:
        """Submit a job and return a unique job id within 10s (Req 13.2, 13.6, 13.7).

        Resolves the target (unsupported task type -> ``ValidationError`` naming
        it, Requirement 13.6), mints a unique job id, and hands the job to the
        backend within :data:`SUBMIT_TIMEOUT_S` seconds. If submission fails or
        does not complete in time, an ``Error_Taxonomy`` error identifying the
        submission failure is raised and **no** job id is returned and **no**
        record is persisted (Requirement 13.7). On success a
        :class:`ComputeJobRecord` (status ``QUEUED``) is stored and a
        :class:`JobSubmission` carrying the job id and target is returned.
        """
        target = select_target(task_type)
        job_id = self._new_job_id()
        try:
            handle = await asyncio.wait_for(
                self._backend.submit(
                    task_type=task_type, target=target, payload=payload or {}
                ),
                timeout=self._submit_timeout_s,
            )
        except asyncio.TimeoutError:
            # Submission did not complete within 10s -> failure, no job id (Req 13.7).
            raise NetworkError(
                "job submission to %s did not complete within %gs"
                % (target.value, self._submit_timeout_s),
                source=self._source,
                detail={"task_type": task_type, "target": target.value},
            )
        except Exception as exc:  # noqa: BLE001 - mapped onto the taxonomy
            # Submission failed -> taxonomy error, no job id persisted (Req 13.7).
            raise self._error_mapper(exc, source=self._source)

        self._records[job_id] = ComputeJobRecord(
            job_id=job_id,
            task_type=task_type,
            target=target,
            status=JobStatus.QUEUED,
            submitted_at=datetime.now(timezone.utc),
            provider_handle=(str(handle) if handle is not None else None),
        )
        return JobSubmission(job_id=job_id, target=target)

    async def status(self, *, job_id: str) -> JobStatusView:
        """Return a submitted job's status within 5s (Req 13.3, 13.4, 13.5, 13.8).

        An id that does not correspond to a submitted job is a
        ``NotFoundError`` (Requirement 13.8). Otherwise the backend is polled
        within :data:`STATUS_TIMEOUT_S` seconds (a timeout -> ``NETWORK`` error)
        and the record updated. A ``succeeded`` job returns a
        :class:`JobStatusView` carrying the S3 output location (Requirement
        13.4); a ``failed`` job raises an ``Error_Taxonomy`` error carrying the
        job's failure detail (Requirement 13.5); ``queued``/``running`` return a
        view with that status.
        """
        record = self._records.get(job_id)
        if record is None:
            raise NotFoundError(
                "no submitted job with id %r" % job_id,
                source=self._source,
                detail={"job_id": job_id},
            )

        try:
            state = await asyncio.wait_for(
                self._backend.poll(
                    job_id=job_id,
                    task_type=record.task_type,
                    target=record.target,
                    provider_handle=record.provider_handle,
                ),
                timeout=self._status_timeout_s,
            )
        except asyncio.TimeoutError:
            raise NetworkError(
                "status query for job %r did not complete within %gs"
                % (job_id, self._status_timeout_s),
                source=self._source,
                detail={"job_id": job_id},
            )
        except Exception as exc:  # noqa: BLE001 - mapped onto the taxonomy
            raise self._error_mapper(exc, source=self._source)

        # Refresh the persisted record with the polled state.
        record.status = state.status
        record.output_s3_uri = state.output_s3_uri
        record.failure_detail = state.failure_detail

        if state.status is JobStatus.FAILED:
            # Failed job -> taxonomy error + failure detail (Requirement 13.5).
            raise UpstreamError(
                "delegated job %r failed" % job_id,
                source=self._source,
                detail={"job_id": job_id, "target": record.target.value},
                original=state.failure_detail,
            )

        return JobStatusView(
            job_id=job_id,
            status=state.status,
            output_s3_uri=(
                state.output_s3_uri
                if state.status is JobStatus.SUCCEEDED
                else None
            ),
            failure_detail=None,
        )

    def get_record(self, job_id: str) -> Optional[ComputeJobRecord]:
        """Return the stored :class:`ComputeJobRecord` for ``job_id``, if any."""
        return self._records.get(job_id)

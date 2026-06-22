"""aws-geo-compute: the peer Power for heavy/distributed geospatial compute.

This package orchestrates heavy processing on AWS. Task 17.1 implements the
**engine-selection decision tree** (Requirement 12.4) and the
**in-process-vs-delegate threshold** (Requirement 12.5) in
:mod:`aws_geo_compute.engine_selection`, and the **60-second delegation gate**
(Requirement 12.8) in :mod:`aws_geo_compute.delegation`. The
:class:`~aws_geo_compute.server.AwsGeoComputeServer` wires all of these into the
shared ``BaseGeoServer`` contract. Job submission and status (``select_target``,
``submit``, ``status``) live in :mod:`aws_geo_compute.jobs` (Requirement 13) and
compose with the delegation acceptor and the injectable job backend the server
accepts.
"""

from __future__ import annotations

from aws_geo_compute.backends import (
    AWS_BATCH_JOB_DEFINITION_KEY,
    AWS_BATCH_JOB_QUEUE_KEY,
    AWS_GEO_COMPUTE_OUTPUT_S3_KEY,
    AwsBatchJobBackend,
    default_job_backend,
)
from aws_geo_compute.delegation import (
    DELEGATION_TIMEOUT_S,
    DelegationAcceptor,
    DelegationGate,
    DelegationReceipt,
    DelegationRequest,
)
from aws_geo_compute.engine_selection import (
    DEFAULT_IN_PROCESS_MEMORY_LIMIT_BYTES,
    DISTRIBUTED_DATASET_LIMIT_BYTES,
    GB,
    MAX_IN_PROCESS_DATASET_BYTES,
    SMALL_DATASET_LIMIT_BYTES,
    AccessPattern,
    Engine,
    ExecutionPlan,
    select_engine,
    select_execution_plan,
    should_delegate,
)
from aws_geo_compute.jobs import (
    STATUS_TIMEOUT_S,
    SUBMIT_TIMEOUT_S,
    TASK_TARGET_MAP,
    ComputeJobRecord,
    ExecutionTarget,
    JobBackend,
    JobManager,
    JobState,
    JobStatus,
    JobStatusView,
    JobSubmission,
    select_target,
)
from aws_geo_compute.server import (
    INSTALL_COMMAND,
    AwsGeoComputeServer,
    ExecutionOutcome,
    main,
)

__all__ = [
    # Engine selection + delegation threshold (Req 12.4, 12.5)
    "Engine",
    "AccessPattern",
    "ExecutionPlan",
    "select_engine",
    "should_delegate",
    "select_execution_plan",
    "GB",
    "SMALL_DATASET_LIMIT_BYTES",
    "DISTRIBUTED_DATASET_LIMIT_BYTES",
    "DEFAULT_IN_PROCESS_MEMORY_LIMIT_BYTES",
    "MAX_IN_PROCESS_DATASET_BYTES",
    # Delegation gate (Req 12.8)
    "DelegationGate",
    "DelegationRequest",
    "DelegationReceipt",
    "DelegationAcceptor",
    "DELEGATION_TIMEOUT_S",
    # Job submission + status (Req 13)
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
    # Concrete AWS Batch backend
    "AwsBatchJobBackend",
    "default_job_backend",
    "AWS_BATCH_JOB_QUEUE_KEY",
    "AWS_BATCH_JOB_DEFINITION_KEY",
    "AWS_GEO_COMPUTE_OUTPUT_S3_KEY",
    # Server
    "AwsGeoComputeServer",
    "ExecutionOutcome",
    "INSTALL_COMMAND",
    "main",
]

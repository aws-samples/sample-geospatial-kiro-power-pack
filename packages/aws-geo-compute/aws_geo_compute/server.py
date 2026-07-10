"""The ``aws-geo-compute`` peer Power server: engine selection + delegation gating.

This task (17.1) wires the engine-selection decision tree (Requirement 12.4),
the in-process-vs-delegate threshold (Requirement 12.5), and the 60-second
delegation gate (Requirement 12.8) into the shared
:class:`~geo_common.server.BaseGeoServer` contract. The job-submission and
status surface (``ExecutionTarget``, ``select_target``, ``submit``, ``status``)
is added by a later task; it composes with the delegation acceptor this server
already accepts.

The server exposes two cooperating concerns:

* :meth:`AwsGeoComputeServer.plan_execution` - a pure, deterministic tool that
  validates its inputs onto the taxonomy and returns the
  :class:`~aws_geo_compute.engine_selection.ExecutionPlan` (selected engine +
  delegate flag) for a task characterized by dataset size, access pattern, and
  required memory (Requirements 12.4, 12.5).
* :meth:`AwsGeoComputeServer.execute_or_delegate` - runs ``plan_execution`` and,
  when the plan calls for delegation, hands the task to the peer Power through
  the injected :class:`~aws_geo_compute.delegation.DelegationGate`, aborting with
  a taxonomy error if delegation fails or is not accepted within 60s
  (Requirement 12.8).

Python 3.10+: uses ``from __future__ import annotations`` together with
``typing`` generics.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from geo_common.errors import AuthenticationError, UpstreamError, ValidationError
from geo_common.models import (
    CatalogEntry,
    CredentialClassification,
    CredentialSpec,
    OpennessTier,
)
from geo_common.server import BaseGeoServer

from aws_geo_compute.delegation import (
    DelegationAcceptor,
    DelegationGate,
    DelegationReceipt,
    DelegationRequest,
)
from aws_geo_compute.engine_selection import (
    AccessPattern,
    DEFAULT_IN_PROCESS_MEMORY_LIMIT_BYTES,
    ExecutionPlan,
    select_execution_plan,
)
from aws_geo_compute.jobs import (
    ExecutionTarget,
    JobBackend,
    JobManager,
    JobStatusView,
    JobSubmission,
    select_target,
)

__all__ = ["AwsGeoComputeServer", "INSTALL_COMMAND", "main"]

#: ``uvx`` command that installs this peer Power (bundle-manifest).
INSTALL_COMMAND = "uvx aws-geo-compute"

#: The standard AWS ``mcp.json`` keys the peer Power reads to submit jobs and
#: read/write S3 outputs. Declared **Optional** so they never block startup
#: (Requirement 16.5): engine selection and planning run entirely in-process
#: with no AWS, and the keys are enforced at job-submission/status time instead
#: (matching the credential-guard pattern used by the other credentialed
#: servers). ``AWS_REGION`` is included because job submission is region-scoped.
AWS_CREDENTIAL_KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")


class AwsGeoComputeServer(BaseGeoServer):
    """Peer Power orchestrating engine selection and delegation (Req 12.4, 12.5, 12.8).

    Holds an optional injectable delegation acceptor (the hand-off to the peer
    Power's job runner). When an acceptor is provided it is wrapped in a
    :class:`~aws_geo_compute.delegation.DelegationGate` using this server's
    :meth:`map_error`, so every delegation failure resolves to exactly one
    taxonomy category.
    """

    pillar = "peer"
    server_name = "aws-geo-compute"
    version = "0.2.0"

    def __init__(
        self,
        *,
        delegation_acceptor: Optional[DelegationAcceptor] = None,
        job_backend: Optional[JobBackend] = None,
        memory_limit_bytes: int = DEFAULT_IN_PROCESS_MEMORY_LIMIT_BYTES,
        http=None,
    ) -> None:
        super().__init__(http=http)
        if memory_limit_bytes <= 0:
            raise ValueError("memory_limit_bytes must be positive")
        self.memory_limit_bytes = memory_limit_bytes
        self._gate: Optional[DelegationGate] = None
        if delegation_acceptor is not None:
            self._gate = DelegationGate(
                delegation_acceptor,
                error_mapper=self.map_error,
                source=self.server_name,
            )
        self._jobs: Optional[JobManager] = None
        if job_backend is not None:
            self._jobs = JobManager(
                job_backend,
                error_mapper=self.map_error,
                source=self.server_name,
            )
        self.register_tool("plan_execution", self.plan_execution)
        self.register_tool("select_target", self.select_target)
        self.register_tool("submit", self.submit)
        self.register_tool("status", self.status)

    # ------------------------------------------------------------------
    # Engine selection + delegation threshold (Requirements 12.4, 12.5)
    # ------------------------------------------------------------------

    def plan_execution(
        self,
        *,
        dataset_size_bytes: int,
        access_pattern: AccessPattern,
        required_memory_bytes: int = 0,
    ) -> ExecutionPlan:
        """Select the engine and the in-process-vs-delegate decision (Req 12.4, 12.5).

        Validates inputs *before* any work: ``dataset_size_bytes`` and
        ``required_memory_bytes`` must be non-negative integers and
        ``access_pattern`` must be a valid :class:`AccessPattern` (a string is
        coerced when it names one); otherwise an ``Error_Taxonomy``
        ``ValidationError`` naming the parameter is raised and nothing is
        selected. Returns the deterministic
        :class:`~aws_geo_compute.engine_selection.ExecutionPlan` (Property 17).
        """
        pattern = self._validate_inputs(
            dataset_size_bytes=dataset_size_bytes,
            access_pattern=access_pattern,
            required_memory_bytes=required_memory_bytes,
        )
        return select_execution_plan(
            dataset_size_bytes=dataset_size_bytes,
            access_pattern=pattern,
            required_memory_bytes=required_memory_bytes,
            memory_limit_bytes=self.memory_limit_bytes,
        )

    async def execute_or_delegate(
        self,
        *,
        task_type: str,
        dataset_size_bytes: int,
        access_pattern: AccessPattern,
        required_memory_bytes: int = 0,
        payload: Optional[Dict[str, Any]] = None,
    ) -> "ExecutionOutcome":
        """Plan, then run in-process or delegate to the peer Power (Req 12.4, 12.5, 12.8).

        Computes the :class:`ExecutionPlan`; if it calls for in-process
        execution, returns an outcome marked accordingly. If it calls for
        delegation, hands the task to the peer Power through the
        :class:`~aws_geo_compute.delegation.DelegationGate`: on success the
        outcome carries the peer's :class:`DelegationReceipt`; on failure or
        non-acceptance within 60s the gate raises an ``Error_Taxonomy`` error and
        the task is aborted (Requirement 12.8).

        Raises an ``Error_Taxonomy`` ``UpstreamError`` when delegation is
        required but no delegation acceptor was configured - the task cannot be
        delegated, so it is aborted rather than silently run in-process.
        """
        plan = self.plan_execution(
            dataset_size_bytes=dataset_size_bytes,
            access_pattern=access_pattern,
            required_memory_bytes=required_memory_bytes,
        )
        if not plan.delegate:
            return ExecutionOutcome(plan=plan, delegated=False, receipt=None)

        if self._gate is None:
            raise UpstreamError(
                "task requires delegation but no aws-geo-compute delegation "
                "target is configured",
                source=self.server_name,
                detail={"task_type": task_type, "engine": plan.engine.value},
            )

        request = DelegationRequest(
            task_type=task_type,
            engine=plan.engine.value,
            dataset_size_bytes=dataset_size_bytes,
            required_memory_bytes=required_memory_bytes,
            payload=payload or {},
        )
        receipt = await self._gate.delegate(request)
        return ExecutionOutcome(plan=plan, delegated=True, receipt=receipt)

    def _validate_inputs(
        self,
        *,
        dataset_size_bytes: int,
        access_pattern: Any,
        required_memory_bytes: int,
    ) -> AccessPattern:
        """Validate selection inputs, returning the resolved access pattern.

        Raises an ``Error_Taxonomy`` ``ValidationError`` (never an unhandled
        exception) identifying the offending parameter.
        """
        if isinstance(dataset_size_bytes, bool) or not isinstance(
            dataset_size_bytes, int
        ):
            raise ValidationError(
                "dataset_size_bytes must be an integer number of bytes",
                source=self.server_name,
                detail={"parameter": "dataset_size_bytes"},
            )
        if dataset_size_bytes < 0:
            raise ValidationError(
                "dataset_size_bytes must be non-negative",
                source=self.server_name,
                detail={"parameter": "dataset_size_bytes"},
            )
        if isinstance(required_memory_bytes, bool) or not isinstance(
            required_memory_bytes, int
        ):
            raise ValidationError(
                "required_memory_bytes must be an integer number of bytes",
                source=self.server_name,
                detail={"parameter": "required_memory_bytes"},
            )
        if required_memory_bytes < 0:
            raise ValidationError(
                "required_memory_bytes must be non-negative",
                source=self.server_name,
                detail={"parameter": "required_memory_bytes"},
            )

        if isinstance(access_pattern, AccessPattern):
            return access_pattern
        try:
            return AccessPattern(access_pattern)
        except ValueError:
            raise ValidationError(
                "access_pattern must be one of: %s"
                % ", ".join(p.value for p in AccessPattern),
                source=self.server_name,
                detail={
                    "parameter": "access_pattern",
                    "supported": [p.value for p in AccessPattern],
                },
            )

    # ------------------------------------------------------------------
    # Job submission and status (Requirement 13)
    # ------------------------------------------------------------------

    def select_target(self, task_type: str) -> ExecutionTarget:
        """Map a supported task type to exactly one execution target (Req 13.1, 13.6).

        Pure and deterministic (Property 18); an unsupported task type raises an
        ``Error_Taxonomy`` ``ValidationError`` naming it (Requirement 13.6).
        Available regardless of whether a job backend is configured.
        """
        return select_target(task_type)

    async def submit(
        self, *, task_type: str, payload: Optional[Dict[str, Any]] = None
    ) -> JobSubmission:
        """Submit a job, returning a unique job id within 10s (Req 13.2, 13.6, 13.7).

        Delegates to the :class:`~aws_geo_compute.jobs.JobManager`. An
        ``Error_Taxonomy`` ``UpstreamError`` is raised when no job backend is
        configured, so a submission is never silently dropped.
        """
        return await self._require_jobs().submit(task_type=task_type, payload=payload)

    async def status(self, *, job_id: str) -> JobStatusView:
        """Return a submitted job's status within 5s (Req 13.3, 13.4, 13.5, 13.8).

        Delegates to the :class:`~aws_geo_compute.jobs.JobManager`: unknown id ->
        not-found (Requirement 13.8); succeeded -> S3 output location
        (Requirement 13.4); failed -> taxonomy error + failure detail
        (Requirement 13.5).
        """
        return await self._require_jobs().status(job_id=job_id)

    def _require_jobs(self) -> JobManager:
        """Return the configured :class:`JobManager` or abort with a taxonomy error.

        When a job backend is injected (production wiring or tests), it is the
        auth boundary and is used directly. Otherwise the AWS credentials are
        required to reach AWS: a missing key raises an ``Error_Taxonomy``
        ``AuthenticationError`` naming it (Requirement 7.8-style guard), and if
        the credentials are present but no backend was wired, the request aborts
        with an ``UpstreamError`` rather than silently doing nothing.
        """
        if self._jobs is not None:
            return self._jobs
        self._require_aws_credentials()
        raise UpstreamError(
            "job submission/status requires an aws-geo-compute job backend "
            "but none is configured",
            source=self.server_name,
        )

    def _require_aws_credentials(self) -> None:
        """Raise an ``AuthenticationError`` naming any missing AWS ``mcp.json`` key.

        Reads the single ``mcp.json`` surface (the environment); a key with a
        non-empty value is configured. The secret values themselves are never
        included in the error.
        """
        missing = [key for key in AWS_CREDENTIAL_KEYS if not os.environ.get(key)]
        if missing:
            raise AuthenticationError(
                "aws-geo-compute requires AWS credentials to submit or query "
                "jobs: missing %s; configure them in mcp.json"
                % ", ".join(missing),
                source=self.server_name,
                detail={"missing_credentials": missing},
            )

    # ------------------------------------------------------------------
    # Catalog + credential declaration (Req 2.1, 16.1)
    # ------------------------------------------------------------------

    def catalog_entries(self) -> "List[CatalogEntry]":
        """Register every ``aws-geo-compute`` capability in the Resource Catalog.

        The peer Power exposes four capabilities (Req 2.1): ``plan_execution``
        and ``select_target`` are pure/deterministic and always available;
        ``submit`` and ``status`` require a configured job backend, so their
        ``installed`` flag reflects whether one is wired (an unconfigured entry
        carries the install command, Req 2.6). All name ``aws-geo-compute`` as
        the provider (Req 11.3).
        """
        jobs_configured = self._jobs is not None
        specs = [
            (
                "plan_execution",
                OpennessTier.FREE_TIER,
                True,
                "Select a processing engine from the documented decision tree "
                "over dataset size and access pattern, and decide whether to "
                "delegate heavy tasks to AWS compute.",
            ),
            (
                "select_target",
                OpennessTier.FREE_TIER,
                True,
                "Map a supported task type to exactly one AWS execution target "
                "(AWS Batch, AWS Fargate, Amazon EMR with Apache Sedona, or Amazon SageMaker AI).",
            ),
            (
                "submit",
                OpennessTier.FREE_TIER,
                jobs_configured,
                "Submit a geospatial compute job to the selected AWS execution "
                "target and return a unique job id.",
            ),
            (
                "status",
                OpennessTier.FREE_TIER,
                jobs_configured,
                "Report a submitted job's status, returning the S3 output "
                "location on success.",
            ),
        ]
        return [
            CatalogEntry(
                name=name,
                pillar=self.pillar,
                capability_description=description,
                openness_tier=tier,
                provider_server=self.server_name,
                installed=installed,
                install_command=None if installed else INSTALL_COMMAND,
            )
            for name, tier, installed, description in specs
        ]

    def required_credentials(self) -> "List[CredentialSpec]":
        """Declare the standard AWS credentials, all Optional (Req 16.1, 16.5).

        Optional so engine selection and planning run with no configuration;
        AWS is needed only when a task is actually submitted, so an absent key
        never blocks startup (Requirement 16.5). The keys are enforced at
        ``submit``/``status`` time instead (an ``AuthenticationError`` names any
        missing key), mirroring the credential-guard pattern of the other
        credentialed servers.
        """
        return [
            CredentialSpec(
                source="AWS account",
                mcp_json_key=key,
                classification=CredentialClassification.OPTIONAL,
                license_reference="AWS Customer Agreement",
            )
            for key in AWS_CREDENTIAL_KEYS
        ]


class ExecutionOutcome:
    """The result of :meth:`AwsGeoComputeServer.execute_or_delegate`.

    Carries the deterministic :class:`ExecutionPlan`, whether the task was
    ``delegated``, and (when delegated and accepted) the peer's
    :class:`DelegationReceipt`. A plain object rather than a pydantic model so
    it can hold the receipt as-is without re-validation.
    """

    __slots__ = ("plan", "delegated", "receipt")

    def __init__(
        self,
        *,
        plan: ExecutionPlan,
        delegated: bool,
        receipt: Optional[DelegationReceipt],
    ) -> None:
        self.plan = plan
        self.delegated = delegated
        self.receipt = receipt

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            "ExecutionOutcome(engine=%r, delegated=%r, job_id=%r)"
            % (
                self.plan.engine.value,
                self.delegated,
                None if self.receipt is None else self.receipt.job_id,
            )
        )


def main() -> None:
    """Console entry point: serve aws-geo-compute over MCP stdio.

    Wires the AWS Batch job backend when ``AWS_ACCESS_KEY_ID`` is configured
    (via :func:`~aws_geo_compute.backends.default_job_backend`), so a deployed
    ``aws-geo-compute`` can submit/track jobs once AWS is set up; engine
    selection and planning always run with no credentials. Then runs the
    startup credential guard and serves this server's registered tools over
    stdin/stdout via the shared geo-common MCP runtime until the client
    disconnects.
    """
    from aws_geo_compute.backends import default_job_backend

    AwsGeoComputeServer(job_backend=default_job_backend()).run()

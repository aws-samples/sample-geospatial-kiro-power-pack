"""Property-based test for unique submitted job identifiers (task 17.5).

**Feature: geospatial-power-pack, Property 19: Submitted job identifiers are
unique**

*For any* sequence of submitted jobs, the returned job identifiers are pairwise
distinct.

**Validates: Requirements 13.2**

The property is exercised against a :class:`JobManager` wired to a controllable
in-memory stub :class:`~aws_geo_compute.jobs.JobBackend` (no AWS). Hypothesis
generates sequences of submissions over the supported task types (>= 100
generated cases per property, enforced by the ``geospatial-power-pack`` profile
registered in the root ``conftest.py``); for each sequence we collect every
returned ``job_id`` and assert they are pairwise distinct (i.e. the number of
distinct ids equals the number of submissions).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from hypothesis import given
from hypothesis import strategies as st

from aws_geo_compute.jobs import (
    TASK_TARGET_MAP,
    ExecutionTarget,
    JobManager,
    JobState,
    JobStatus,
)


class StubBackend:
    """A deterministic, AWS-free :class:`JobBackend` for the property test.

    ``submit`` simply records the call and returns a provider handle; it never
    influences the job id the :class:`JobManager` mints, so it cannot mask a
    uniqueness regression in the manager.
    """

    def __init__(self) -> None:
        self.submitted: List[Dict[str, Any]] = []

    async def submit(
        self, *, task_type: str, target: ExecutionTarget, payload: Dict[str, Any]
    ) -> str:
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
        return JobState(status=JobStatus.QUEUED)


# Supported task types are exactly the keys of the task -> target map.
_SUPPORTED_TASK_TYPES = sorted(TASK_TARGET_MAP)

# A sequence of submissions: at least one, up to a healthy batch, each naming a
# supported task type (the only thing ``submit`` requires).
submission_sequences = st.lists(
    st.sampled_from(_SUPPORTED_TASK_TYPES),
    min_size=1,
    max_size=64,
)


async def _submit_all(task_types: List[str]) -> List[str]:
    """Submit every task type in order on a fresh manager and return the ids."""
    manager = JobManager(StubBackend())
    job_ids: List[str] = []
    for task_type in task_types:
        submission = await manager.submit(task_type=task_type)
        job_ids.append(submission.job_id)
    return job_ids


@given(task_types=submission_sequences)
def test_submitted_job_identifiers_are_pairwise_distinct(
    task_types: List[str],
) -> None:
    """Property 19: ids returned for a sequence of submissions are all distinct."""
    job_ids = asyncio.run(_submit_all(task_types))

    # One id per submission, and every id is pairwise distinct.
    assert len(job_ids) == len(task_types)
    assert all(job_id for job_id in job_ids)  # non-empty ids
    assert len(set(job_ids)) == len(job_ids)

"""The delegation gate: hand a heavy task to the peer Power within 60s (Req 12.8).

Once :mod:`aws_geo_compute.engine_selection` decides a task must be delegated
(Requirement 12.5), the actual hand-off to the ``aws-geo-compute`` peer Power
can fail or stall. Requirement 12.8 governs that act: *if delegation fails or
the peer Power does not accept the task within 60 seconds, the task is aborted
and an ``Error_Taxonomy`` error indicating the delegation failure is returned.*

This module isolates that side-effecting concern behind a small, injectable
interface so it is testable without a real peer:

* :class:`DelegationRequest` / :class:`DelegationReceipt` - the request handed
  to the peer and the acceptance it returns (an accepted receipt carries the
  peer's job id; a rejection carries the failure detail).
* :data:`DelegationAcceptor` - the injectable async callable that performs the
  hand-off (in production, the peer Power's job-submission path added in a later
  task; in tests, a controllable stub).
* :class:`DelegationGate` - wraps the acceptor with the **60-second acceptance
  deadline** and maps every failure path onto the ``Error_Taxonomy``:
  non-acceptance within 60s -> ``NETWORK``; an explicit rejection -> ``UPSTREAM``
  (retaining the peer's detail); any other raised exception -> mapped via the
  injected error mapper (an already-classified :class:`GeoError` passes through,
  anything else becomes ``UPSTREAM`` with its original detail retained).

Python 3.10+: uses ``from __future__ import annotations`` together with
``typing`` generics.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional

from pydantic import BaseModel, Field

from geo_common.errors import GeoError, NetworkError, UpstreamError

__all__ = [
    "DELEGATION_TIMEOUT_S",
    "DelegationRequest",
    "DelegationReceipt",
    "DelegationAcceptor",
    "ErrorMapper",
    "DelegationGate",
]

#: The acceptance deadline for delegation (Requirement 12.8, "within 60
#: seconds"). The peer Power must accept the task within this many seconds or
#: the task is aborted with a taxonomy error.
DELEGATION_TIMEOUT_S = 60.0


class DelegationRequest(BaseModel):
    """A heavy processing task handed to the ``aws-geo-compute`` peer Power.

    Carries the task type, the engine the decision tree selected, and the
    dataset/memory characteristics that triggered delegation, plus an opaque
    ``payload`` the peer needs to run the job. Secret-free by contract (the
    single credential surface lives in ``mcp.json``, never in the payload).
    """

    task_type: str = Field(min_length=1)
    engine: str = Field(min_length=1)
    dataset_size_bytes: int = Field(ge=0)
    required_memory_bytes: int = Field(default=0, ge=0)
    payload: Dict[str, Any] = Field(default_factory=dict)


class DelegationReceipt(BaseModel):
    """The peer Power's response to a delegation request.

    ``accepted`` is the gate's acceptance signal: a ``True`` receipt carries the
    peer's ``job_id`` so the caller can later query status; a ``False`` receipt
    carries a ``detail`` explaining the refusal, which the gate surfaces as the
    ``original`` detail of an ``UPSTREAM`` error (Requirement 12.8).
    """

    accepted: bool
    job_id: Optional[str] = None
    detail: Optional[str] = None


#: The injectable hand-off. Given a :class:`DelegationRequest`, perform the
#: delegation to the peer Power and return a :class:`DelegationReceipt`. May
#: raise; the gate maps any raised error onto the taxonomy.
DelegationAcceptor = Callable[[DelegationRequest], Awaitable[DelegationReceipt]]

#: Maps an arbitrary exception onto the ``Error_Taxonomy``. The server injects
#: :meth:`~geo_common.server.BaseGeoServer.map_error`; a default is used when
#: the gate is constructed standalone.
ErrorMapper = Callable[..., GeoError]


def _default_error_mapper(exc: Exception, *, source: str) -> GeoError:
    """Fallback mapper: pass a ``GeoError`` through, else wrap as ``UPSTREAM``.

    Mirrors :meth:`BaseGeoServer.map_error`'s catch-all: an already-classified
    error keeps its single category, and any other error becomes ``UPSTREAM``
    while retaining the original detail (Requirement 11.5).
    """
    if isinstance(exc, GeoError):
        return exc
    detail = str(exc).strip() or type(exc).__name__
    return UpstreamError(
        "delegation to %s failed" % source,
        source=source,
        original=detail,
    )


class DelegationGate:
    """Delegate a task to the peer Power within 60s or abort with a taxonomy error.

    Wraps an injected :data:`DelegationAcceptor` with the
    :data:`DELEGATION_TIMEOUT_S` acceptance deadline (Requirement 12.8). The
    ``timeout_s`` and ``error_mapper`` are injectable so the gate is exercisable
    standalone in tests; in the server the mapper is the server's
    :meth:`~geo_common.server.BaseGeoServer.map_error`.
    """

    def __init__(
        self,
        acceptor: DelegationAcceptor,
        *,
        timeout_s: float = DELEGATION_TIMEOUT_S,
        error_mapper: Optional[ErrorMapper] = None,
        source: str = "aws-geo-compute",
    ) -> None:
        if not callable(acceptor):
            raise TypeError("acceptor must be a callable returning an awaitable")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._acceptor = acceptor
        self._timeout_s = float(timeout_s)
        self._error_mapper = error_mapper or _default_error_mapper
        self._source = source

    @property
    def timeout_s(self) -> float:
        """The acceptance deadline in seconds (Requirement 12.8)."""
        return self._timeout_s

    async def delegate(self, request: DelegationRequest) -> DelegationReceipt:
        """Hand ``request`` to the peer Power, returning its accepted receipt.

        Aborts with an ``Error_Taxonomy`` error indicating the delegation
        failure (Requirement 12.8) when:

        * the peer does not accept within :data:`DELEGATION_TIMEOUT_S` seconds
          -> :class:`~geo_common.errors.NetworkError` (``NETWORK``);
        * the acceptor raises -> mapped via the injected error mapper (a
          ``GeoError`` passes through; anything else -> ``UPSTREAM`` with the
          original detail retained);
        * the peer returns a non-accepted receipt ->
          :class:`~geo_common.errors.UpstreamError` (``UPSTREAM``), retaining the
          peer's refusal detail.

        On success returns the :class:`DelegationReceipt` (``accepted is True``)
        carrying the peer's job id.
        """
        try:
            receipt = await asyncio.wait_for(
                self._acceptor(request), timeout=self._timeout_s
            )
        except asyncio.TimeoutError:
            # Non-acceptance within the 60s window (Requirement 12.8).
            raise NetworkError(
                "aws-geo-compute did not accept the task within %gs"
                % self._timeout_s,
                source=self._source,
                detail={
                    "task_type": request.task_type,
                    "timeout_s": self._timeout_s,
                },
            )
        except Exception as exc:  # noqa: BLE001 - mapped onto the taxonomy
            raise self._error_mapper(exc, source=self._source)

        if not receipt.accepted:
            # Explicit rejection by the peer (Requirement 12.8).
            raise UpstreamError(
                "aws-geo-compute rejected the delegated task",
                source=self._source,
                detail={"task_type": request.task_type},
                original=receipt.detail,
            )
        return receipt

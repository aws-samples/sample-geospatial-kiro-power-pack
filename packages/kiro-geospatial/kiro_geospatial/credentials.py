"""The Power Hub Credential Manager (Requirement 3).

The Credential Manager records, classifies, validates, and reports the status
of every source credential, reading them all from a **single configuration
surface** in ``mcp.json`` (Requirement 3.1). Its contract (see design.md "Power
Hub -> Credential Manager"):

* Classify each configured source into exactly one
  :class:`CredentialClassification` - Required / Optional / License-Needed
  (Requirement 3.2).
* Report each source's :class:`CredentialStatus` - Configured / Invalid /
  Missing / Unverifiable - within 5 seconds (Requirement 3.3). A per-source
  config-read failure degrades only that source to ``Unverifiable`` while the
  report still covers every other source (Requirement 3.9).
* Validate a present credential against its source within 10 seconds:
  ``Configured`` if accepted, ``Invalid`` if rejected (Requirement 3.4);
  ``Missing`` when absent (Requirement 3.5); ``Unverifiable`` on timeout or
  failure, classified per the ``Error_Taxonomy`` (Requirement 3.9).
* Guard a Required-but-Missing invocation with an
  :class:`~geo_common.errors.AuthenticationError` that names the missing
  credential and its ``mcp.json`` key (Requirement 3.6).
* Echo a licensing reference for License-Needed sources in the status report
  (Requirement 3.7).
* **Never** emit a credential secret value - not in any model, status report,
  error response, or log record (Requirement 3.8). The data model has no field
  that can hold a secret, and :class:`CredentialRedactionFilter` scrubs secret
  values out of log records.

The single ``mcp.json`` credential surface is modeled as a read-only
``Mapping[str, str]`` (defaulting to the process environment, which is how MCP
exposes ``mcp.json`` ``env`` values to a running server). Secret values are
read **transiently** for validation and redaction only; they are never stored
on the manager, copied into a model, or logged.

Python 3.10+ (per the package's ``requires-python``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from enum import Enum
from typing import (
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Union,
)

from pydantic import BaseModel, Field

# Reuse the shared base-contract types so the Hub and every server share one
# definition of how a credential is classified and specified (design.md notes
# the Hub's CredentialSpec is the same shape as geo-common's).
from geo_common.errors import AuthenticationError, ErrorCategory, GeoError
from geo_common.models import CredentialClassification, CredentialSpec

__all__ = [
    "CredentialClassification",
    "CredentialSpec",
    "CredentialStatus",
    "CredentialStatusView",
    "CredentialRecord",
    "CredentialManager",
    "CredentialRedactionFilter",
    "Validator",
]

#: A per-source credential validator. Given the (transient) secret value, it
#: returns ``True`` when the source accepts the credential and ``False`` when
#: the source rejects it (Requirement 3.4). It may be a coroutine function or a
#: plain callable; either way it is bounded by the 10s validation timeout. A
#: validator MUST NOT return or log the secret value.
Validator = Callable[[str], Union[bool, Awaitable[bool]]]

#: Bounded timeouts from Requirement 3.
_STATUS_REPORT_BUDGET_S = 5.0  # Requirement 3.3
_VALIDATION_TIMEOUT_S = 10.0  # Requirements 3.4 / 3.9


class CredentialStatus(str, Enum):
    """The runtime validation state of a source's credential (Requirement 3.3).

    A ``str`` enum so the value serializes directly to its wire string.
    """

    CONFIGURED = "Configured"  # source accepted the credential (Req 3.4)
    INVALID = "Invalid"  # source rejected the credential (Req 3.4)
    MISSING = "Missing"  # no credential present (Req 3.5)
    UNVERIFIABLE = "Unverifiable"  # timeout / failure / not yet verified (Req 3.9)


class CredentialStatusView(BaseModel):
    """A single source's status, safe to return and log (Requirements 3.3, 3.8).

    Carries the classification (Requirement 3.2), the resolved
    :class:`CredentialStatus`, and - for License-Needed sources - the
    ``license_reference`` echoed from the spec (Requirement 3.7). When the
    status is ``Unverifiable`` because validation timed out or failed,
    ``error_category`` records the ``Error_Taxonomy`` classification of that
    failure (Requirement 3.9).

    There is deliberately **no field that can hold a secret value**
    (Requirement 3.8): the view names *which* credential it describes via
    ``source`` / ``mcp_json_key`` only.
    """

    source: str
    mcp_json_key: str
    classification: CredentialClassification
    status: CredentialStatus
    license_reference: Optional[str] = None  # echoed for License-Needed (Req 3.7)
    error_category: Optional[ErrorCategory] = None  # set on Unverifiable (Req 3.9)


class CredentialRecord(BaseModel):
    """The Credential Manager's internal record for one managed source.

    Combines the source's :class:`CredentialSpec` with the manager's knowledge
    of whether a credential is *present* on the surface and the last resolved
    :class:`CredentialStatus`. Like every other model here it has **no
    secret-bearing field** (Requirement 3.8): ``present`` records only that a
    non-empty value exists for ``mcp_json_key``, never the value itself.
    """

    spec: CredentialSpec
    present: bool = False
    status: CredentialStatus = CredentialStatus.UNVERIFIABLE
    error_category: Optional[ErrorCategory] = None

    def to_view(self) -> CredentialStatusView:
        """Project this record onto an outward-facing :class:`CredentialStatusView`."""
        return CredentialStatusView(
            source=self.spec.source,
            mcp_json_key=self.spec.mcp_json_key,
            classification=self.spec.classification,
            status=self.status,
            # Echo the licensing reference only for License-Needed sources (Req 3.7).
            license_reference=(
                self.spec.license_reference
                if self.spec.classification
                is CredentialClassification.LICENSE_NEEDED
                else None
            ),
            error_category=self.error_category,
        )


class CredentialManager:
    """Records, classifies, validates, and reports credential status (Req 3).

    Parameters
    ----------
    specs:
        The credential specs to manage - one per source the Power Pack can
        reach. Each names an ``mcp.json`` key and its classification. Typically
        gathered from installed servers' ``required_credentials()`` and the
        bundle manifest. Specs carry no secret value (Requirement 3.8).
    surface:
        The single ``mcp.json`` credential surface, modeled as a read-only
        mapping of ``mcp.json`` key -> value. Defaults to the process
        environment, which is how MCP exposes ``mcp.json`` ``env`` values to a
        running server (Requirement 3.1). Values are read transiently for
        validation/redaction only and are never stored or emitted.
    validators:
        Optional mapping of source identifier -> :data:`Validator`. When a
        credential is present and a validator is registered for its source, the
        manager validates it against the source (Requirement 3.4). Sources with
        no validator can never reach ``Configured``/``Invalid`` and report
        ``Unverifiable`` instead (Requirement 3.9).
    """

    def __init__(
        self,
        specs: Optional[Iterable[CredentialSpec]] = None,
        *,
        surface: Optional[Mapping[str, str]] = None,
        validators: Optional[Mapping[str, Validator]] = None,
    ) -> None:
        self._surface: Mapping[str, str] = surface if surface is not None else os.environ
        self._validators: Dict[str, Validator] = dict(validators or {})
        self._records: Dict[str, CredentialRecord] = {}
        if specs is not None:
            for spec in specs:
                self.register_spec(spec)
        # Build the presence snapshot up front (Requirement 3.1).
        self.load()

    # ------------------------------------------------------------------
    # Registration + loading (Requirement 3.1)
    # ------------------------------------------------------------------

    def register_spec(self, spec: CredentialSpec) -> None:
        """Register a source's :class:`CredentialSpec` under its source id.

        Validates the License-Needed invariant: a License-Needed source must
        carry a ``license_reference`` so the status report can echo it
        (Requirement 3.7).
        """
        if (
            spec.classification is CredentialClassification.LICENSE_NEEDED
            and not spec.license_reference
        ):
            raise ValueError(
                "License-Needed source %r must declare a license_reference "
                "(Requirement 3.7)" % spec.source
            )
        self._records[spec.source] = CredentialRecord(spec=spec)

    def register_validator(self, source: str, validator: Validator) -> None:
        """Register the live validator for ``source`` (used by :meth:`validate`)."""
        self._validators[source] = validator

    def load(self) -> None:
        """Read presence for every managed source from the ``mcp.json`` surface.

        Records only *whether* a non-empty value exists for each
        ``mcp_json_key`` (Requirement 3.1); it never stores the value itself
        (Requirement 3.8). Absent credentials are marked ``Missing`` straight
        away (Requirement 3.5); present-but-unvalidated credentials remain
        ``Unverifiable`` until :meth:`validate` runs (Requirement 3.9). A
        config-read failure for a single source degrades only that source to
        ``Unverifiable`` (Requirement 3.9).
        """
        for source, record in self._records.items():
            try:
                present = self._is_present(record.spec.mcp_json_key)
            except Exception:
                # Reading the surface failed for this source only: degrade it
                # to Unverifiable, keep covering the rest (Requirement 3.9).
                record.present = False
                record.status = CredentialStatus.UNVERIFIABLE
                record.error_category = ErrorCategory.UPSTREAM
                continue
            record.present = present
            record.error_category = None
            record.status = (
                CredentialStatus.UNVERIFIABLE
                if present
                else CredentialStatus.MISSING
            )

    # ------------------------------------------------------------------
    # Classification (Requirement 3.2)
    # ------------------------------------------------------------------

    def sources(self) -> List[str]:
        """The identifiers of every managed source, in registration order."""
        return list(self._records.keys())

    def classify(self, source: str) -> CredentialClassification:
        """Return the single :class:`CredentialClassification` of ``source``.

        Each configured source belongs to exactly one classification
        (Requirement 3.2). Raises :class:`~geo_common.errors.GeoError`
        (``not-found``) for an unknown source.
        """
        return self._record(source).spec.classification

    # ------------------------------------------------------------------
    # Status reporting (Requirements 3.3, 3.7, 3.9)
    # ------------------------------------------------------------------

    def status_report(self) -> List[CredentialStatusView]:
        """Report each source's status without network I/O (Requirement 3.3).

        Returns a :class:`CredentialStatusView` for every managed source within
        the 5-second budget (Requirement 3.3) by reporting the manager's
        current knowledge: ``Missing`` for absent credentials (Requirement
        3.5), the last validated status for sources :meth:`validate` has
        confirmed, and ``Unverifiable`` for present-but-unvalidated sources
        (which cannot be verified synchronously - Requirement 3.9). License
        references are echoed for License-Needed sources (Requirement 3.7).

        A per-source failure never aborts the report; that source is reported
        ``Unverifiable`` while the report still covers every other source
        (Requirement 3.9). No secret value appears in the output (Requirement
        3.8).
        """
        views: List[CredentialStatusView] = []
        for record in self._records.values():
            try:
                # Refresh presence cheaply so a credential added/removed since
                # construction is reflected (still no network, well within 5s).
                self._refresh_presence(record)
                views.append(record.to_view())
            except Exception:
                # Building one view failed: degrade only this source, keep
                # covering the rest (Requirement 3.9).
                views.append(
                    CredentialStatusView(
                        source=record.spec.source,
                        mcp_json_key=record.spec.mcp_json_key,
                        classification=record.spec.classification,
                        status=CredentialStatus.UNVERIFIABLE,
                        license_reference=(
                            record.spec.license_reference
                            if record.spec.classification
                            is CredentialClassification.LICENSE_NEEDED
                            else None
                        ),
                        error_category=ErrorCategory.UPSTREAM,
                    )
                )
        return views

    async def report(self) -> List[CredentialStatusView]:
        """Validate all present sources concurrently, then report (Req 3.3/3.4/3.9).

        A live status report: every present credential is validated against its
        source concurrently, each bounded by the 10-second validation timeout
        (Requirement 3.4); absent credentials are reported ``Missing``
        (Requirement 3.5). One source's validation timing out or failing
        degrades only that source to ``Unverifiable`` while the report still
        covers every other source (Requirement 3.9).
        """
        # Validate present sources concurrently; return_exceptions so one
        # failure cannot abort coverage of the others (Requirement 3.9).
        present_sources = [
            source
            for source, record in self._records.items()
            if record.present
        ]
        await asyncio.gather(
            *(self.validate(source) for source in present_sources),
            return_exceptions=True,
        )
        return self.status_report()

    # ------------------------------------------------------------------
    # Validation (Requirements 3.4, 3.5, 3.9)
    # ------------------------------------------------------------------

    async def validate(self, source: str) -> CredentialStatus:
        """Validate ``source``'s credential against the source (Req 3.4/3.5/3.9).

        * Absent credential -> ``Missing`` (Requirement 3.5).
        * Present credential validated within 10s: ``Configured`` if the source
          accepts it, ``Invalid`` if the source rejects it (Requirement 3.4).
        * Validation timeout, validator failure, or no registered validator ->
          ``Unverifiable``, classified per the ``Error_Taxonomy`` (Requirement
          3.9).

        The resolved status is cached on the source's record so a subsequent
        :meth:`status_report` reflects it. The secret value is read transiently
        and passed to the validator only; it is never stored or logged
        (Requirement 3.8).
        """
        record = self._record(source)
        self._refresh_presence(record)

        if not record.present:
            record.status = CredentialStatus.MISSING
            record.error_category = None
            return record.status

        validator = self._validators.get(source)
        if validator is None:
            # Present but no way to check it against the source within budget.
            record.status = CredentialStatus.UNVERIFIABLE
            record.error_category = ErrorCategory.UPSTREAM
            return record.status

        try:
            secret = self._read_secret(record.spec.mcp_json_key)
            accepted = await asyncio.wait_for(
                self._invoke_validator(validator, secret),
                timeout=_VALIDATION_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            # Did not complete within 10s (Requirement 3.9).
            record.status = CredentialStatus.UNVERIFIABLE
            record.error_category = ErrorCategory.NETWORK
            return record.status
        except GeoError as exc:
            # Validator already mapped onto the taxonomy: keep its category.
            record.status = CredentialStatus.UNVERIFIABLE
            record.error_category = exc.category
            return record.status
        except Exception:
            # Any other validator failure -> Unverifiable, upstream (Req 3.9).
            record.status = CredentialStatus.UNVERIFIABLE
            record.error_category = ErrorCategory.UPSTREAM
            return record.status

        record.status = (
            CredentialStatus.CONFIGURED if accepted else CredentialStatus.INVALID
        )
        record.error_category = None
        return record.status

    # ------------------------------------------------------------------
    # Invocation guard (Requirement 3.6)
    # ------------------------------------------------------------------

    def guard_invocation(self, source: str) -> None:
        """Deny a Required-but-Missing invocation (Requirement 3.6).

        If ``source`` is classified :attr:`CredentialClassification.REQUIRED`
        and no credential is present, raise an
        :class:`~geo_common.errors.AuthenticationError` (taxonomy
        ``authentication``) that names the missing credential and its
        ``mcp.json`` configuration key, without performing the capability. For
        any other classification, or when the credential is present, the guard
        permits the invocation and returns ``None``.
        """
        record = self._record(source)
        self._refresh_presence(record)
        if (
            record.spec.classification is CredentialClassification.REQUIRED
            and not record.present
        ):
            key = record.spec.mcp_json_key
            raise AuthenticationError(
                "required credential for source %r is missing; configure the "
                "%r key in mcp.json" % (source, key),
                source=source,
                detail={"missing_key": key, "source": source},
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record(self, source: str) -> CredentialRecord:
        try:
            return self._records[source]
        except KeyError:
            raise GeoError(
                ErrorCategory.NOT_FOUND,
                "no credential is managed for source %r" % source,
                source=source,
            )

    def _refresh_presence(self, record: CredentialRecord) -> None:
        """Re-read presence for ``record`` and reconcile cached status.

        Keeps the synchronous status fast (no network). If a credential has
        become absent, the cached status is reset to ``Missing``; if it has
        appeared since the last validation, it becomes ``Unverifiable`` until
        validated.
        """
        present = self._is_present(record.spec.mcp_json_key)
        if present == record.present:
            return
        record.present = present
        if not present:
            record.status = CredentialStatus.MISSING
            record.error_category = None
        else:
            record.status = CredentialStatus.UNVERIFIABLE
            record.error_category = None

    def _is_present(self, key: str) -> bool:
        """True when a non-empty value exists for ``key`` on the surface."""
        value = self._surface.get(key)
        return bool(value) and bool(str(value).strip())

    def _read_secret(self, key: str) -> str:
        """Read the secret value for ``key`` transiently (never stored/logged)."""
        return str(self._surface.get(key) or "")

    @staticmethod
    async def _invoke_validator(validator: Validator, secret: str) -> bool:
        """Invoke a sync or async validator and coerce its result to ``bool``."""
        result = validator(secret)
        if asyncio.iscoroutine(result):
            result = await result
        return bool(result)

    def secret_values(self) -> List[str]:
        """Current non-empty secret values on the surface, for redaction only.

        Used by :class:`CredentialRedactionFilter` to scrub secrets out of log
        records. The values are read transiently and never persisted on the
        manager (Requirement 3.8).
        """
        values: List[str] = []
        for record in self._records.values():
            value = self._surface.get(record.spec.mcp_json_key)
            if value and str(value).strip():
                values.append(str(value))
        return values

    def managed_keys(self) -> List[str]:
        """The ``mcp.json`` key names this manager knows about (for redaction)."""
        return [record.spec.mcp_json_key for record in self._records.values()]


class CredentialRedactionFilter(logging.Filter):
    """A logging filter that scrubs credential secrets out of log records (Req 3.8).

    Attach this to any logger/handler that might process Credential Manager
    output so that, even if a secret value or a ``KEY=value`` assignment reaches
    a log call, it is replaced with a redaction marker before the record is
    emitted. Two complementary strategies are applied:

    * **Value redaction** - any current secret value on the manager's surface
      that appears verbatim in the formatted message is replaced. Secret values
      are read transiently from the surface at filter time and never stored on
      the filter (Requirement 3.8).
    * **Key-assignment redaction** - occurrences of ``<KEY>=...`` or
      ``"<KEY>": "..."`` for any managed key (and common secret-bearing key
      patterns) are masked, covering secrets the manager does not directly hold.

    The filter mutates the record's ``msg``/``args`` in place and always returns
    ``True`` so the (now-redacted) record is still emitted.
    """

    REDACTION = "***REDACTED***"

    #: Key-name fragments that conventionally carry secrets.
    _SECRETISH = re.compile(
        r"(?i)(key|token|secret|password|passwd|credential|client_secret|api_key)"
    )

    def __init__(
        self,
        manager: Optional[CredentialManager] = None,
        *,
        extra_keys: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()
        self._manager = manager
        self._extra_keys = list(extra_keys or [])

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 - logging API
        try:
            message = record.getMessage()
        except Exception:
            # If the record can't even be formatted, drop args defensively so
            # nothing unredacted slips through, and keep the record.
            record.msg = self.REDACTION
            record.args = None
            return True

        redacted = self._redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = None
        return True

    def _redact(self, text: str) -> str:
        # 1) Redact verbatim secret values (longest first so substrings of a
        #    longer secret don't leave fragments behind).
        for value in sorted(self._secret_values(), key=len, reverse=True):
            if value and value in text:
                text = text.replace(value, self.REDACTION)

        # 2) Redact KEY=value / "KEY": "value" assignments for known and
        #    secret-looking keys.
        keys = set(self._managed_keys()) | set(self._extra_keys)
        for key in keys:
            text = self._redact_assignment(text, re.escape(key))

        # 3) Generic safety net for any secret-looking key name.
        text = re.sub(
            r'(?i)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)[A-Z0-9_]*)'
            r'(\s*[=:]\s*)("?)([^\s",}]+)(\3)',
            lambda m: m.group(1) + m.group(2) + m.group(3) + self.REDACTION + m.group(5),
            text,
        )
        return text

    @staticmethod
    def _redact_assignment(text: str, key_pattern: str) -> str:
        # KEY=value  (env-style)  and  "KEY": "value" / KEY: value (json-ish)
        text = re.sub(
            r'(' + key_pattern + r'\s*=\s*)([^\s,;]+)',
            lambda m: m.group(1) + CredentialRedactionFilter.REDACTION,
            text,
        )
        text = re.sub(
            r'(["\']?' + key_pattern + r'["\']?\s*:\s*)("?)([^\s",}]+)(\2)',
            lambda m: m.group(1) + m.group(2) + CredentialRedactionFilter.REDACTION + m.group(4),
            text,
        )
        return text

    def _secret_values(self) -> List[str]:
        if self._manager is None:
            return []
        try:
            return self._manager.secret_values()
        except Exception:
            return []

    def _managed_keys(self) -> List[str]:
        if self._manager is None:
            return []
        try:
            return self._manager.managed_keys()
        except Exception:
            return []

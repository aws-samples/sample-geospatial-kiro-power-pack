"""Example-based unit tests for the Power Hub Credential Manager (Requirement 3).

Covers, against mocked validators and a mocked ``mcp.json`` surface:

* status reporting for every configured source (Requirement 3.3), including the
  License-Needed license-reference echo (Requirement 3.7);
* validation accept/reject -> Configured / Invalid (Requirement 3.4);
* the Missing status when no credential is present (Requirement 3.5);
* Unverifiable degradation on validation timeout, validator failure, a
  missing validator, and a per-source config-read failure, while the report
  still covers every other source (Requirement 3.9);
* the Required-but-Missing invocation guard raising an ``AuthenticationError``
  that names the missing credential and its ``mcp.json`` key (Requirement 3.6).

These are example-based unit tests; Property 7 (secrets never emitted) is
covered separately by ``test_credential_redaction_property.py``. The repo's
``asyncio_mode = "auto"`` lets the ``async def`` tests run directly.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Mapping, Optional

import pytest

from geo_common.errors import (
    AuthenticationError,
    ErrorCategory,
    GeoError,
    RateLimitError,
)
from geo_common.models import CredentialClassification, CredentialSpec
from kiro_geospatial import credentials as credentials_module
from kiro_geospatial.credentials import (
    CredentialManager,
    CredentialStatus,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
REQUIRED = CredentialClassification.REQUIRED
OPTIONAL = CredentialClassification.OPTIONAL
LICENSE_NEEDED = CredentialClassification.LICENSE_NEEDED


def _spec(
    source: str,
    key: str,
    classification: CredentialClassification = REQUIRED,
    license_reference: Optional[str] = None,
) -> CredentialSpec:
    return CredentialSpec(
        source=source,
        mcp_json_key=key,
        classification=classification,
        license_reference=license_reference,
    )


def _manager(
    specs: List[CredentialSpec],
    surface: Optional[Mapping[str, str]] = None,
    validators: Optional[Dict[str, object]] = None,
) -> CredentialManager:
    return CredentialManager(
        specs,
        surface=dict(surface or {}),
        validators=validators or {},
    )


def _view_by_source(views, source):
    return next(v for v in views if v.source == source)


# --------------------------------------------------------------------------- #
# Status reporting (Requirement 3.3, with 3.7 license echo)
# --------------------------------------------------------------------------- #
def test_status_report_covers_every_source():
    manager = _manager(
        [
            _spec("alpha", "ALPHA_KEY", REQUIRED),
            _spec("beta", "BETA_KEY", OPTIONAL),
            _spec("gamma", "GAMMA_KEY", LICENSE_NEEDED, license_reference="lic://gamma"),
        ],
        surface={"ALPHA_KEY": "present"},
    )

    views = manager.status_report()

    assert {v.source for v in views} == {"alpha", "beta", "gamma"}
    # Each view carries the source's single classification (Requirement 3.2/3.3).
    assert _view_by_source(views, "alpha").classification is REQUIRED
    assert _view_by_source(views, "beta").classification is OPTIONAL
    assert _view_by_source(views, "gamma").classification is LICENSE_NEEDED


def test_status_report_status_is_one_of_the_four_values():
    manager = _manager(
        [_spec("alpha", "ALPHA_KEY", REQUIRED)],
        surface={"ALPHA_KEY": "present"},
    )
    [view] = manager.status_report()
    assert view.status in {
        CredentialStatus.CONFIGURED,
        CredentialStatus.INVALID,
        CredentialStatus.MISSING,
        CredentialStatus.UNVERIFIABLE,
    }


def test_present_but_unvalidated_source_reports_unverifiable():
    # Present credential, but status_report performs no network I/O so it
    # cannot synchronously confirm validity (Requirement 3.9).
    manager = _manager(
        [_spec("alpha", "ALPHA_KEY", REQUIRED)],
        surface={"ALPHA_KEY": "present"},
    )
    [view] = manager.status_report()
    assert view.status is CredentialStatus.UNVERIFIABLE


def test_license_reference_echoed_only_for_license_needed():
    manager = _manager(
        [
            _spec("gamma", "GAMMA_KEY", LICENSE_NEEDED, license_reference="lic://gamma"),
            _spec("alpha", "ALPHA_KEY", REQUIRED),
        ],
        surface={"GAMMA_KEY": "present", "ALPHA_KEY": "present"},
    )
    views = manager.status_report()
    assert _view_by_source(views, "gamma").license_reference == "lic://gamma"
    # Non-License-Needed sources never echo a reference (Requirement 3.7).
    assert _view_by_source(views, "alpha").license_reference is None


def test_classify_returns_single_classification():
    manager = _manager([_spec("alpha", "ALPHA_KEY", OPTIONAL)])
    assert manager.classify("alpha") is OPTIONAL


def test_classify_unknown_source_raises_not_found():
    manager = _manager([_spec("alpha", "ALPHA_KEY", REQUIRED)])
    with pytest.raises(GeoError) as exc_info:
        manager.classify("does-not-exist")
    assert exc_info.value.category is ErrorCategory.NOT_FOUND


# --------------------------------------------------------------------------- #
# Validation accept / reject (Requirement 3.4)
# --------------------------------------------------------------------------- #
async def test_validate_configured_when_source_accepts():
    manager = _manager(
        [_spec("alpha", "ALPHA_KEY", REQUIRED)],
        surface={"ALPHA_KEY": "good-secret"},
        validators={"alpha": lambda secret: True},
    )
    status = await manager.validate("alpha")
    assert status is CredentialStatus.CONFIGURED
    # Cached so a subsequent synchronous report reflects it.
    assert manager.status_report()[0].status is CredentialStatus.CONFIGURED


async def test_validate_invalid_when_source_rejects():
    manager = _manager(
        [_spec("alpha", "ALPHA_KEY", REQUIRED)],
        surface={"ALPHA_KEY": "bad-secret"},
        validators={"alpha": lambda secret: False},
    )
    status = await manager.validate("alpha")
    assert status is CredentialStatus.INVALID
    assert manager.status_report()[0].status is CredentialStatus.INVALID


async def test_validate_supports_async_validator():
    async def accepts(secret: str) -> bool:
        await asyncio.sleep(0)
        return True

    manager = _manager(
        [_spec("alpha", "ALPHA_KEY", REQUIRED)],
        surface={"ALPHA_KEY": "good-secret"},
        validators={"alpha": accepts},
    )
    assert await manager.validate("alpha") is CredentialStatus.CONFIGURED


async def test_validator_receives_the_secret_value():
    seen: List[str] = []

    def validator(secret: str) -> bool:
        seen.append(secret)
        return True

    manager = _manager(
        [_spec("alpha", "ALPHA_KEY", REQUIRED)],
        surface={"ALPHA_KEY": "transient-secret"},
        validators={"alpha": validator},
    )
    await manager.validate("alpha")
    assert seen == ["transient-secret"]


async def test_report_validates_present_sources_and_marks_missing():
    manager = _manager(
        [
            _spec("alpha", "ALPHA_KEY", REQUIRED),
            _spec("beta", "BETA_KEY", REQUIRED),
        ],
        surface={"ALPHA_KEY": "good"},  # beta absent
        validators={
            "alpha": lambda secret: True,
            "beta": lambda secret: True,
        },
    )
    views = await manager.report()
    assert _view_by_source(views, "alpha").status is CredentialStatus.CONFIGURED
    assert _view_by_source(views, "beta").status is CredentialStatus.MISSING


# --------------------------------------------------------------------------- #
# Missing (Requirement 3.5)
# --------------------------------------------------------------------------- #
async def test_validate_missing_when_no_credential_present():
    manager = _manager(
        [_spec("alpha", "ALPHA_KEY", REQUIRED)],
        surface={},  # nothing configured
        validators={"alpha": lambda secret: True},
    )
    status = await manager.validate("alpha")
    assert status is CredentialStatus.MISSING


def test_status_report_marks_absent_credential_missing():
    manager = _manager(
        [_spec("alpha", "ALPHA_KEY", REQUIRED)],
        surface={},
    )
    [view] = manager.status_report()
    assert view.status is CredentialStatus.MISSING


async def test_empty_string_credential_is_treated_as_missing():
    manager = _manager(
        [_spec("alpha", "ALPHA_KEY", REQUIRED)],
        surface={"ALPHA_KEY": "   "},  # whitespace-only counts as absent
        validators={"alpha": lambda secret: True},
    )
    assert await manager.validate("alpha") is CredentialStatus.MISSING


# --------------------------------------------------------------------------- #
# Unverifiable degradation (Requirement 3.9)
# --------------------------------------------------------------------------- #
async def test_validate_unverifiable_when_no_validator_registered():
    manager = _manager(
        [_spec("alpha", "ALPHA_KEY", REQUIRED)],
        surface={"ALPHA_KEY": "present"},
        validators={},  # present, but nothing can check it
    )
    status = await manager.validate("alpha")
    assert status is CredentialStatus.UNVERIFIABLE
    assert manager.status_report()[0].error_category is ErrorCategory.UPSTREAM


async def test_validate_unverifiable_on_validator_failure():
    def boom(secret: str) -> bool:
        raise RuntimeError("source unavailable")

    manager = _manager(
        [_spec("alpha", "ALPHA_KEY", REQUIRED)],
        surface={"ALPHA_KEY": "present"},
        validators={"alpha": boom},
    )
    status = await manager.validate("alpha")
    assert status is CredentialStatus.UNVERIFIABLE
    assert manager.status_report()[0].error_category is ErrorCategory.UPSTREAM


async def test_validate_unverifiable_keeps_geoerror_category():
    def rate_limited(secret: str) -> bool:
        raise RateLimitError("slow down", source="alpha")

    manager = _manager(
        [_spec("alpha", "ALPHA_KEY", REQUIRED)],
        surface={"ALPHA_KEY": "present"},
        validators={"alpha": rate_limited},
    )
    status = await manager.validate("alpha")
    assert status is CredentialStatus.UNVERIFIABLE
    # The taxonomy category mapped by the validator is retained (Requirement 3.9).
    assert manager.status_report()[0].error_category is ErrorCategory.RATE_LIMIT


async def test_validate_unverifiable_on_timeout(monkeypatch):
    # Shrink the validation budget so the test does not actually wait 10s.
    monkeypatch.setattr(credentials_module, "_VALIDATION_TIMEOUT_S", 0.05)

    async def too_slow(secret: str) -> bool:
        await asyncio.sleep(1.0)
        return True

    manager = _manager(
        [_spec("alpha", "ALPHA_KEY", REQUIRED)],
        surface={"ALPHA_KEY": "present"},
        validators={"alpha": too_slow},
    )
    status = await manager.validate("alpha")
    assert status is CredentialStatus.UNVERIFIABLE
    # A timeout is classified as a network failure (Requirement 3.9).
    assert manager.status_report()[0].error_category is ErrorCategory.NETWORK


def test_status_report_degrades_only_failing_source_on_config_read_error():
    class _ExplodingSurface(dict):
        """A surface whose read fails for one specific key only."""

        def get(self, key, default=None):  # type: ignore[override]
            if key == "BETA_KEY":
                raise OSError("cannot read config for beta")
            return super().get(key, default)

    surface = _ExplodingSurface({"ALPHA_KEY": "present"})
    manager = CredentialManager(
        [
            _spec("alpha", "ALPHA_KEY", REQUIRED),
            _spec("beta", "BETA_KEY", REQUIRED),
            _spec("delta", "DELTA_KEY", REQUIRED),
        ],
        surface=surface,
        validators={},
    )

    views = manager.status_report()

    # The report still covers every source despite beta's read failing.
    assert {v.source for v in views} == {"alpha", "beta", "delta"}
    beta = _view_by_source(views, "beta")
    assert beta.status is CredentialStatus.UNVERIFIABLE
    assert beta.error_category is ErrorCategory.UPSTREAM
    # The other sources are unaffected.
    assert _view_by_source(views, "delta").status is CredentialStatus.MISSING


async def test_report_isolates_one_sources_failure_from_the_rest():
    def boom(secret: str) -> bool:
        raise RuntimeError("alpha source is down")

    manager = _manager(
        [
            _spec("alpha", "ALPHA_KEY", REQUIRED),
            _spec("beta", "BETA_KEY", REQUIRED),
        ],
        surface={"ALPHA_KEY": "present", "BETA_KEY": "present"},
        validators={"alpha": boom, "beta": lambda secret: True},
    )

    views = await manager.report()

    assert _view_by_source(views, "alpha").status is CredentialStatus.UNVERIFIABLE
    # beta is still validated successfully (Requirement 3.9 coverage).
    assert _view_by_source(views, "beta").status is CredentialStatus.CONFIGURED


# --------------------------------------------------------------------------- #
# Required-but-Missing invocation guard (Requirement 3.6)
# --------------------------------------------------------------------------- #
def test_guard_denies_required_but_missing_invocation():
    manager = _manager(
        [_spec("alpha", "ALPHA_KEY", REQUIRED)],
        surface={},  # missing
    )
    with pytest.raises(AuthenticationError) as exc_info:
        manager.guard_invocation("alpha")

    err = exc_info.value
    assert err.category is ErrorCategory.AUTHENTICATION
    # Names the missing credential and its mcp.json key, never a secret.
    assert "alpha" in str(err)
    assert "ALPHA_KEY" in str(err)
    assert err.detail == {"missing_key": "ALPHA_KEY", "source": "alpha"}


def test_guard_permits_required_when_credential_present():
    manager = _manager(
        [_spec("alpha", "ALPHA_KEY", REQUIRED)],
        surface={"ALPHA_KEY": "present"},
    )
    # No exception -> invocation permitted.
    assert manager.guard_invocation("alpha") is None


def test_guard_permits_optional_source_even_when_missing():
    manager = _manager(
        [_spec("beta", "BETA_KEY", OPTIONAL)],
        surface={},
    )
    assert manager.guard_invocation("beta") is None


def test_guard_permits_license_needed_source_even_when_missing():
    manager = _manager(
        [_spec("gamma", "GAMMA_KEY", LICENSE_NEEDED, license_reference="lic://gamma")],
        surface={},
    )
    assert manager.guard_invocation("gamma") is None


def test_guard_unknown_source_raises_not_found():
    manager = _manager([_spec("alpha", "ALPHA_KEY", REQUIRED)])
    with pytest.raises(GeoError) as exc_info:
        manager.guard_invocation("nope")
    assert exc_info.value.category is ErrorCategory.NOT_FOUND

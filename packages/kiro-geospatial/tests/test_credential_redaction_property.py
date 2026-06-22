"""Property test: credential secrets are never emitted.

Feature: geospatial-power-pack, Property 7: Credential secrets are never emitted

This module validates the Credential Manager (Requirement 3.8) against
Property 7 of the design:

    *For any* credential configuration and any output the Credential Manager
    produces (status report, error response, or log record), no credential
    secret value appears in the output.

The Credential Manager enforces this two ways (see design.md "Power Hub ->
Credential Manager" and ``kiro_geospatial.credentials``): its data model has
**no field that can hold a secret**, and :class:`CredentialRedactionFilter`
scrubs secret values (and ``KEY=value`` assignments) out of log records. This
test exercises all three output surfaces named by the property -

* the synchronous :meth:`CredentialManager.status_report` and the live
  :meth:`CredentialManager.report` (status reports),
* the :class:`~geo_common.errors.AuthenticationError` raised by
  :meth:`CredentialManager.guard_invocation` and its serialized
  :class:`~geo_common.errors.ErrorObject` (error responses), and
* log records routed through :class:`CredentialRedactionFilter` (log output) -

and asserts that no configured secret value survives in any of them, across
many randomly generated credential configurations.

The testing framework is Hypothesis (design.md Testing Strategy); the loaded
profile guarantees at least 100 generated cases per property (Requirement
15.1).
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Dict, List, Optional

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel

from geo_common.models import CredentialClassification, CredentialSpec
from kiro_geospatial.credentials import (
    CredentialManager,
    CredentialRedactionFilter,
)

# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
# Secret values are tagged with a distinctive uppercase marker so that (a) a
# detected appearance in any output is unambiguously a real leak rather than a
# coincidental substring of a field name / enum value, and (b) the marker can
# never be produced by the lowercase identifier strategies used for the
# non-secret structural fields (source / key / license reference). This keeps
# the property check both meaningful and free of false positives.
_SECRET_MARKER = "SeCrEtVaLuE"

_secret_body = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    min_size=6,
    max_size=32,
)
#: A non-empty, distinctive secret value (e.g. an API key/token).
secret_values = _secret_body.map(lambda body: _SECRET_MARKER + body)

#: Lowercase identifiers for the non-secret structural fields. Disjoint from
#: the (uppercase-marked) secret alphabet by construction.
_ident_body = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_", max_size=10)


class _Config(BaseModel):
    """One generated credential configuration row."""

    source: str
    mcp_json_key: str
    classification: CredentialClassification
    license_reference: Optional[str]
    secret: Optional[str]  # the (present) secret value, or None when Missing


@st.composite
def credential_configs(draw) -> List[_Config]:
    """Generate a credential configuration: a set of distinctly-keyed sources.

    Each source draws a classification, whether a credential is present, and -
    when present - a distinctive secret value. License-Needed sources always
    carry a (non-secret) license reference, as the model requires. Source ids
    and ``mcp.json`` keys are made unique via their row index so the manager
    registers every generated source.
    """
    count = draw(st.integers(min_value=1, max_value=6))
    configs: List[_Config] = []
    for i in range(count):
        classification = draw(st.sampled_from(list(CredentialClassification)))
        present = draw(st.booleans())
        secret = draw(secret_values) if present else None
        license_reference = (
            "license-ref-" + draw(_ident_body)
            if classification is CredentialClassification.LICENSE_NEEDED
            else draw(st.one_of(st.none(), st.builds(lambda b: "lic-" + b, _ident_body)))
        )
        configs.append(
            _Config(
                source=f"source-{i}-" + draw(_ident_body),
                mcp_json_key=f"MCP_KEY_{i}_" + draw(_ident_body),
                classification=classification,
                license_reference=license_reference,
                secret=secret,
            )
        )
    return configs


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _build_manager(configs: List[_Config]) -> CredentialManager:
    specs = [
        CredentialSpec(
            source=c.source,
            mcp_json_key=c.mcp_json_key,
            classification=c.classification,
            license_reference=c.license_reference,
        )
        for c in configs
    ]
    surface: Dict[str, str] = {
        c.mcp_json_key: c.secret for c in configs if c.secret is not None
    }
    # A validator per present source so report() exercises the live
    # Configured/Invalid path; it receives the transient secret but the
    # manager's output must still never contain it.
    validators = {
        c.source: (lambda secret: len(secret) % 2 == 0)
        for c in configs
        if c.secret is not None
    }
    return CredentialManager(specs, surface=surface, validators=validators)


def _present_secrets(configs: List[_Config]) -> List[str]:
    return [c.secret for c in configs if c.secret]


def _assert_secret_free(text: str, secrets: List[str], where: str) -> None:
    for secret in secrets:
        assert secret not in text, (
            f"credential secret leaked into {where} (Requirement 3.8): "
            f"found {secret!r} in {text!r}"
        )


# --------------------------------------------------------------------------- #
# Property 7
# --------------------------------------------------------------------------- #
@pytest.mark.property
@settings(deadline=None)  # max_examples (>=100) inherited from the loaded profile
@given(configs=credential_configs())
def test_credential_secrets_are_never_emitted(configs: List[_Config]) -> None:
    """Feature: geospatial-power-pack, Property 7: Credential secrets are never emitted.

    Validates: Requirements 3.8
    """
    secrets = _present_secrets(configs)
    manager = _build_manager(configs)

    # ---- (1) status report (synchronous, no network) ----------------------
    sync_views = manager.status_report()
    for view in sync_views:
        _assert_secret_free(view.model_dump_json(), secrets, "status_report() JSON")
        _assert_secret_free(str(view), secrets, "status_report() str")
        _assert_secret_free(repr(view), secrets, "status_report() repr")

    # ---- (2) live status report (validates present sources) ---------------
    live_views = asyncio.run(manager.report())
    for view in live_views:
        _assert_secret_free(view.model_dump_json(), secrets, "report() JSON")
        _assert_secret_free(repr(view), secrets, "report() repr")

    # ---- (3) error responses (invocation guard) ---------------------------
    # Guarding every source raises only for Required-but-Missing sources; any
    # error produced must name the key, never the secret value.
    from geo_common.errors import GeoError

    for config in configs:
        try:
            manager.guard_invocation(config.source)
        except GeoError as exc:
            error_text = " ".join(
                [
                    str(exc),
                    repr(exc),
                    exc.to_error_object().model_dump_json(),
                ]
            )
            _assert_secret_free(error_text, secrets, "error response")

    # ---- (4) log records routed through the redaction filter --------------
    # Even if a secret value or a KEY=value assignment reaches a log call, the
    # CredentialRedactionFilter must scrub it before the record is emitted.
    if secrets:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
        handler.addFilter(CredentialRedactionFilter(manager))
        logger = logging.getLogger(f"cred-redaction-test-{id(configs)}")
        logger.handlers = [handler]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        for config in configs:
            if not config.secret:
                continue
            secret = config.secret
            key = config.mcp_json_key
            # Verbatim secret in the message.
            logger.info("raw secret %s here", secret)
            # Env-style and json-style key assignments.
            logger.warning("%s=%s", key, secret)
            logger.error('config {"%s": "%s"}', key, secret)
            # Secret embedded directly as the message.
            logger.debug("%s", secret)

        handler.flush()
        _assert_secret_free(stream.getvalue(), secrets, "log output")

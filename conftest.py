"""Shared pytest configuration and fixtures for the Geospatial Power Pack.

This root ``conftest.py`` is loaded by pytest for every test under
``packages/*``. It serves two purposes:

1. Registers and loads a Hypothesis profile that enforces a **minimum of 100
   generated cases per property test** (Requirement 15.1) so that every
   property test in the monorepo inherits the minimum without having to repeat
   ``@settings(max_examples=...)`` itself.
2. Provides shared fixtures for the two mock backends the design's Testing
   Strategy calls out: an ``httpx`` mock transport (for retry/backoff/
   rate-limit HTTP behavior) and AWS mocking via ``moto``/``localstack``.

Imports of optional test dependencies are kept lazy/guarded so that this
conftest never breaks collection of plain unit tests when a given backend is
not installed in the current environment.
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Hypothesis: enforce >= 100 generated cases per property test (Requirement 15.1)
# ---------------------------------------------------------------------------
# The profile is loaded at import time (i.e. as soon as pytest imports this
# conftest) so that the setting is active before any property test module is
# collected. Selecting a richer profile in CI is possible via the
# HYPOTHESIS_PROFILE environment variable without editing any test.
HYPOTHESIS_MIN_EXAMPLES = 100
DEFAULT_HYPOTHESIS_PROFILE = "geospatial-power-pack"

try:
    from hypothesis import HealthCheck, settings

    # The baseline profile required by Requirement 15.1: at least 100 cases.
    settings.register_profile(
        DEFAULT_HYPOTHESIS_PROFILE,
        max_examples=HYPOTHESIS_MIN_EXAMPLES,
    )
    # An opt-in deeper search for CI runs; never goes below the minimum.
    settings.register_profile(
        "ci",
        max_examples=max(1000, HYPOTHESIS_MIN_EXAMPLES),
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )

    # Load the profile named by HYPOTHESIS_PROFILE, defaulting to the
    # >=100-case profile so every property test inherits the minimum.
    settings.load_profile(
        os.environ.get("HYPOTHESIS_PROFILE", DEFAULT_HYPOTHESIS_PROFILE)
    )
    _HYPOTHESIS_AVAILABLE = True
except ImportError:  # pragma: no cover - hypothesis is a declared test dependency
    _HYPOTHESIS_AVAILABLE = False


# ---------------------------------------------------------------------------
# httpx mock transport fixtures (HTTP retry/backoff/rate-limit behavior)
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_http_handler():
    """Default ``httpx.MockTransport`` handler.

    Returns an empty ``200`` JSON response. Override in a test by providing a
    custom handler and building a transport with ``make_mock_transport``.
    """
    import httpx

    def handler(request: "httpx.Request") -> "httpx.Response":
        return httpx.Response(200, json={})

    return handler


@pytest.fixture
def make_mock_transport():
    """Factory that wraps a request handler in an ``httpx.MockTransport``."""
    import httpx

    def _factory(handler):
        return httpx.MockTransport(handler)

    return _factory


@pytest.fixture
def mock_transport(make_mock_transport, mock_http_handler):
    """A ready-to-use ``httpx.MockTransport`` returning empty 200 responses."""
    return make_mock_transport(mock_http_handler)


@pytest.fixture
async def mock_async_client(mock_transport):
    """An ``httpx.AsyncClient`` wired to the mock transport.

    Requires ``asyncio_mode = "auto"`` (configured in pyproject.toml).
    """
    import httpx

    async with httpx.AsyncClient(transport=mock_transport) as client:
        yield client


# ---------------------------------------------------------------------------
# AWS mocking fixtures (moto / localstack)
# ---------------------------------------------------------------------------
@pytest.fixture
def aws_credentials(monkeypatch):
    """Set dummy AWS credentials so moto/boto3 never touch real AWS.

    This is the standard moto guard fixture; depend on it before activating any
    moto mock so that an accidental real-AWS call cannot succeed.
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    yield


@pytest.fixture
def aws_mock(aws_credentials):
    """Activate moto's mock for all supported AWS services.

    Yields the active mock context so tests can create boto3 clients/resources
    that are fully intercepted by moto.
    """
    moto = pytest.importorskip("moto")
    with moto.mock_aws():
        yield

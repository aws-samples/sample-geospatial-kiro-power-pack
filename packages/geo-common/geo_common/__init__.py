"""geo-common: the shared base package for the Geospatial Power Pack.

Every MCP server depends on ``geo-common`` for a single async HTTP client with
retry/backoff, rate-limit handling, the shared :class:`Error_Taxonomy`, and the
``BaseGeoServer`` contract.

This module re-exports the package's public surfaces so servers can do::

    from geo_common import (
        ErrorCategory, GeoError, ValidationError,   # Error_Taxonomy
        RetryPolicy, backoff_schedule,              # retry / backoff
        HttpClient,                                 # async HTTP client
        BaseGeoServer,                              # shared server contract
        CatalogEntry, CredentialSpec,               # base-contract models
    )

As further surfaces land in their own modules they are added here and to
``__all__`` below.
"""

from __future__ import annotations

from geo_common.errors import (
    AuthenticationError,
    AuthorizationError,
    ErrorCategory,
    ErrorObject,
    GeoError,
    NetworkError,
    NotFoundError,
    RateLimitError,
    UpstreamError,
    ValidationError,
)
from geo_common.http import HttpClient
from geo_common.models import (
    CatalogEntry,
    CredentialClassification,
    CredentialSpec,
    OpennessTier,
)
from geo_common.retry import RetryPolicy, backoff_schedule
from geo_common.runtime import (
    build_tool_input_schema,
    coerce_tool_arguments,
    run_server,
    serve_stdio,
    tool_result_to_text,
)
from geo_common.server import BaseGeoServer, RegisteredTool

__all__ = [
    # Error_Taxonomy (Req 5.5, 5.10, 11.5)
    "ErrorCategory",
    "GeoError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitError",
    "NotFoundError",
    "ValidationError",
    "UpstreamError",
    "NetworkError",
    "ErrorObject",
    # Retry / backoff policy (Req 5.2, 5.3, 5.8)
    "RetryPolicy",
    "backoff_schedule",
    # Async HTTP client (Req 5.1, 5.4, 5.6, 5.7, 5.9)
    "HttpClient",
    # Shared base-contract models (Req 2.1, 11.3, 16.1)
    "OpennessTier",
    "CredentialClassification",
    "CredentialSpec",
    "CatalogEntry",
    # BaseGeoServer contract (Req 5.1, 6.3, 11.5, 16.4, 16.5, 16.6)
    "BaseGeoServer",
    "RegisteredTool",
    # MCP stdio runtime (deploy-time serving of registered tools)
    "run_server",
    "serve_stdio",
    "build_tool_input_schema",
    "coerce_tool_arguments",
    "tool_result_to_text",
]

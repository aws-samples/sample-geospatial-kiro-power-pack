"""Validation sweep for ``geo-query``'s ``spatial_sql`` (task 14.12).

Task 14.4 covers the main validation cases (blank query, unknown engine,
malformed ``sources`` *entries*). This module fills the remaining validation
branches the Pillar B I/O + validation sweep calls out, so every rejection
surfaces on the shared ``Error_Taxonomy`` and runs nothing:

* a ``sources`` argument that is not a mapping at all (e.g. a list) is a
  ``ValidationError`` - distinct from the per-entry checks already covered;
* a non-string ``query`` (e.g. ``None``) is rejected as a validation error;
* validation happens before any credential/configuration check, so an
  unconfigured credentialed engine combined with a bad query still fails as a
  validation error and never executes.

All cases run against a mock engine (no driver). The repo's
``asyncio_mode = "auto"`` runs the async tests.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Tuple

import pytest

from geo_common.errors import ErrorCategory, ValidationError
from geo_query.engine import QueryEngine
from geo_query.models import ResultSet
from geo_query.server import GeoQueryServer


class _MockEngine(QueryEngine):
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: List[Tuple[str, Optional[Mapping[str, str]]]] = []

    def execute(
        self, *, query: str, sources: Optional[Mapping[str, str]] = None
    ) -> ResultSet:
        self.calls.append((query, sources))
        return ResultSet.from_rows(engine=self.name, columns=["n"], rows=[[1]])


def _server() -> Tuple[GeoQueryServer, _MockEngine]:
    engine = _MockEngine("duckdb")
    return GeoQueryServer(engines={"duckdb": engine}), engine


async def test_non_mapping_sources_is_validation_error() -> None:
    """A ``sources`` that is not a mapping (a list) is rejected (no execution)."""
    server, engine = _server()
    with pytest.raises(ValidationError) as exc_info:
        await server.spatial_sql(
            query="SELECT 1",
            engine="duckdb",
            sources=["s3://amzn-s3-demo-bucket/x.parquet"],  # type: ignore[arg-type]
        )
    assert exc_info.value.category is ErrorCategory.VALIDATION
    assert exc_info.value.detail and exc_info.value.detail.get("parameter") == "sources"
    assert engine.calls == []


async def test_non_string_query_is_validation_error() -> None:
    """A non-string query (``None``) is a validation error, not a crash."""
    server, engine = _server()
    with pytest.raises(ValidationError) as exc_info:
        await server.spatial_sql(query=None, engine="duckdb")  # type: ignore[arg-type]
    assert exc_info.value.category is ErrorCategory.VALIDATION
    assert engine.calls == []


async def test_validation_precedes_credential_guard() -> None:
    """A bad query against an unconfigured credentialed engine fails validation
    first - the credential guard is never reached and nothing executes."""
    server = GeoQueryServer(engines={})  # athena unconfigured
    with pytest.raises(ValidationError) as exc_info:
        await server.spatial_sql(query="   ", engine="athena")
    assert exc_info.value.category is ErrorCategory.VALIDATION

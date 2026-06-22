"""Tests for the real DuckDB Spatial engine (:mod:`geo_query.duckdb_engine`).

Unlike ``test_spatial_sql.py`` (which exercises the server against a *mock*
engine and never needs a driver), these tests run the actual DuckDB engine, so
they are skipped when the optional ``duckdb`` package is not installed
(``pytest.importorskip``). They cover Requirement 8.6 against a live engine:

* a plain ``SELECT`` returns a :class:`ResultSet` attributed to ``duckdb``;
* the ``spatial`` extension is loaded so ``ST_*`` functions resolve;
* a ``sources`` mapping is exposed as a queryable view over a file;
* a malformed query is surfaced as an ``Error_Taxonomy`` ``ValidationError``;
* :func:`default_query_engines` wires DuckDB when it is importable; and
* the engine plugs into ``GeoQueryServer`` end-to-end via ``main()``-style wiring.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")  # skip the whole module without the driver

from geo_common.errors import ValidationError  # noqa: E402

from geo_query.duckdb_engine import DuckDBEngine, default_query_engines  # noqa: E402
from geo_query.models import ResultSet  # noqa: E402
from geo_query.server import GeoQueryServer  # noqa: E402


def test_select_returns_result_set() -> None:
    """Req 8.6: a plain SELECT runs and returns a ResultSet tagged ``duckdb``."""
    engine = DuckDBEngine()
    result = engine.execute(query="SELECT 1 AS one, 'a' AS label")
    assert isinstance(result, ResultSet)
    assert result.engine == "duckdb"
    assert result.columns == ["one", "label"]
    assert result.rows == [[1, "a"]]
    assert result.row_count == 1


def test_spatial_extension_is_loaded() -> None:
    """The ``spatial`` extension is available, so ``ST_*`` functions resolve."""
    engine = DuckDBEngine()
    result = engine.execute(query="SELECT ST_AsText(ST_Point(1, 2)) AS wkt")
    assert result.columns == ["wkt"]
    assert result.rows == [["POINT (1 2)"]]


def test_sources_are_exposed_as_views(tmp_path: Path) -> None:
    """Req 8.6: a ``sources`` entry becomes a queryable view over the file."""
    rows = [{"name": "a", "v": 1}, {"name": "b", "v": 2}]
    data = tmp_path / "points.json"
    data.write_text("\n".join(json.dumps(r) for r in rows))
    engine = DuckDBEngine()
    result = engine.execute(
        query="SELECT name, v FROM pts ORDER BY v",
        sources={"pts": str(data)},
    )
    assert result.columns == ["name", "v"]
    assert result.rows == [["a", 1], ["b", 2]]


def test_empty_result_is_valid() -> None:
    """Req 8.6: a query matching nothing returns an empty ResultSet."""
    engine = DuckDBEngine()
    result = engine.execute(query="SELECT 1 AS x WHERE 1 = 0")
    assert result.row_count == 0
    assert result.rows == []


def test_bad_query_is_validation_error() -> None:
    """A malformed query surfaces as a taxonomy ValidationError (not a crash)."""
    engine = DuckDBEngine()
    with pytest.raises(ValidationError):
        engine.execute(query="SELECT FROM WHERE not valid sql")


def test_missing_source_file_is_not_found_with_clear_message() -> None:
    """A ``sources`` entry pointing at a nonexistent file is NOT_FOUND, carrying
    DuckDB's own 'No files found'-style message rather than a generic upstream
    error (regression for the opaque sources-binding failure)."""
    from geo_common.errors import NotFoundError

    engine = DuckDBEngine()
    with pytest.raises(NotFoundError) as exc_info:
        engine.execute(
            query="SELECT * FROM pts",
            sources={"pts": "/tmp/geo_query_does_not_exist_12345.parquet"},
        )
    # The precise engine message is preserved (not swallowed into "upstream").
    assert "could not open a source" in str(exc_info.value).lower()


def test_default_query_engines_wires_duckdb() -> None:
    """When duckdb is importable, the auto-wired set contains the DuckDB engine."""
    engines = default_query_engines()
    assert "duckdb" in engines
    assert isinstance(engines["duckdb"], DuckDBEngine)


async def test_server_executes_via_default_engine() -> None:
    """End-to-end: the engine wired the way ``main()`` does runs a real query."""
    server = GeoQueryServer(engines=default_query_engines())
    result = await server.spatial_sql(query="SELECT 42 AS answer")
    assert result.engine == "duckdb"
    assert result.rows == [[42]]

"""Data models for the ``geo-warehouse`` server (first expansion, Pillar B-class).

``geo-warehouse`` runs warehouse-scale spatial SQL against a proprietary
analytics engine (BigQuery, Snowflake, Redshift, or Databricks) and
returns a tabular result set. This module defines the single result type that
``warehouse_spatial_sql`` returns:

* :class:`ResultSet` - a column-oriented description plus row-major values, so
  the result of any engine's spatial SQL query is shaped uniformly regardless
  of which warehouse produced it (Requirement 8.6).

Python 3.10+ (matching ``pyproject.toml``): uses ``from __future__ import
annotations`` together with ``typing`` generics so the pydantic model resolves
its annotations cleanly.
"""

from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, Field, model_validator

__all__ = ["ResultSet"]


class ResultSet(BaseModel):
    """The tabular result of a spatial SQL query (Requirement 8.6).

    Column-oriented header (``columns``) plus row-major ``rows`` (each row has
    one value per column, in column order). ``row_count`` is the number of rows
    and ``engine`` records which warehouse engine produced the result so callers
    can attribute provenance. An empty query result is a valid ``ResultSet``
    with no rows.
    """

    engine: str = Field(min_length=1, description="The warehouse engine that executed the query.")
    columns: List[str] = Field(default_factory=list, description="Column names in column order.")
    rows: List[List[Any]] = Field(
        default_factory=list,
        description="Row-major values; each row has one entry per column.",
    )
    row_count: int = Field(default=0, ge=0, description="Number of rows returned.")

    @model_validator(mode="after")
    def _check_shape(self) -> "ResultSet":
        """Keep ``row_count`` consistent with ``rows`` and rows rectangular.

        Every row must have exactly one value per declared column, and
        ``row_count`` must equal the number of rows, so a ``ResultSet`` can
        never describe a malformed/ragged table.
        """
        if self.row_count != len(self.rows):
            raise ValueError(
                "row_count (%d) does not match number of rows (%d)"
                % (self.row_count, len(self.rows))
            )
        width = len(self.columns)
        for index, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(
                    "row %d has %d values but %d columns are declared"
                    % (index, len(row), width)
                )
        return self

    @classmethod
    def from_rows(
        cls, *, engine: str, columns: List[str], rows: List[List[Any]]
    ) -> "ResultSet":
        """Build a :class:`ResultSet`, deriving ``row_count`` from ``rows``."""
        return cls(engine=engine, columns=list(columns), rows=[list(r) for r in rows], row_count=len(rows))

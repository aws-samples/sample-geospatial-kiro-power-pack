---
name: spatial-sql
description: "Run spatial SQL across engines: pick geo-query vs geo-warehouse, and use the right ST_* function names per engine."
activation:
  - "code referencing spatial_sql"
  - "code referencing warehouse_spatial_sql"
  - "code referencing ST_ functions"
  - "**/*.sql"
  - "geo-query context"
  - "geo-warehouse context"
---

# Spatial SQL

Encoded best practice for the Geospatial Power Pack: the pack runs spatial SQL
through **two** surfaces with **five** engines, and the same query is **not**
portable across them — the geometry constructors and serializers have
engine-specific names. Pick the right surface for the workload, then use the
right dialect.

## Activation context

This skill applies to any spatial-SQL work: the open `geo-query` `spatial_sql`
tool, the credentialed `geo-warehouse` `warehouse_spatial_sql` tool, `.sql`
files, or any code building `ST_*` expressions.

## Pick the surface first

- **`geo-query` — open, ad-hoc, bring-the-query-to-the-data.** In-process
  **DuckDB Spatial** (opt-in `[duckdb]` extra) for local/single-user analytical
  work, and **Amazon Athena** (the `[athena]` extra) for ad-hoc SQL directly
  over data in S3. No managed warehouse required.
- **`geo-warehouse` — credentialed, warehouse-scale.** Runs against a managed
  warehouse the data already lives in: **BigQuery, Snowflake, Redshift,
  Databricks**. Use this when the data is in the warehouse or exceeds what
  DuckDB/Athena handle comfortably and the engine is licensed/configured.

Rule of thumb (see the `tool-selection` skill for the full size/access-pattern
tree): reach for `geo-query` first; use `geo-warehouse` when the data is already
in a warehouse or the scale demands it.

## The `engine` is an explicit parameter

Both tools take the target **`engine`** as an argument and a non-secret
**`connection`** selector (the database/project/schema) — never inline
credentials in the query or the connection arg. Credentials live only in
`mcp.json` (`BIGQUERY_CREDENTIALS`, `SNOWFLAKE_CONNECTION`, `REDSHIFT_CONNECTION`,
`DATABRICKS_CONNECTION`, or AWS keys + `ATHENA_S3_STAGING_DIR`). An unconfigured
engine returns a clean `authentication` error naming the missing key.

## `ST_*` names differ by engine (the main footgun)

The same "make a point, serialize to WKT" operation is spelled differently on
each engine. These are verified against the live engines:

| Engine | Point constructor | WKT serializer | Example |
|--------|-------------------|----------------|---------|
| DuckDB (`geo-query`) | `ST_Point(lon, lat)` | `ST_AsText(...)` | `SELECT ST_AsText(ST_Point(-122.4, 37.8))` |
| Athena (`geo-query`) | `ST_POINT(lon, lat)` | `ST_ASTEXT(...)` | (Trino/Presto geospatial) |
| BigQuery | `ST_GEOGPOINT(lon, lat)` | `ST_ASTEXT(...)` | `SELECT ST_ASTEXT(ST_GEOGPOINT(-122.4, 37.8))` |
| Redshift | `ST_Point(lon, lat)` | `ST_AsText(...)` | `SELECT ST_AsText(ST_Point(-122.4, 37.8))` |
| Snowflake | `ST_MAKEPOINT(lon, lat)` | `ST_ASWKT(...)` | `SELECT ST_ASWKT(ST_MAKEPOINT(-122.4, 37.8))` |
| Databricks | `st_point(lon, lat)` | `st_astext(...)` | `SELECT st_astext(st_point(-122.4, 37.8))` |

- **BigQuery** uses the `GEOGRAPHY` type and the distinctive `ST_GEOGPOINT`
  constructor; most others use a PostGIS-style `ST_Point`/`ST_MakePoint`.
- All produce the same WKT (`POINT(-122.4 37.8)`), but you must match the
  engine's spelling — a wrong name surfaces as the engine's own error (see
  below), not a silent empty result.
- Longitude comes first (`lon, lat`) on every engine here.

## Identifier case folding

**Snowflake folds unquoted identifiers to UPPERCASE**, so `SELECT 1 AS n`
returns a column named `N`. BigQuery, Redshift, and Databricks preserve the
lowercase `n`. Don't assume the output column name matches your literal casing;
quote identifiers if exact case matters.

## How errors surface

Tool-side validation (blank query, unsupported engine, missing connection)
raises a `validation` error **before** any I/O. A query that reaches the engine
and is rejected there is mapped onto the taxonomy by the engine's own message:

- A **syntax error** → `validation` (e.g. Redshift SQLSTATE `42601`, BigQuery
  "SELECT list must not be empty", Snowflake `42000`).
- An **unresolved column / missing object** → `not-found` (e.g. Databricks
  `UNRESOLVED_COLUMN` / SQLSTATE `42703`, "table does not exist"). Note a lenient
  parser may classify a malformed query as a missing column rather than a syntax
  error — same "your query is bad" meaning, different taxonomy bucket.
- Bad/absent credentials → `authentication` (naming the `mcp.json` key); a 5xx
  or unmapped driver error → `upstream`.

The engine's native message and SQLSTATE are preserved inside the taxonomy
wrapper, so the exact parser position survives for debugging.

## Best practices

- **Push predicates and column projection down** into the query — both surfaces
  read columnar/byte-range data, so `SELECT` only the columns and rows you need.
- **Parameterize/validate inputs** that build SQL; never interpolate untrusted
  text into a query string.
- **Keep one CRS** in mind — most warehouse `GEOGRAPHY`/`GEOMETRY` types assume
  WGS84 lon/lat; reproject upstream (`crs-handling`) if your data isn't.
- **Probe a new connection** with a trivial `SELECT 1` before a real spatial
  query, so a connectivity/credential problem is obvious and separate from a
  dialect problem.

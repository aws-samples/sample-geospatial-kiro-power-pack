---
name: tool-selection
description: Decision tree for choosing the right server/engine for a task and data size.
activation:
  - "natural-language task context"
---

# Tool Selection

Encoded best practice for the Geospatial Power Pack: pick the **right server and
engine** for the task and the data size. The Power Pack is modular — the goal is
to route each request to the cheapest tool that can correctly handle it, and to
delegate to distributed compute only when the data genuinely demands it.

## Activation context

This skill applies to natural-language task context — whenever a request
describes a geospatial goal ("find imagery over this area", "compute zonal
stats", "reproject this layer", "detect change between two dates") and you must
decide which server/engine to use.

## Step 1 — Route by pillar (what kind of task is this?)

- **Pillar A — Data access (discover/read):** `geo-stac` (catalog search),
  `geo-vector` (OSM/Overture features),
  `geo-geocode-route`, `geo-terrain`, `geo-weather-climate`, `geo-biodiversity`.
- **Pillar B — Processing/compute (transform/analyze):** `geo-ops` (CRS +
  geometry, local pure functions), `geo-formats` (format conversion),
  `geo-query` (spatial SQL), `geo-raster` (windowed COG reads/band math/zonal stats),
  `geo-pointcloud` (COPC), `geo-index` (H3/S2).
- **Pillar C — GeoAI:** `geo-foundation-models` (embeddings/segmentation/change),
  `geo-embedding-search` (embed + index + similarity search).
- **Credentialed / proprietary:** `geo-warehouse`,
  `geo-commercial-imagery` — use only when the open/free-tier path can't satisfy
  the requirement and the user has the license configured.

A typical request flows **discover (A) → process (B) → analyze (B/C)** through
the Orchestration Router.

## Step 2 — Choose the processing engine by dataset size and access pattern

This is the engine-selection decision tree (Requirement 12.4). It evaluates the
**input dataset size in bytes** and the **access pattern**:

```
Dataset size?
├─ < 1 GB ............... GeoPandas / Shapely in-process (geo-ops)
├─ 1 GB – 100 GB ........ Access pattern?
│   ├─ single-user analytical ... DuckDB Spatial (geo-query)
│   ├─ multi-user / transactional ... PostGIS (external; bring-your-own)
│   └─ ad-hoc over S3 ............... Athena (geo-query)
└─ > 100 GB / distributed .......... Apache Sedona on Amazon EMR (aws-geo-compute)
```

The tree above keys on **raster/dataset size in bytes**. That is not the only
scaling axis — see the next step for the vector one.

## Step 2b — Route by vector complexity and request payload

Dataset *bytes* is not the only thing that makes a call slow or fail. A tool
that takes **inline GeoJSON** (zonal statistics, `transform_crs`, `spatial_join`,
`overlay`, band-math zones) is also bounded by **geometry vertex count** and the
**size of the request payload** — thousands of coordinates round-tripping
through the model. A 3-KB raster window with a 3,400-vertex zone fails not
because the data is big but because the *inline geometry payload* is.

```
Geometry vertex count / payload size?
├─ small (a few hundred vertices) ...... pass inline as-is
├─ large (thousands of vertices) ....... reduce first:
│     geo-ops.simplify (shape-preserving, keeps concavity)
│       → convex_hull (lossy superset; only if a coarse zone is OK)
│       → bounding box (coarsest)
└─ available as a file/URL ............. reference by path/href instead of
                                         inlining (as geo-formats.to_geoparquet
                                         already accepts for `src`)
```

Reduce the geometry **before** the call, and reproject the *reduced* geometry,
not the full perimeter (`geometry-complexity`, `crs-handling` skills).

## Step 3 — In-process or delegate?

After choosing an in-process path, apply the **delegation threshold**
(Requirement 12.5):

> If the task needs **more than the in-process memory limit (default 4 GB)**
> **or** operates on an **input dataset larger than 5 GB**, delegate it to the
> `aws-geo-compute` peer Power.

- Below both limits → run in-process.
- Above either limit → delegate. Treat non-acceptance within 60s or a delegation
  failure as an `Error_Taxonomy` error (Requirement 12.8).

## Step 4 — Prefer warehouse/proprietary only when warranted

- Reach for `geo-warehouse` (BigQuery/Snowflake/Redshift/Databricks) when
  the data already lives in that warehouse or exceeds what DuckDB/Athena handle
  comfortably and the user has the engine licensed.
- Reach for `geo-commercial-imagery` only when open sources can't
  meet the requirement and credentials are configured.

## Heuristics

- **Smallest correct tool wins.** Don't spin up distributed compute for a
  100 MB join.
- **Read windows, not whole assets.** Prefer windowed/byte-range reads
  (`geo-raster`, `geo-query` over S3) before downloading.
- **Local pure-function work** (CRS transforms, geometry ops, H3/S2 indexing)
  stays in `geo-ops` / `geo-index` — no network, no credentials.
- **Reduce vector complexity before inlining.** For a high-vertex geometry,
  `geo-ops.simplify` (shape-preserving) → `convex_hull` → bbox, or reference it
  by path/href. Payload size is a scaling axis independent of dataset bytes
  (`geometry-complexity` skill).
- **Honor graceful degradation.** Multi-source discovery continues on per-source
  failure and labels results `partial=True` with provenance — don't fail the
  whole request because one source was down.

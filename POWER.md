# Geospatial Power Pack

A modular Kiro Power that gives developers unified, AI-assisted access to the
fragmented geospatial landscape across open, free-tier, and proprietary sources.

The Power is built on a **dual-fragmentation** framing: in geospatial work,
*data access* (discover/read) and *processing/compute* (transform/analyze) are
equally fragmented and receive **equal weight**, with a third pillar for
**GeoAI** (foundation-model embeddings, segmentation, semantic and vector
search).

## Keywords

`geospatial`, `gis`, `geo`, `spatial`, `mapping`, `stac`, `cog`,
`cloud-optimized`, `geoparquet`, `raster`, `vector`, `imagery`, `satellite`,
`remote-sensing`, `geocoding`, `routing`, `crs`, `reprojection`, `zonal-statistics`,
`point-cloud`, `copc`, `h3`, `s2`, `spatial-index`, `geoai`, `embeddings`,
`change-detection`, `embedding-search`, `similarity-search`,
`bigquery`, `snowflake`, `duckdb`, `sedona`, `aws`, `s3`, `batch`, `amazon-sagemaker-ai`,
`stac-search`, `overture`, `openstreetmap`, `terrain`, `elevation`, `weather`,
`climate`, `biodiversity`, `foundation-models`, `clay`, `prithvi`, `satclip`

## The five cooperating parts

1. **Power Hub** (`kiro-geospatial`) — the central coordinator providing an
   Onboarding Dashboard, a searchable Resource Catalog, a Credential Manager
   (single `mcp.json` surface), and a "discover → process → analyze"
   Orchestration Router that fans a single natural-language request across
   multiple sources/tools in parallel with graceful degradation.
2. **Modular MCP servers** — standalone Python packages, one per domain,
   installed à la carte via `uvx`, so users install only what they need.
3. **`geo-common` shared base** — one async HTTP client (built on `httpx`) with
   retry + exponential backoff, a common `Error_Taxonomy`, and rate-limit
   handling that every server inherits.
4. **Skills and steering** — context/file-pattern-activated encoded best
   practices (Skills) plus executable multi-step workflow guides (Steering
   Workflows).
5. **`aws-geo-compute` peer Power** — orchestrates heavy/distributed compute on
   AWS Batch, Amazon ECS with AWS Fargate, Amazon EMR with Apache Sedona, and
   Amazon SageMaker AI via a
   deterministic task-type → execution-target mapping.

## Pillars

- **Pillar A — Data Connectors:** discover and read data (catalogs, imagery,
  vector, geocoding/routing, terrain, weather/climate, biodiversity).
- **Pillar B — Processing & Compute:** transform and analyze data (CRS
  transforms, geometry operations, format conversion, spatial SQL,
  raster/zonal statistics, point clouds, spatial indexing).
- **Pillar C — GeoAI:** foundation-model embeddings, segmentation, vector
  search, change detection, semantic search.

By count, Pillar B (processing/compute) is never smaller than Pillar A
(data access).

## Install matrix

Each server is a standalone package installed independently via `uvx`.
Installing a server installs it plus `geo-common` and nothing else.

| Server | Pillar | Status | Openness tier | Install |
|---|---|---|---|---|
| `kiro-geospatial` | Hub | MVP | Open | `uvx kiro-geospatial` |
| `geo-common` | Shared base | MVP | Open | (installed as a dependency) |
| `geo-stac` | A | MVP | Open | `uvx geo-stac` |
| `geo-vector` | A | MVP | Open | `uvx geo-vector` |
| `geo-geocode-route` | A | Expansion | Open / Free-Tier | `uvx geo-geocode-route` |
| `geo-terrain` | A | Expansion | Open | `uvx geo-terrain` |
| `geo-weather-climate` | A | Expansion | Open / Free-Tier | `uvx geo-weather-climate` |
| `geo-biodiversity` | A | Expansion | Open | `uvx geo-biodiversity` |
| `geo-ops` | B | MVP | Open | `uvx geo-ops` |
| `geo-formats` | B | Expansion | Open | `uvx geo-formats` |
| `geo-query` | B | Expansion | Open | `uvx geo-query` |
| `geo-raster` | B | Expansion | Open | `uvx geo-raster` |
| `geo-pointcloud` | B | Expansion | Open | `uvx geo-pointcloud` |
| `geo-index` | B | Expansion | Open | `uvx geo-index` |
| `geo-foundation-models` | C | MVP | Open | `uvx geo-foundation-models` |
| `geo-embedding-search` | C | Expansion | Open | `uvx geo-embedding-search` |
| `geo-warehouse` | Credentialed (Pillar B-class) | First expansion | Proprietary/Licensed | `uvx geo-warehouse` |
| `geo-commercial-imagery` | Credentialed | Expansion | Proprietary/Licensed | `uvx geo-commercial-imagery` |
| `aws-geo-compute` | Peer Power | Expansion | Open (AWS account) | `uvx aws-geo-compute` |

### MVP set

`geo-common`, `geo-stac`, `geo-vector`, `geo-ops`, `geo-foundation-models`, and
the Power Hub (`kiro-geospatial`), plus an initial set of skills and steering
workflows. Install the MVP set together to get a working
discover → process → analyze pipeline.

### First expansions

`geo-warehouse`.

## Recommended companion MCPs

These are **third-party** MCP servers that pair well with this Power. They are
**not part of the Geospatial Power Pack and are not redistributed with it** —
you install them yourself, and each is governed by its own license. We list them
purely as recommendations, with attribution.

- **`gdal-mcp`** — a GDAL/Rasterio operations MCP server
  ([JordanGunn/gdal-mcp](https://github.com/JordanGunn/gdal-mcp), MIT). It
  complements this pack's cloud-native connectors with **local-file** GDAL
  muscle: file-level raster/vector reprojection and format conversion, plus
  common vector ops (`buffer`, `simplify`, `clip`) that `geo-ops` doesn't cover.
  Reach for it for local/desktop file processing; reach for this pack for
  cloud-native reads (STAC, COG byte-range, open APIs).
  - Install (separately, from PyPI): `uvx --from gdal-mcp gdal --transport stdio`
  - **Caveat 1:** it is scoped by `GDAL_MCP_WORKSPACES` (a directory allowlist).
    If unset, *all paths are allowed* — set it to constrain file access.
  - **Caveat 2:** its reflection middleware requires a structured justification
    before methodology-sensitive tools (reprojection, resampling, query extent)
    will run — a deliberate, opinionated UX distinct from this pack's tools.

- **`gis-mcp`** — a single local GIS-operations MCP server
  ([mahdin75/gis-mcp](https://github.com/mahdin75/gis-mcp), MIT) built on the
  classic desktop Python GIS stack (Shapely/PyProj/GeoPandas/Rasterio/PySAL).
  Its geometry, CRS, raster, and connector functions overlap with this pack's
  native servers — prefer this pack's cloud-native versions there. Reach for
  `gis-mcp` for what this pack does **not** cover: **PySAL spatial statistics /
  ESDA** (Moran's I, Geary's C, Getis-Ord, LISA, spatial regression) and **map
  visualization** (static Matplotlib + interactive Folium web maps).
  - Install (separately, from PyPI): `uvx gis-mcp`
  - **Note:** earlier versions of this pack wrapped `gis-mcp` for two generic
    geometry ops; `geo-ops` now implements `buffer` and `convex_hull` natively,
    so `gis-mcp` is a recommended companion rather than a dependency.

### Hosted, commercial MCP servers (third-party)

The companions above are open-source, locally-run tools. The servers below are
**third-party, commercial, hosted** MCP services — **not part of this Power, not
redistributed, and not affiliated with or endorsed by AWS**. Each requires its
own account/license. They are remote MCP endpoints (unlike the pack's local
`uvx` servers), listed for awareness because they offer managed, at-scale
capabilities the open servers here do not. Esri and Wherobots are members of the
AWS Partner Network.

**Esri** — two hosted MCP servers (both public beta):

- **ArcGIS Enterprise MCP — public beta** — Esri's MCP server over ArcGIS
  Enterprise content (item/layer search + describe, attribute/spatial queries,
  map-image export, geocoding). **Beta: not GA, subject to change, not for
  production.** Commercial; requires an ArcGIS Enterprise deployment + API key.
  Remote HTTP, bridged into Kiro via the `mcp-remote` npx proxy with a
  `Authorization: Bearer` header. See the
  [ArcGIS Enterprise MCP beta docs](https://mcpbeta.webgistesting.net/mcpbetadoc/).
- **ArcGIS Location Services MCP — public beta** — Esri's hosted MCP server over
  the ArcGIS Location Platform:
  geocoding, routing, elevation, and static maps. **Beta: not GA, subject to change, not for production.**
  Commercial and metered — calls bill against ArcGIS Location Platform usage.
  Needs a Location Platform account + an access token with beta-access and
  Geocoding/Routing/Elevation/Static-maps privileges. Remote HTTP, bridged into
  Kiro via the `mcp-remote` npx proxy with a `Authorization: Bearer` header. See
  the [ArcGIS Location Services MCP docs](https://developers.arcgis.com/ai-tools/mcp-arcgis-location-services/get-started/).

**Wherobots** — one hosted MCP server:

- **Wherobots Cloud MCP** — managed Apache Sedona spatial lakehouse:
  catalog exploration, planetary-scale Spatial SQL, and job submission (the
  open, self-managed equivalent here is the EMR + Apache Sedona path in
  `aws-geo-compute`). Commercial; the MCP server needs a Wherobots
  Professional/Innovation/Enterprise organization. Configured with Kiro's
  `mcpServers` `url` + `x-api-key` header schema. See the
  [Wherobots Kiro setup docs](https://docs.wherobots.com/develop/agentic-tools/kiro).

## Cross-cutting principles

- **Single credential surface** — configure keys once in `mcp.json`, each
  flagged Required, Optional, or License-Needed.
- **Modular installation** — install only the servers you need via `uvx`.
- **Cloud-optimized formats** — COG (raster), GeoParquet (vector), Zarr
  (multidimensional), COPC (point clouds).
- **AWS-native "bring compute to the data"** — read only the byte ranges
  required directly from S3; delegate heavy jobs to `aws-geo-compute`.
- **Reuse vs build** — every connector is implemented natively on the shared
  `geo-common` base; mature external MCP servers are recommended as companions
  rather than wrapped or redistributed.
- **Property-based testing** — Hypothesis-backed properties for the
  logic-bearing components.

## Repository layout

```
geospatial-kiro-power-pack/
├── POWER.md                  # this file: Power overview, keywords, install matrix
├── bundle-manifest.json      # declares servers, skills, steering, wrapped externals
├── packages/                 # one standalone, uvx-installable package per server
├── skills/                   # encoded best-practice skill files
└── steering/                 # file-pattern-triggered workflow guides
```

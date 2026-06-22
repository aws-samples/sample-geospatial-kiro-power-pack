# Geospatial Power Pack

A modular [Kiro](https://kiro.dev) Power that gives developers unified,
AI-assisted access to the fragmented geospatial landscape across **open**,
**free-tier**, and **proprietary** sources.

It is built on a **dual-fragmentation** framing: in geospatial work, *data
access* (discover/read) and *processing/compute* (transform/analyze) are
equally fragmented and receive **equal weight**, plus a third pillar for
**GeoAI** (foundation-model embeddings, segmentation, semantic and vector
search). The architecture mirrors the
[Kiro for Life Sciences](https://github.com/aws-samples/sample-kiro-power-life-sciences) pack:
one central Power acts as the hub, and users install only the standalone MCP
servers they need, each runnable via `uvx`.

> See [`POWER.md`](POWER.md) for the Power overview, keywords, and the full
> install matrix, and [`bundle-manifest.json`](bundle-manifest.json) for the
> machine-readable declaration of every server, skill, steering workflow, and
> wrapped external.

---

## The five cooperating parts

1. **Power Hub** (`kiro-geospatial`) — the central coordinator: an Onboarding
   Dashboard, a searchable Resource Catalog, a Credential Manager (single
   `mcp.json` surface), and a "discover → process → analyze" Orchestration
   Router that fans a single natural-language request across multiple
   sources/tools in parallel with graceful degradation.
2. **Modular MCP servers** — standalone Python packages, one per domain,
   installed à la carte via `uvx`.
3. **`geo-common` shared base** — one async HTTP client (built on `httpx`) with
   retry + exponential backoff, a common `Error_Taxonomy`, rate-limit handling,
   and the `BaseGeoServer` contract every server inherits.
4. **Skills and steering** — context/file-pattern-activated best practices
   (`skills/`) plus executable multi-step workflow guides (`steering/`).
5. **`aws-geo-compute` peer Power** — orchestrates heavy/distributed compute on
   AWS Batch, Amazon ECS with AWS Fargate, Amazon EMR with Apache Sedona, and
   Amazon SageMaker AI via a
   deterministic task-type → execution-target mapping.

## Pillars

- **Pillar A — Data Connectors:** catalogs, imagery, vector, geocoding/routing,
  terrain, weather/climate, biodiversity.
- **Pillar B — Processing & Compute:** CRS transforms, geometry operations,
  format conversion, spatial SQL, raster/zonal statistics, point clouds,
  spatial indexing.
- **Pillar C — GeoAI:** foundation-model embeddings, segmentation, vector
  search, change detection.

By design, the Pillar B (processing/compute) server count is never smaller than
Pillar A (data access).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                            Kiro IDE                            │
├──────────────────────────────────────────────────────────────┤
│                kiro-geospatial Power (Hub)                     │
│  ┌────────────┐ ┌────────────┐ ┌──────────────────────────┐  │
│  │ Dashboard  │ │  Catalog   │ │ Credential Manager        │  │
│  └────────────┘ └────────────┘ └──────────────────────────┘  │
│  ┌────────────┐ ┌────────────┐ ┌──────────────────────────┐  │
│  │  Skills    │ │  Steering  │ │ Orchestration Router      │  │
│  │ (7 guides) │ │(5 workflows│ │ discover → process →      │  │
│  │            │ │            │ │ analyze (graceful degrade)│  │
│  └────────────┘ └────────────┘ └──────────────────────────┘  │
├──────────────────────────────────────────────────────────────┤
│         Modular MCP servers (install as needed via uvx)        │
│   A: stac · vector · geocode-route · terrain ·                 │
│      weather-climate · biodiversity · ogc                      │
│   B: ops · formats · query · raster · pointcloud · index · 3d  │
│   C: foundation-models · embedding-search                      │
│   Credentialed: warehouse · commercial-imagery                 │
├──────────────────────────────────────────────────────────────┤
│   geo-common (shared base: HttpClient, Error_Taxonomy,         │
│   BaseGeoServer)                                               │
├──────────────────────────────────────────────────────────────┤
│            Peer Power: aws-geo-compute                         │
└──────────────────────────────────────────────────────────────┘
```

A request flows strictly in pillar order: every **discover** step precedes
every **process** step, which precedes every **analyze** step. Discovery
sources are queried concurrently; on a per-source failure the router records
provenance and continues, labelling any degraded result *partial*.

---

## Repository layout

```
geospatial-kiro-power-pack/
├── README.md                 # this file
├── POWER.md                  # Power overview, keywords, install matrix
├── bundle-manifest.json      # servers, skills, steering, wrapped externals
├── pyproject.toml            # repo-wide pytest config + shared test deps
├── conftest.py               # Hypothesis profile + shared mock fixtures
├── packages/                 # one standalone, uvx-installable package per server
│   ├── kiro-geospatial/      # the Power Hub
│   ├── geo-common/           # shared base (dependency of every server)
│   ├── geo-stac/ geo-vector/ … (Pillar A)
│   ├── geo-ops/ geo-formats/ … (Pillar B)
│   ├── geo-foundation-models/ geo-embedding-search/ (Pillar C)
│   ├── geo-warehouse/ geo-commercial-imagery/ … (credentialed)
│   └── aws-geo-compute/      # peer compute Power
├── skills/                   # encoded best-practice skill files
├── steering/                 # file-pattern-triggered workflow guides
└── docs/                     # supporting documentation
```

---

## Capabilities map

**21 packages** — the Power Hub, the `geo-common` shared base, and **19 MCP
servers** exposing **46 tools** across Pillars A (data access), B
(processing/compute), and C (GeoAI), plus credentialed and peer-Power servers.
Per-server credential keys are in the Credentials reference below.

| # | Server | Pillar | Tier | Tools | Key capabilities |
|---|--------|--------|------|-------|------------------|
| — | `kiro-geospatial` | Hub | Open | — | dashboard, catalog, credential manager, orchestration router |
| — | `geo-common` | Shared base | Open | — | HttpClient, retry/backoff, `Error_Taxonomy`, `BaseGeoServer` |
| 1 | `geo-stac` | A | Open | 2 | STAC search — single catalog (`stac_search`) and federated multi-catalog (`stac_search_multi`, round-robin merge across Earth Search, Planetary Computer, CMR-STAC, Copernicus, USGS) |
| 2 | `geo-vector` | A | Open | 1 | vector features from OpenStreetMap (Overpass); optional configured Overture endpoint |
| 3 | `geo-geocode-route` | A | Open / Free-Tier | 4 | geocode + reverse-geocode (Nominatim, Photon), routing (OSRM, Valhalla), isochrones (Valhalla); Amazon Location selectable via `source` (credentialed) |
| 4 | `geo-terrain` | A | Open | 3 | elevation, slope, hillshade from SRTM (default) and USGS 3DEP (public OpenTopoData; endpoint/dataset overridable) |
| 5 | `geo-weather-climate` | A | Open / Free-Tier | 1 | observations from Open-Meteo (default), OpenAQ, NOAA/NWS; NOAA CDO (credentialed). ERA5 reanalysis is covered via the Open-Meteo archive, not a direct CDS client |
| 6 | `geo-biodiversity` | A | Open | 1 | species occurrences from GBIF + iNaturalist; IUCN Red List conservation status (credentialed) |
| 7 | `geo-ogc` | A | Open | 1 | fetch GeoJSON features from any OGC API - Features service (pygeoapi, GeoServer OGC API, ldproxy) by endpoint + collection |
| 8 | `geo-ops` | B | Open | 6 | CRS transforms, geometry validation/ops, spatial join, overlay, buffer, convex hull (PyProj, Shapely/GEOS, GeoPandas) |
| 9 | `geo-formats` | B | Open | 3 | COG / GeoParquet conversion + validation |
| 10 | `geo-query` | B | Open | 1 | ad-hoc, in-process spatial SQL via DuckDB Spatial (open default, opt-in `[duckdb]` extra) / Amazon Athena over S3 (bring-your-own AWS creds) — for warehouse-scale SQL see `geo-warehouse` |
| 11 | `geo-raster` | B | Open | 3 | windowed COG reads + band math (NDVI/NDWI/NBR); zonal statistics over vector zones |
| 12 | `geo-pointcloud` | B | Open | 2 | Cloud-Optimized Point Cloud (COPC) read/write |
| 13 | `geo-index` | B | Open | 1 | H3 (0-15) and S2 (0-30) spatial indexing |
| 14 | `geo-3d` | B | Open | 4 | inspect 3D Tiles + glTF/GLB; tile point clouds → 3D Tiles `.pnts` (optional octree LOD); mesh DEM grids → glTF/GLB terrain (with normals) |
| 15 | `geo-foundation-models` | C | Open | 4 | embeddings, change detection, segmentation. The on-tile `embed_tile`/`segment` paths use deterministic local stand-in backends for Clay/Prithvi-EO-2.0/SatCLIP/SAMGeo (pluggable for real weights; each result's `backend` field records its provenance); `lookup_embeddings` retrieves **real** published Clay v1.5 (1024-d) Sentinel-2 vectors |
| 16 | `geo-embedding-search` | C | Open | 2 | embedding store + similarity search (OpenSearch, LanceDB) |
| 17 | `geo-warehouse` | Credentialed (B-class) | Proprietary | 1 | warehouse-scale spatial SQL against managed cloud warehouses (BigQuery, Snowflake, Redshift, Databricks) — for open/ad-hoc SQL see `geo-query` |
| 18 | `geo-commercial-imagery` | Credentialed | Proprietary | 2 | commercial imagery search + ordering, routed by provider: Maxar via Sentinel Hub TPDI, Planet via Planet's Data/Orders APIs |
| 19 | `aws-geo-compute` | Peer Power | Open (AWS account) | 4 | engine selection + job submit/status on AWS Batch, AWS Fargate, Amazon EMR, and Amazon SageMaker AI |

All servers above are fully implemented, with passing unit, property-based, and
integration test suites.

**Recommended starter set** (a working discover → process → analyze pipeline):
`geo-common`, `geo-stac`, `geo-vector`, `geo-ops`, `geo-foundation-models`, and
the Power Hub (`kiro-geospatial`). Common next additions: `geo-warehouse`.

---

## Recommended companion MCPs

These **third-party** MCP servers pair well with the pack. They are **not part
of the Geospatial Power Pack and are not redistributed with it** — install them
yourself; each is governed by its own license. Listed as recommendations, with
attribution.

| Companion | Source | License | Use it for |
|-----------|--------|---------|------------|
| `gdal-mcp` | [JordanGunn/gdal-mcp](https://github.com/JordanGunn/gdal-mcp) | MIT | Local-file GDAL/Rasterio ops: file-level raster/vector reproject + convert, and vector `buffer`/`simplify`/`clip` |
| `gis-mcp` | [mahdin75/gis-mcp](https://github.com/mahdin75/gis-mcp) | MIT | Spatial statistics / ESDA you don't get here (PySAL: Moran's I, Geary's C, Getis-Ord, LISA, spatial regression) and map rendering (static Matplotlib + interactive Folium web maps) |

`gis-mcp` is a single local server built on the classic desktop Python GIS
stack (Shapely/PyProj/GeoPandas/Rasterio/PySAL). Its **geometry, CRS, raster,
and connector** functions overlap with this pack's native servers (`geo-ops`,
`geo-raster`, `geo-stac`, …) — prefer this pack's cloud-native versions for
those. Reach for `gis-mcp` for the parts this pack does **not** cover:
**PySAL spatial statistics** and **visualization**. Install it separately from
PyPI:

```bash
uvx gis-mcp
```

(Earlier versions of this pack wrapped `gis-mcp` for two generic geometry ops;
`geo-ops` now implements `buffer` and `convex_hull` natively, so `gis-mcp` is a
recommended companion rather than a dependency.)

`gdal-mcp` is the **local-file** complement to this pack's cloud-native
connectors. Install it separately from PyPI:

```bash
uvx --from gdal-mcp gdal --transport stdio
```

Two things to know before using it:
- It is scoped by `GDAL_MCP_WORKSPACES` (a colon-separated directory allowlist).
  **If unset, all paths are allowed** — set it to constrain file access.
- Its reflection middleware requires a structured justification before
  methodology-sensitive tools (reprojection, resampling, query extent) run —
  an opinionated UX distinct from this pack's validate-then-run tools.

To wire it into Kiro, it needs [`uv`](https://docs.astral.sh/uv/) installed
(`brew install uv`, or `curl -LsSf https://astral.sh/uv/install.sh | sh`) — this
is the first server launched via `uvx` rather than a `.venv` console script. Use
the **absolute** `uvx` path in `mcp.json`, because Kiro launched from Finder/Dock
does not inherit your shell `PATH` and a bare `"uvx"` fails with
`spawn uvx ENOENT` (run `which uvx` to find it):

```json
"gdal-mcp": {
  "command": "/opt/homebrew/bin/uvx",
  "args": ["--from", "gdal-mcp", "gdal", "--transport", "stdio"],
  "env": { "GDAL_MCP_WORKSPACES": "/path/to/your/geo/data" }
}
```

Rule of thumb: use `gdal-mcp` for local/desktop file processing; use this pack
for cloud-native reads (STAC, COG byte-range, open APIs).

---

## Credentials reference

All credentials are configured once in `mcp.json` and flagged **Required**,
**Optional**, or **License-Needed**. The MVP servers are open and need no
credentials. Notable keys:

| Server | Key(s) | Classification |
|---|---|---|
| `geo-stac` | `PC_SDK_SUBSCRIPTION_KEY`, `EARTHDATA_TOKEN` | Optional |
| `geo-foundation-models` | `HF_TOKEN` | Optional |
| `geo-formats` / `geo-query` / `geo-raster` / `geo-pointcloud` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (+ `ATHENA_S3_STAGING_DIR`) | Optional |
| `geo-embedding-search` | `OPENSEARCH_URL` / `OPENSEARCH_USERNAME` / `OPENSEARCH_PASSWORD` | Optional |
| `geo-warehouse` | `BIGQUERY_CREDENTIALS`, `SNOWFLAKE_CONNECTION`, `REDSHIFT_CONNECTION`, `DATABRICKS_CONNECTION` | License-Needed |
| `geo-commercial-imagery` | `SENTINELHUB_CLIENT_ID`, `SENTINELHUB_CLIENT_SECRET` (Maxar via Sentinel Hub), `PL_API_KEY` (Planet) | License-Needed |
| `aws-geo-compute` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | Optional (enforced at job submit/status) |

Secret values never appear in any status report, error, or log.

---

## Installation

**Prerequisites**

- Kiro IDE
- Python 3.10 or higher
- The [`uv`](https://docs.astral.sh/uv/) package manager
  (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

**Create an environment and install the shared base, hub, and the servers you
want**:

```bash
cd geospatial-kiro-power-pack

uv venv .venv
source .venv/bin/activate

# 1) Shared base first (every server depends on it)
uv pip install -e ./packages/geo-common

# 2) The Power Hub
uv pip install -e ./packages/kiro-geospatial

# 3) The MVP servers (a working discover → process → analyze pipeline)
uv pip install -e ./packages/geo-stac
uv pip install -e ./packages/geo-vector
uv pip install -e ./packages/geo-ops
uv pip install -e ./packages/geo-foundation-models

# 4) Any expansion / credentialed servers you need, e.g.
uv pip install -e ./packages/geo-formats
uv pip install -e ./packages/geo-warehouse
```

Each install pulls in `geo-common` and nothing else, so you install only what
you need.

---

## Configure the Power in Kiro

1. Copy the Power directory (which holds `POWER.md`, `bundle-manifest.json`,
   `skills/`, and `steering/`) into Kiro's powers directory:

   ```bash
   mkdir -p ~/.kiro/powers/kiro-geospatial
   cp -r POWER.md bundle-manifest.json skills steering ~/.kiro/powers/kiro-geospatial/
   ```

2. Register the MCP servers in your workspace `.kiro/settings/mcp.json`, with
   each `command` pointing at the console entry point installed in your venv
   (add only the servers you installed):

   ```json
   {
     "mcpServers": {
       "geo-stac":   { "command": "/abs/path/.venv/bin/geo-stac" },
       "geo-vector": { "command": "/abs/path/.venv/bin/geo-vector" },
       "geo-ops":    { "command": "/abs/path/.venv/bin/geo-ops" }
     }
   }
   ```

3. Add credentials for any License-Needed / Required servers via `env`:

   ```json
   {
     "mcpServers": {
       "geo-warehouse": {
         "command": "/abs/path/.venv/bin/geo-warehouse",
         "env": { "BIGQUERY_CREDENTIALS": "/abs/path/service-account.json" }
       },
       "aws-geo-compute": {
         "command": "/abs/path/.venv/bin/aws-geo-compute",
         "env": { "AWS_REGION": "us-east-1" }
       }
     }
   }
   ```

   **Prefer short-lived credentials.** Avoid putting long-term AWS access keys
   in `mcp.json`. For AWS-backed servers (`aws-geo-compute`, and the optional
   S3 paths of `geo-formats`/`geo-query`/`geo-raster`/`geo-pointcloud`), leave
   the key fields unset and let the server pick up credentials from the standard
   AWS provider chain — an IAM role (on EC2/ECS/Lambda) or an `aws configure`
   profile (`AWS_PROFILE`) for local development. Where a server reads a key
   directly from `mcp.json` (e.g. `BIGQUERY_CREDENTIALS`), point it at a file
   restricted with `chmod 600`, never commit that file, and rotate the
   credential regularly. Secret values never appear in any status report,
   error, or log.

4. Activate the **kiro-geospatial** Power from the Kiro Powers panel.

Each domain server's console entry point (e.g. `geo-stac`) runs the startup
credential guard and then serves its registered tools over an MCP **stdio**
connection via the shared `geo-common` runtime, so Kiro launches it directly
from the `command` you configure. The `kiro-geospatial` hub entry point is a
readiness/summary command — the hub is the Power Kiro activates (via `POWER.md`,
`bundle-manifest.json`, `skills/`, `steering/`), not a tool server, so it is not
listed in `mcp.json`.

### Windows

The pack runs on Windows-based Kiro. Differences from the Unix examples above:

- Console scripts live in the venv `Scripts` dir with `.exe`, so `mcp.json`
  `command` is e.g. `C:\\path\\.venv\\Scripts\\geo-stac.exe`.
- Activate with `.venv\\Scripts\\activate`; copy the Power files into
  `%USERPROFILE%\\.kiro\\powers\\kiro-geospatial\\` (Explorer or `Copy-Item`).
- The `Makefile` and `scripts/build_wheels.sh` are Unix shell wrappers — use
  WSL/Git Bash, or run the underlying `python`/`pip` commands directly. The
  Python smoke scripts (`scripts/smoke_*.py`) run natively on Windows.
- Heavy wheels (rasterio/GDAL, shapely, pyproj, geopandas, pyarrow, h3, duckdb)
  have Windows wheels. The optional `geo-pointcloud` PDAL backend needs conda
  (`conda install -c conda-forge pdal python-pdal`); the default COPC backend is
  pure-Python.

---

## Testing

Test configuration lives in the root `pyproject.toml`
(`[tool.pytest.ini_options]`) and `conftest.py` (Hypothesis profile enforcing
≥100 generated cases per property test, plus shared `httpx` mock-transport and
`moto` fixtures).

```bash
# Install the shared dev/test dependency group
uv sync --group test

# Run the whole suite (discovers everything under packages/)
uv run pytest
```

Alternatively, each package declares a `dev` extra, so you can install a single
package with its test tools (life-sciences style):

```bash
uv pip install -e "./packages/geo-formats[dev]"
uv run pytest packages/geo-formats
```

More selections:

```bash
uv run pytest packages/geo-raster          # one package
uv run pytest -m property                  # only Hypothesis property tests
uv run pytest -m "not integration"         # skip integration/moto tests
HYPOTHESIS_PROFILE=ci uv run pytest -m property   # deeper 1000-case search
```

If you prefer not to install the packages, you can run a single package against
the source tree with `PYTHONPATH`:

```bash
PYTHONPATH="packages/geo-common:packages/geo-formats" python -m pytest packages/geo-formats
```

Property-based tests use [Hypothesis](https://hypothesis.readthedocs.io/) and
are tagged `Feature: geospatial-power-pack, Property {n}`. Integration tests
that need AWS use [`moto`](https://github.com/getmoto/moto) and skip
automatically when it is not installed.

---

## Smoke testing

Two layered smoke tests give deploy confidence beyond the unit suites. A
`Makefile` wraps them (`make help` lists all targets).

**Layer 1 — import + construction** (`scripts/smoke_import.py`): imports every
package, asserts each server constructs and registers tools, and checks each
console entry point resolves. No network, no `mcp` SDK, no AWS.

```bash
make smoke-import        # or: python3 scripts/smoke_import.py
```

**Layer 2/3 — MCP stdio handshake + local tool calls** (`scripts/smoke_mcp.py`):
launches a server's console entry point over MCP stdio exactly as Kiro does,
runs the `initialize` + `tools/list` handshake, and calls one no-network local
tool. This requires the `mcp` SDK and the target servers installed in a Python
≥3.10 environment. Since `uv` may not be present, the `Makefile` builds a plain
`venv`:

```bash
make venv            # python3.12 -m venv .venv; pip install mcp + test tools + geo-common
make install-smoke   # editable-install geo-index + geo-embedding-search (lightweight)
make smoke-mcp       # handshake + call_tool against those servers
```

To get the SDK without the `Makefile`:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install mcp pytest pytest-asyncio hypothesis
pip install -e ./packages/geo-common -e ./packages/geo-index
python scripts/smoke_mcp.py geo-index
# expected: [PASS] geo-index  tools/list -> index_cell; call_tool(index_cell) OK -> '89283082803ffff'
```

Add a server to the handshake run with `make smoke-mcp SERVERS="geo-index geo-ops"`
(install it first; servers with heavy native deps — GDAL/rasterio, PDAL — need
those wheels available).

**Heavy / native-dependency servers** (GDAL/rasterio, shapely, pyproj, numpy,
pyarrow) have their own targets, since their wheels are larger:

```bash
make install-heavy   # editable-install geo-ops, geo-formats, geo-raster,
                     # geo-pointcloud, geo-foundation-models, geo-query
make smoke-heavy     # MCP handshake for all of them; geo-ops also runs a reprojection call
```

All of these were verified serving over MCP, including a `geo-ops`
`transform_crs` call returning a reprojected geometry. Note `geo-query` runs the
open **DuckDB Spatial** engine in-process when its optional `[duckdb]` extra is
installed (`pip install -e "./packages/geo-query[duckdb]"`), wired through the
server's entry point; without the extra it still serves and lists `spatial_sql`
but reports the engine as not available; **Amazon Athena** is now a shipped
concrete engine too (the `[athena]` extra, wired when AWS is configured).
`geo-pointcloud`'s default COPC backend is pure-Python
(PDAL is an optional, lazily-imported production backend).

**Credentialed / peer servers** (`geo-commercial-imagery`, `geo-warehouse`,
`aws-geo-compute`) are all httpx-only, so
they install and verify with **no credentials**:

```bash
make install-credentialed
make smoke-credentialed   # handshake + credential-guard probe; needs no accounts
```

`smoke-credentialed` runs each tool with the relevant key unset and asserts the
credential guard returns a taxonomy `AuthenticationError` naming the missing
`mcp.json` key (e.g. `SENTINELHUB_*`, `BIGQUERY_CREDENTIALS`). Each credentialed
capability also ships a **concrete success path** verified at Layer 1 (fake-client
unit tests, no account) and exercisable live at Layer 2 with an account — see
"Phase 10 — credentialed success paths" in `docs/testing-workflows.md`:
`geo-commercial-imagery` (Maxar via Sentinel Hub TPDI + Planet via Planet's own
Data/Orders APIs),
`geo-warehouse` (BigQuery via the free sandbox; `[bigquery]` extra — plus
concrete **Redshift**, **Snowflake**, and **Databricks** engines via their
`[redshift]`/`[snowflake]`/`[databricks]` extras),
`geo-query` (Athena; `[athena]` extra), and `aws-geo-compute` (AWS Batch +
Fargate targets; `[aws]` extra). Only the Amazon EMR with Apache Sedona and
Amazon SageMaker AI compute targets
stay bring-your-own.

**Packaging check** — build a wheel for every package (validates packaging
metadata) and optionally install + serve one from its wheel in a throwaway env:

```bash
make build          # builds all 21 wheels into dist/
make build-verify   # + installs geo-common/geo-index from wheels and MCP-smokes them
```

**Optional PDAL backend** — `geo-pointcloud`'s production COPC path needs the
native PDAL library (`brew install pdal` / `conda install -c conda-forge pdal
python-pdal`), then `pip install -e "./packages/geo-pointcloud[pdal]"`. Until
installed, its PDAL test skips and the default pure-Python backend is used.

See [`docs/testing-workflows.md`](docs/testing-workflows.md) for step-by-step
**in-Kiro manual acceptance** (with example prompts and expected results) and
PDAL enablement.

**Layer 4 — live sources** is reserved for opt-in network tests tagged with the
`live` marker (registered in the root `pyproject.toml`); the default
`-m "not integration"` / `-m "not live"` selections keep CI hermetic.

---

## Cross-cutting principles

- **Single credential surface** — configure keys once in `mcp.json`.
- **Modular installation** — install only the servers you need via `uvx`.
- **Federated search with fair merging** — `stac_search_multi` queries multiple
  STAC catalogs concurrently and **round-robin merges** the results (one item
  per catalog in turn) up to the limit, so every responding catalog is
  represented rather than the first one filling the quota. By default
  (`dedupe="scene"`) it also collapses the *same physical scene* served by
  multiple catalogs under different id schemes (matching on acquisition instant
  + MGRS tile, conservatively); pass `dedupe="id"` to keep each catalog's copy
  (e.g. when the per-catalog asset format matters). Each endpoint's outcome is
  reported as provenance, and a failed catalog marks the result *partial*
  rather than failing the whole search (graceful degradation).
- **Cloud-optimized formats** — COG, GeoParquet, Zarr, COPC.
- **AWS-native "bring compute to the data"** — read only the byte ranges
  required from S3; delegate heavy jobs to `aws-geo-compute`.
- **Reuse vs build** — every connector is implemented natively on the shared
  `geo-common` base for one consistent error/credential/catalog surface; mature
  external MCP servers (`gdal-mcp`, `gis-mcp`) are recommended as companions
  rather than wrapped or redistributed.
- **Property-based testing** — Hypothesis-backed correctness properties for the
  logic-bearing components.

---

## Authors

- [Taylor Teske](https://www.linkedin.com/in/taylor-teske-b37aaa71/)
- [Chris Stoner](https://www.linkedin.com/in/chrisstoner/)

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to add or change servers and
tools, the `BaseGeoServer` contract, the error taxonomy, and the static checks
that keep the manifest and code in sync.

## Security

See [`CONTRIBUTING.md`](CONTRIBUTING.md#security-issue-notifications) for how to
report a potential security issue — please notify AWS/Amazon Security via the
[vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/)
rather than opening a public GitHub issue. Credentials are configured only in
`mcp.json` (or a `0600` file referenced from it), never committed; servers
enforce TLS on outbound calls and never echo secret values into errors or logs.

## License

This project is licensed under the MIT-0 License. See [`LICENSE`](LICENSE) for
details.

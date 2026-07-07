# Testing workflows

Concrete, copy-pasteable workflows for verifying the Geospatial Power Pack in
Kiro, plus the optional PDAL backend and companion server. For the automated
suites and smoke tests see the README ("Testing" and "Smoke testing").

> **Before an acceptance run**, sanity-check the generated MCP tool schemas:
> `make check-schemas` (or `python scripts/check_tool_schemas.py`). It validates
> every server's tool `inputSchema` is well-formed with no dangling `$ref`,
> across all servers in one pass, with no network or credentials.

## How this document is organized

- **Section 1 — Setup**: install the servers, the Power files, and a scratch
  workspace once.
- **Section 2 — Acceptance phases**: chat prompts grouped into **Phases 1–10**,
  ordered by setup friction (local → open live → opt-in installs → credentialed).
  Every prompt has a unique id `P1`…`P99`; ids are stable so you can track which
  you've run. Each phase lists the servers it needs and how to enable them.
- **Section 3 — Optional PDAL backend** and **Section 4 — Optional companion**.
- **Section 5 — Coverage checklist**: what's verified vs. still untested.

Phases 1–8 need **no credentials**. Phase 9 confirms the *clean-refusal* behavior
of credentialed servers with **no accounts**. Phase 10 is the only one that needs
real accounts (and is the only place anything can incur cost).

---

## 1. Setup

There are three distinct "where"s — keep them separate:

- **Install location** (the venv + packages): stays in *this* repo. `mcp.json`
  points at the servers by absolute path, so the workspace you open does not
  need the code.
- **Power location** (`~/.kiro/powers/kiro-geospatial/`): global to Kiro,
  copied once.
- **Acceptance workspace** (where you type prompts): a **separate scratch
  folder**, not this repo. It does not need to be a git repo.

> **Why a separate workspace?** It keeps this dev repo clean and gives
> **predictable steering/skill activation** (steering triggers on file patterns
> like `*.tif`, `*stac*`, `*zonal*`; this repo is full of matching files). A
> clean folder also mirrors what a real user does.

### Step 1 — install (in this repo, once)

```bash
# From this repo's root, in a Python >=3.10 environment:
python3.12 -m venv .venv && . .venv/bin/activate
pip install mcp
pip install -e ./packages/geo-common -e ./packages/kiro-geospatial \
  -e ./packages/geo-stac -e ./packages/geo-vector \
  -e ./packages/geo-ops -e ./packages/geo-foundation-models

# Note the absolute path to the venv's console scripts — you'll need it below:
echo "$(pwd)/.venv/bin"        # Windows: $(pwd)\.venv\Scripts
```

### Step 2 — install the Power files Kiro reads (global, once)

```bash
mkdir -p ~/.kiro/powers/kiro-geospatial
cp -r POWER.md bundle-manifest.json skills steering ~/.kiro/powers/kiro-geospatial/
```

### Step 3 — create a scratch acceptance workspace

```bash
mkdir -p ~/geo-pack-acceptance/.kiro/settings
cd ~/geo-pack-acceptance
```

Create `~/geo-pack-acceptance/.kiro/settings/mcp.json`. Each `command` must be
the **full absolute path** to the console script from Step 1 (it must begin with
`/`; a relative path resolves wrong and yields `ENOENT`). Paste the `echo` output
from Step 1 in place of `<VENV_BIN>`:

```json
{
  "mcpServers": {
    "geo-stac":              { "command": "<VENV_BIN>/geo-stac" },
    "geo-vector":            { "command": "<VENV_BIN>/geo-vector" },
    "geo-ops":               { "command": "<VENV_BIN>/geo-ops" },
    "geo-foundation-models": { "command": "<VENV_BIN>/geo-foundation-models" }
  }
}
```

Verify a path resolves before launching Kiro: `ls <VENV_BIN>/geo-stac` prints
the path. On Windows the paths are `...\.venv\Scripts\geo-stac.exe`, etc.

### Step 4 — activate

Open `~/geo-pack-acceptance` as the Kiro workspace → Powers panel → activate
**kiro-geospatial** → confirm the servers show as connected and their tools are
listed in the MCP panel.

> **Capability discovery is a Powers-panel check, not a chat prompt.** The
> Resource Catalog / Onboarding Dashboard (which capabilities exist, which are
> installed, the `uvx` install command for ones that aren't) is a **Hub** concern
> Kiro renders from `POWER.md` / `bundle-manifest.json` in the **Powers panel**.
> The Hub (`kiro-geospatial`) is deliberately not a tool MCP server, so there's
> no "search the catalog" tool in chat — verify discoverability by inspecting the
> Powers panel, not by prompting.

> **Reconnect after editing `mcp.json`.** Each later phase adds servers to
> `mcp.json`; toggle the affected servers off/on in Kiro to load the change.
> Editable installs mean code changes also reload on reconnect — no reinstall.

---

## 2. Acceptance phases

### Phase 1 — MVP core (local, plus two live reads)

The four MVP servers (`geo-ops`, `geo-stac`, `geo-vector`,
`geo-foundation-models`), already wired in Section 1.

| # | Prompt | Exercises | Expect |
|---|--------|-----------|--------|
| P1 | "Reproject the point [-122.4194, 37.7749] from EPSG:4326 to EPSG:3857." | `geo-ops` `transform_crs` | A point near `[-13,627,665, 4,547,000]` |
| P2 | "Is this polygon valid? `{type:Polygon, coordinates:[[[0,0],[1,1],[1,0],[0,1],[0,0]]]}`" | `geo-ops` `validate_geometry` | Reported invalid (self-intersection) with a reason |
| P3 | "Search STAC for Sentinel-2 imagery over bbox -122.5,37.7,-122.3,37.9 in Jan 2024, limit 5." | `geo-stac` `stac_search` | Up to 5 items with assets + datetime |
| P4 | "Get OpenStreetMap features for bbox `-122.4200,37.7790,-122.4190,37.7798` filtered to the `amenity` layer." | `geo-vector` `vector_features` | A small FeatureCollection (a handful of `amenity` nodes) |
| P5 | "Embed a 64x64x3 GTiff tile (no inline pixel data) with the Clay model, then run change detection between that embedding and an identical copy of it." | `geo-foundation-models` `embed_tile` + `detect_change` | A 768-d Clay embedding, and change `0.0` for identical embeddings |
| P6 | "Segment a 16x16 GTiff tile (no inline pixel data) with the SAMGeo segmenter." | `geo-foundation-models` `segment` | A `SegmentationMask` covering the tile (one whole-tile segment when no pixel data) |

**Guard checks**

| # | Prompt | Expect |
|---|--------|--------|
| P7 | "Reproject geometry `{}` (no type)." | A **validation error** naming the bad parameter, no crash |
| P8 | "Search STAC with start date after end date." | A **validation error** (start-after-end), no partial result |

> **`vector_features` payload sizing.** Overpass returns *every* element in the
> extent, so a 1 km² urban bbox can return tens of thousands of features. Keep
> runs small with a tiny bbox (P4's box is ~0.008 km²) and the `layers` filter.
> A call matching more than `max_features` (default **2,000**) is refused with a
> validation error. Overture Maps is not queried by default (no public REST
> endpoint); only OpenStreetMap/Overpass runs unless configured.

> **`embed_tile` tile shape.** Pass a structured `tile`: required `width`,
> `height`, `format` (`geotiff`/`gtiff`/`cog`/`png`/`jpeg`/`jpg`/`numpy`), plus
> optional `bands`, `dtype`, and a flat `data` array. **Omit `data`** for
> acceptance runs (the embedding is derived from the tile's structure, keeping
> the request tiny). `model` is one of `Clay` (768-d), `Prithvi-EO-2.0` (1024-d),
> `SatCLIP` (256-d), matched case-insensitively.

> **`segment` mask sizing.** Returns a flat per-pixel mask, capped at
> `max_mask_pixels` (default **65,536** = 256x256); keep tiles small (e.g. 64x64).

> **Two different "Clay"s.** `embed_tile`'s `Clay` is a 768-d deterministic local
> stub; the separate `lookup_embeddings` tool retrieves *real* Clay v1.5 (1024-d)
> from the open LGND/Source Cooperative dataset (different vector space, not
> comparable). `lookup_embeddings` reads large cloud GeoParquet (best in-region,
> can take minutes), needs the `[clay-lookup]` extra, and is not a quick-
> acceptance prompt.

### Phase 2 — pure-compute expansion (local, no credentials)

Adds `geo-index` and `geo-embedding-search` (and exercises `geo-ops`
`spatial_join`/`overlay`). All local — deterministic, chat-safe.

```bash
pip install -e ./packages/geo-index -e ./packages/geo-embedding-search
# mcp.json: add  "geo-index": {...}  and  "geo-embedding-search": {...}
```

| # | Prompt | Exercises | Expect |
|---|--------|-----------|--------|
| P9 | "Spatial-join these with the `intersects` predicate — left: a Point at [0.5, 0.5] with property id='pt'; right: a unit-square Polygon `[[[0,0],[1,0],[1,1],[0,1],[0,0]]]` with property zone='A'." | `geo-ops` `spatial_join` | One feature: the point, with merged properties `{id: "pt", zone: "A"}` |
| P10 | "Overlay (intersection) polygon A `[[[0,0],[2,0],[2,2],[0,2],[0,0]]]` with polygon B `[[[1,1],[3,1],[3,3],[1,3],[1,1]]]`." | `geo-ops` `overlay` | One `Polygon` — the overlap (1×1 square from [1,1] to [2,2]) |
| P11 | "What's the H3 cell at resolution 9 for longitude -122.4194, latitude 37.7749?" | `geo-index` `index_cell` | A well-formed H3 id (e.g. `89283082803ffff`) |
| P12 | "What's the S2 cell at level 15 for longitude -122.4194, latitude 37.7749?" | `geo-index` `index_cell` | A well-formed S2 id (e.g. `8085809ec`) |
| P13 | "Store the embedding [0.1, 0.2, 0.3] with metadata {label: 'a'}, then search for the 5 nearest embeddings to [0.1, 0.2, 0.3]." | `geo-embedding-search` `store_embedding` + `search_embeddings` | Store confirms `retrievable: true` (dim 3); search returns 1 hit, `similarity: 1.0` |

**Guard checks**

| # | Prompt | Expect |
|---|--------|--------|
| P14 | "Spatial-join the same two collections with predicate `nonsense`." | A **validation error** listing supported predicates |
| P15 | "Index longitude -122.4194, latitude 37.7749 with H3 at resolution 99." | A **validation error** (resolution out of 0–15) |
| P16 | "Search embeddings with k = 0." | A **validation error** (`k` must be 1–1000) |

### Phase 3 — open live-API servers (no credentials)

Real public upstreams, no keys. Values vary; the table gives stable inputs and
the expected shape.

```bash
pip install -e ./packages/geo-geocode-route -e ./packages/geo-terrain \
  -e ./packages/geo-biodiversity -e ./packages/geo-weather-climate
# mcp.json: add geo-geocode-route, geo-terrain, geo-biodiversity, geo-weather-climate
```

| # | Prompt | Exercises | Expect |
|---|--------|-----------|--------|
| P17 | "Geocode '1600 Amphitheatre Parkway, Mountain View, CA'." | `geo-geocode-route` `geocode` | A coordinate near lon **-122.085**, lat **37.423** (source `nominatim`) |
| P18 | "Route by car from [-122.4194, 37.7749] to [-122.2712, 37.8044]." | `geo-geocode-route` `route` | A `LineString`, ~**18 km** / ~**20 min**, source `osrm` |
| P19 | "What's the elevation at longitude -122.4194, latitude 37.7749?" | `geo-terrain` `elevation` | A scalar in metres (~**33 m** from SRTM, the default source) |
| P20 | "Species occurrences for bbox -122.45,37.74,-122.39,37.80, taxon 'Aves', limit 5." | `geo-biodiversity` `species_occurrences` | Up to 5 GBIF occurrence records (lat/lon, species, dataset) |
| P21 | "Weather observations at longitude -122.4194, latitude 37.7749 for 2024-06-01 to 2024-06-02." | `geo-weather-climate` `observations` | Hourly Open-Meteo observations (`temperature_2m`, `relative_humidity_2m`, …) |

**Guard checks**

| # | Prompt | Expect |
|---|--------|--------|
| P22 | "Route from [-122.4194, 37.7749] to [-122.2712, 37.8044] with profile 'teleport'." | A **validation error** (supported: `car`/`bike`/`foot`) |
| P23 | "Geocode the empty string ''." | A **validation error** naming the `address` parameter |
| P24 | "Weather observations at the same point for 2024-06-02 to 2024-06-01 (start after end)." | A **validation error** (start later than end) |
| P25 | "Elevation at that point with source 'mars-dem'." | A **validation error** listing the configured sources |

> **Notes.** Public services (Nominatim, OSRM, GBIF, Open-Meteo) with no key —
> transient hiccups are availability errors, not Power failures; retry.
> `geo-terrain` supports SRTM (default) and 3DEP via the public OpenTopoData
> endpoint, overridable to a self-hosted instance.

### Phase 4 — formats, imagery, raster, point cloud

Mostly local/deterministic; two prompts read a real COG over HTTP byte ranges.

```bash
pip install -e ./packages/geo-formats \
  -e ./packages/geo-raster -e ./packages/geo-pointcloud
# mcp.json: add geo-formats, geo-raster, geo-pointcloud
```

| # | Prompt | Exercises | Expect |
|---|--------|-----------|--------|
| P26 | "Write a COPC point cloud to /tmp/cloud.copc with 3 points: (-122.42, 37.77, 10) class 2, (-122.41, 37.78, 12.5) class 2, (-122.40, 37.79, 15) class 6 — then read it back." | `geo-pointcloud` `write_pointcloud` + `read_pointcloud` | Writes/reads back **3** points (CRS EPSG:4326) |
| P27 | "Read /tmp/cloud.copc clipped to bounds min_x -122.425, min_y 37.765, max_x -122.415, max_y 37.775." | `geo-pointcloud` `read_pointcloud` (windowed) | **1** point (only the one inside the window) |
| P28 | "Convert a FeatureCollection of two points — (-122.42, 37.77) name 'a' and (-122.40, 37.79) name 'b' — to GeoParquet at /tmp/out.parquet." | `geo-formats` `to_geoparquet` | GeoParquet written; `feature_count` **2**, column `name`, CRS EPSG:4326 |
| P29 | "Validate that /tmp/out.parquet is valid GeoParquet." | `geo-formats` `validate_format` | `valid: true`; `geometry` column + `has_geo_metadata: true` |
| P30 | "Read an 8×8 pixel window at col 0, row 0 from this COG: `https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/10/S/EG/2024/1/S2A_10SEG_20240104_0_L2A/B04.tif`" | `geo-raster` `read_window` | An 8×8 `RasterArray`, `dtype` `uint16`, band `[1]` |
| P31 | "Run band math `B1 / 10000` over a 4×4 window at col 0, row 0 of that same COG." | `geo-raster` `band_math` | A 4×4 array of reflectance-scaled values (~0.01) |
| P31b | "Run zonal band math NDVI `(B08 - B04) / (B08 + B04)`, binding B08 to `https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/11/S/KV/2026/6/S2B_11SKV_20260614_0_L2A/B08.tif` and B04 to the sibling `.../B04.tif`, over this EPSG:32611 zone: `{\"type\":\"Polygon\",\"coordinates\":[[[247089.7,3921520.3],[247543.8,3921507.5],[247559.4,3922062.2],[247105.3,3922075.0],[247089.7,3921520.3]]]}`" | `geo-raster` `zonal_band_math` | One zone with `min`/`max`/`mean`/`sum`/`count` over true per-pixel NDVI (~2,500 px; mean ≈ 0.2, `no_data: false`) — NIR and Red are **separate** COGs |

**Guard checks**

| # | Prompt | Expect |
|---|--------|--------|
| P32 | "Run band math `b1 * 2` (lowercase) over that COG window." | A **validation error**: bands must be `B<n>` (e.g. `B1`) |
| P33 | "Read the point cloud /tmp/does-not-exist.copc." | A **not-found** error |
| P34 | "Validate that /tmp/out.parquet is a valid COG." | `valid: false` with a reason (it's GeoParquet) — a clean negative |
| P34b | "Run the same zonal band math NDVI but bind **only** B08 (omit the B04 asset)." | A **validation error** naming `B4` (referenced band with no matching asset) — nothing computed |

> **Notes.** `band_math` references bands as `B<n>` (capital B, 1-based), and
> zero-padded tokens work (`B08`/`B04` as Sentinel-2 uses them). `zonal_band_math`
> (P31b) computes a **true per-pixel** index (NDVI/NDWI/SAVI/EVI/… — any band-math
> expression) across bands that live in **separate single-band COGs** and reduces
> it to per-zone stats in one call; its zone polygons must be in the raster's own
> CRS (EPSG:32611 for this tile), which is why P31b supplies a UTM polygon. Plain
> `zonal_statistics` is omitted from the quick table for the same CRS reason.
> `read_window`/`band_math`/`zonal_band_math` issue real HTTP range requests — a
> transient blip is an availability error.

### Phase 5 — more open tools (no credentials)

Added tools on Phase 2/3 servers: reverse-geocode, slope, hillshade, isochrone,
iNaturalist occurrences, NWS observations, and multi-endpoint STAC. No new
servers — reconnect the affected ones. Live calls, so values vary.

| # | Prompt | Exercises | Expect |
|---|--------|-----------|--------|
| P35 | "Reverse-geocode longitude -122.085, latitude 37.423." | `geo-geocode-route` `reverse_geocode` | A `label` near the Mountain View / Googleplex area (source `nominatim`) |
| P36 | "Compute the slope over bbox -119.56,37.74,-119.54,37.76 (Sierra Nevada)." | `geo-terrain` `slope` | A grid of per-cell slopes in degrees (0–90); steep terrain reaches ~60° |
| P37 | "Compute hillshade over bbox -119.56,37.74,-119.54,37.76 with the default sun azimuth/altitude." | `geo-terrain` `hillshade` | A grid of hillshade values (0–255) |
| P37b | "Compute the aspect over bbox -119.56,37.74,-119.54,37.76 (Sierra Nevada)." | `geo-terrain` `aspect` | A grid of compass bearings in degrees (0–360, the downslope direction; -1 for flat cells) |
| P38 | "Compute 5- and 10-minute driving isochrones from [-122.4194, 37.7749]." | `geo-geocode-route` `isochrone` | A GeoJSON `FeatureCollection` of reachability polygons, one per contour (via Valhalla) |
| P39 | "Species occurrences for bbox -122.45,37.74,-122.39,37.80, taxon 'Aves', limit 5, source iNaturalist." | `geo-biodiversity` `species_occurrences` (iNaturalist) | Up to 5 iNaturalist records (`source: inaturalist`) |
| P40 | "Weather observations at longitude -77.04, latitude 38.90 for the last 3 days, source nws." | `geo-weather-climate` `observations` (NWS) | Recent `api.weather.gov` station observations (empty for a non-US point or an old range) |
| P41 | "Search STAC across multiple endpoints for the `sentinel-2-l2a` collection over bbox -122.5,37.7,-122.3,37.9 in Jan 2024, limit 20." | `geo-stac` `stac_search_multi` | A round-robin-merged result: Earth Search, Planetary Computer, and Copernicus all contribute, de-duplicated, with per-endpoint provenance in `sources` |

**Guard checks**

| # | Prompt | Expect |
|---|--------|--------|
| P42 | "Reverse-geocode longitude 999, latitude 999." | A **validation error** (out-of-range coordinate), no source queried |
| P43 | "Isochrone from [-122.4194, 37.7749] with contours_minutes [] (empty)." | A **validation error** (at least one positive time budget required) |

> **Notes.** A date-only bound (`2024-01-01`) is expanded to a full-day RFC3339
> interval before search, so STAC ranges no longer silently return 0 items.
> `slope`/`hillshade` need an *extent* (bbox), not a point; windowed terrain reads
> are auto-batched to ≤100 locations for the OpenTopoData cap. In
> `stac_search_multi`, use `limit` ≥ ~15 to see multiple catalogs contribute;
> `dedupe="scene"` (default) collapses the same physical scene across catalogs,
> `dedupe="id"` keeps each copy.

### Phase 6 — geo-ogc (OGC API - Features)

Bridges any conformant OGC API - Features service (pygeoapi, GeoServer OGC API,
ldproxy) — endpoint + collection id per call, no bundled endpoint, no credential.

```bash
pip install -e ./packages/geo-ogc
# mcp.json:  "geo-ogc": { "command": "<VENV_BIN>/geo-ogc" }
```

| # | Prompt | Exercises | Expect |
|---|--------|-----------|--------|
| P44 | "Fetch OGC features from endpoint https://demo.pygeoapi.io/master, collection lakes, bbox -130,20,-60,55, limit 5." | `geo-ogc` `ogc_features` | A GeoJSON `FeatureCollection` of up to 5 lake features, with `number_matched`/`number_returned` and `endpoint`/`collection` provenance |

**Guard checks**

| # | Prompt | Expect |
|---|--------|--------|
| P45 | "Fetch OGC features from endpoint https://demo.pygeoapi.io/master, collection lakes, bbox 200,0,201,1." | A **validation error** (longitude out of range), no request made |
| P46 | "Fetch OGC features from endpoint https://demo.pygeoapi.io/master, collection does-not-exist, bbox -130,20,-60,55." | An **upstream** error reporting the collection was not found (HTTP 404) |

> **Note.** The public pygeoapi demo is a shared sandbox; a transient
> `network`/`upstream` error is an availability blip — retry or point at another
> OGC API - Features instance (the service URL is a parameter). The
> `https://demo.pygeoapi.io/master` endpoint is the upstream project's own fixed
> public deployment path (an external URL the pack does not control), not a
> source-branch reference.

### Phase 7 — geo-3d (inspect, tile, mesh)

Works with 3D geospatial formats: **inspects/validates** OGC 3D Tiles tilesets
and glTF/GLB, **tiles** point clouds into 3D Tiles (`.pnts`, optional octree LOD),
and **meshes** DEM grids into glTF/GLB terrain. All local, no native libs, no
credential.

```bash
pip install -e ./packages/geo-3d
# mcp.json:  "geo-3d": { "command": "<VENV_BIN>/geo-3d" }
```

| # | Prompt | Exercises | Expect |
|---|--------|-----------|--------|
| P47 | "Inspect this 3D Tiles tileset: `{\"asset\":{\"version\":\"1.1\"},\"root\":{\"geometricError\":100,\"refine\":\"ADD\",\"boundingVolume\":{\"box\":[0,0,0,1,0,0,0,1,0,0,0,1]},\"content\":{\"uri\":\"r.b3dm\"},\"children\":[{\"geometricError\":0,\"boundingVolume\":{\"box\":[0,0,0,1,0,0,0,1,0,0,0,1]},\"content\":{\"uri\":\"a.b3dm\"}}]}}`" | `geo-3d` `inspect_tileset` | `valid: true`, asset 1.1, geometric_error 100, refine ADD, tile_count 2, max_depth 1, no issues |
| P48 | "Inspect a glTF asset at https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Box/glTF-Binary/Box.glb" | `geo-3d` `inspect_gltf` | `valid: true`, `binary: true`, glTF 2.0, meshes 1 / nodes 2 / materials 1 |
| P49 | "Tile these 3 points into 3D Tiles under /tmp/my-tileset: [-122.42, 37.77, 10], [-122.41, 37.78, 12.5], [-122.40, 37.79, 15]." | `geo-3d` `points_to_3d_tiles` | `tileset.json` + `points.pnts`; `point_count: 3`; box bounding volume; validates with `inspect_tileset` |
| P50 | "Tile those same 3 points with colors [[255,0,0],[0,255,0],[0,0,255]] under /tmp/my-tileset-rgb." | `geo-3d` `points_to_3d_tiles` (colors) | Same as P49 plus `has_colors: true` |
| P51 | "Mesh this DEM into a glTF terrain at /tmp/terrain.glb, bbox -122.45,37.74,-122.44,37.75: [[10,11,12],[10.5,12,13],[11,12.5,14]]." | `geo-3d` `dem_to_mesh` | A `.glb`: `vertex_count: 9`, `triangle_count: 8`, per-vertex NORMALs; validates with `inspect_gltf` |
| P52 | "Tile points into 3D Tiles under /tmp/octree with max_points_per_tile 8 (ask the agent to generate a 7×7×1 grid of ~49 points)." | `geo-3d` `points_to_3d_tiles` (octree) | `tile_count > 1`, `max_depth ≥ 1`; per-tile counts sum to the total; validates |

**Guard checks**

| # | Prompt | Expect |
|---|--------|--------|
| P53 | "Inspect this 3D Tiles tileset: `{\"asset\":{\"version\":\"1.1\"},\"root\":{\"content\":{\"uri\":\"x.b3dm\"}}}` (root has no geometricError or boundingVolume)." | A report `valid: false` with `issues` naming the missing root fields — a clean negative |
| P54 | "Inspect a glTF asset at https://example.com/not-a-model.txt" | `valid: false` (or upstream/not-found if the URL 404s) — never a crash |
| P55 | "Tile an empty point list [] under /tmp/empty-tileset." | A **validation error** naming `points` — nothing written |
| P56 | "Tile 3 points with colors [[255,0,0]] (mismatched length) under /tmp/bad-colors." | A **validation error** naming `colors` — nothing written |
| P57 | "Mesh a non-rectangular DEM [[1,2,3],[1,2]] at /tmp/bad.glb, bbox -122.45,37.74,-122.44,37.75." | A **validation error** naming `elevations` (grid not rectangular) — nothing written |

> **Note.** Inputs can be inline JSON, a local file path, or an `https` URL. Some
> chat clients drop a large/sparse **inline** JSON argument; if an inline tileset
> seems to vanish (tripping the "provide inline tileset or a source" guard), pass
> a `source` (path/URL) instead.

### Phase 8 — geo-query DuckDB engine (open, opt-in install)

`geo-query`'s DuckDB Spatial engine is the open default, but DuckDB is an
optional dependency — install the extra to make `spatial_sql` execute.

```bash
pip install -e ./packages/geo-query    # add to mcp.json: "geo-query": {...}
pip install -e "./packages/geo-query[duckdb]"
```

| # | Prompt | Exercises | Expect |
|---|--------|-----------|--------|
| P58 | "Run spatial SQL `SELECT 1 AS one` on the default engine." | `geo-query` `spatial_sql` (DuckDB) | One row: column `one`, value `1`, `engine: duckdb` |
| P59 | "Run spatial SQL `SELECT ST_AsText(ST_Point(1, 2)) AS wkt`." | `geo-query` `spatial_sql` (DuckDB spatial ext.) | One row, `wkt` = `POINT (1 2)` — confirms the `spatial` extension loaded |
| P60 | "Run spatial SQL `SELECT name FROM pts` with sources `{pts: '<path-or-URL to a Parquet/GeoJSON file>'}`." | `geo-query` `spatial_sql` (sources → views) | The rows of that file (DuckDB reads Parquet/GeoJSON/CSV from path/`s3://`/`https://`) |

**Guard check**

| # | Prompt | Expect |
|---|--------|--------|
| P61 | "Run spatial SQL `SELECT FROM WHERE` (malformed) on DuckDB." | A **validation error** — DuckDB's parse error surfaced cleanly, not a crash |

> Without the `[duckdb]` extra, `spatial_sql` on the default engine returns an
> **upstream** "DuckDB Spatial engine is not configured/available" — that's the
> expected refusal, not a failure (see P64).

### Phase 9 — credentialed servers: clean refusals (no accounts)

The credentialed/engine-backed servers (`geo-query`, `geo-warehouse`,
`geo-commercial-imagery`, `aws-geo-compute`) **start fine with no credentials**;
the healthy unconfigured outcome is a correctly-categorized refusal, not a
result. Two `aws-geo-compute` tools run fully offline. This phase needs **no
accounts** — it verifies the refusal/validation behavior.

```bash
pip install -e ./packages/geo-warehouse \
  -e ./packages/geo-commercial-imagery -e ./packages/aws-geo-compute
# mcp.json: add geo-query, geo-warehouse, geo-commercial-imagery, aws-geo-compute
```

**These two run offline (no credentials needed):**

| # | Prompt | Exercises | Expect |
|---|--------|-----------|--------|
| P62 | "Plan execution for a 1 GB dataset with an 'ad-hoc-over-s3' access pattern." | `aws-geo-compute` `plan_execution` | An `ExecutionPlan` — engine `athena`, `delegate: false` (deterministic) |
| P63 | "Which execution target handles a 'bulk-cog-conversion' task?" | `aws-geo-compute` `select_target` | `AWS Batch` (deterministic mapping) |

**Refusal / validation guard checks** (healthy outcome = a clean, categorized error):

| # | Prompt | Expect |
|---|--------|--------|
| P64 | "Run spatial SQL `SELECT 1` on the default (DuckDB) engine" *(without the `[duckdb]` extra)*. | An **upstream** "DuckDB Spatial engine is not configured/available". (With the extra installed — Phase 8 — it returns a one-row result instead.) |
| P65 | "Run warehouse spatial SQL `SELECT 1` on the snowflake engine." | An **authentication** error naming `SNOWFLAKE_CONNECTION` |
| P66 | "Search Maxar commercial imagery over bbox -122.5,37.7,-122.3,37.9 for Jan 2024." | An **authentication** error naming `SENTINELHUB_CLIENT_ID` — no partial data |
| P67 | "Submit a 'large-zonal-statistics' job." | An **authentication** error naming `AWS_ACCESS_KEY_ID` |
| P68 | "Run warehouse spatial SQL with a blank query `''` on bigquery." | A **validation** error (query must be non-empty) |
| P69 | "Plan execution with access pattern 'warp-speed'." | A **validation** error listing valid patterns |
| P70 | "Which target handles a 'teleport-raster' task?" | A **validation** error listing supported task types |
| P71 | "Search Landsat commercial imagery over that bbox for Jan 2024." | A **validation** error naming `provider` (only `MAXAR` / `PLANET`) — account-free |
| P72 | "Search Planet commercial imagery over that bbox for Jan 2024." | An **authentication** error naming `PL_API_KEY` (Planet routes to Planet's own API) |
| P73 | "Weather observations at longitude -77.04, latitude 38.90 for 2024-06-01 to 2024-06-02, source cdo." | An **authentication** error naming `NOAA_CDO_TOKEN` |
| P74 | "Geocode '1600 Pennsylvania Ave NW, Washington, DC' with source amazon-location." | An **authentication** error naming `AMAZON_LOCATION_API_KEY` (omitting `source` uses the open Nominatim/Photon chain) |
| P75 | "Species occurrences for taxon 'Panthera leo', source IUCN." | An **authentication** error naming `IUCN_TOKEN` (IUCN is name-based: selecting it also requires a `taxon`) |

> **Open vs. credentialed.** A missing *open* engine (DuckDB) surfaces as
> `upstream` "not available"; a missing *credentialed* engine/source surfaces as
> `authentication` **naming the `mcp.json` key**, never the secret. These
> refusals are the documented healthy behavior when no credential is wired.

### Phase 10 — credentialed success paths (needs real accounts)

The only phase that needs accounts — and the only place anything can incur
cost. Each capability is already verified at **Layer 1** (fake-client unit tests
in CI, no account); this is **Layer 2**, the live round-trip. Set credentials via
`mcp.json` env (or env references), never paste secrets, and reconnect.

> Secrets stay in env. For warehouse engines, each `*_CONNECTION` can be a path
> to a `chmod 600` JSON file instead of inline JSON. Ordering/quota-spending
> prompts are flagged; start with the read/quote prompts.

> **Cost warning.** Phase 10 creates billable resources. AWS Batch jobs run on
> EC2 or Fargate compute; Amazon Redshift Serverless workgroups bill per
> RPU-hour and provisioned clusters bill per node-hour; Snowflake warehouses
> consume credits; Databricks SQL warehouses consume DBUs; and Athena scans
> billable S3 data and writes to a staging bucket. These resources keep
> incurring charges until you stop or delete them. Run the cleanup steps below
> as soon as testing is complete.

#### 10a. `geo-commercial-imagery` — Maxar (Sentinel Hub TPDI) + Planet (Planet APIs)

Routes by `provider`. Maxar → Sentinel Hub TPDI (`SENTINELHUB_CLIENT_ID` /
`SENTINELHUB_CLIENT_SECRET`); Planet → Planet's own Data + Orders APIs
(`PL_API_KEY`). Airbus is no longer offered. No extra to install.

| # | Prompt | Expect (live) |
|---|--------|---------------|
| P76 | "Search Maxar commercial imagery over bbox -122.5,37.7,-122.3,37.9 for Jan 2024, max cloud 20." | Maxar `StacItem`s (TPDI `/dataimport/search`, ids from `catalogID`). Empty = no archive imagery, not an error. |
| P77 | "Create a Maxar order for product ids [\<catalogID from P76\>] over that bbox (do not confirm)." | An `OrderConfirmation`, `confirmed: false`, `sqkm` cost — **no quota spent**. |
| P78 | "Search Planet (PSScene) imagery over that bbox for Jan 2024, max cloud 20." | Planet `StacItem`s via Planet Data API `quick-search` (`PL_API_KEY` Basic auth). |
| P79 | "Order Planet item ids [\<id from P78\>], bundle analytic_udm2, and confirm." | **Spends quota.** A Planet `OrderConfirmation` (`state: queued`, `confirmed: true`). Without `confirm=true` it's refused by design (Planet has no draft step). |

> On any upstream rejection both providers surface the provider's own message
> (`detail.sh_message` / `detail.planet_message`). A 403 (Maxar) or order-time
> "no access to assets" (Planet) is an **account entitlement** matter, not a
> connector bug.

#### Configuring a warehouse connection (applies to 10b–10e)

All four warehouse engines share one rule. The `<ENGINE>_CONNECTION` env value is
**either**:

- **(A) a path to a JSON file** holding the connection object — **recommended**,
  because it keeps the secret out of `mcp.json` and avoids JSON-inside-JSON
  escaping; or
- **(B) the JSON object inline**, as an escaped string.

The per-call **`connection` argument is the non-secret database/schema**, never a
password/token. You can configure several engines at once by setting multiple
`*_CONNECTION` keys in the same `geo-warehouse` env block.

**Option A — file (recommended).** Write the JSON to a file only you can read,
then point the env at its absolute path:

```bash
mkdir -p ~/.config/geo
cat > ~/.config/geo/redshift.json <<'JSON'
{ "host": "my-cluster.abc123.us-east-1.redshift.amazonaws.com", "port": 5439, "user": "awsuser", "password": "REPLACE_ME" }
JSON
chmod 600 ~/.config/geo/redshift.json
```
```json
"geo-warehouse": {
  "command": "<VENV_BIN>/geo-warehouse",
  "env": { "REDSHIFT_CONNECTION": "/Users/you/.config/geo/redshift.json" }
}
```

**Option B — inline JSON string.** The whole object is one JSON string, so the
inner quotes must be escaped (`\"`):

```json
"geo-warehouse": {
  "command": "<VENV_BIN>/geo-warehouse",
  "env": { "REDSHIFT_CONNECTION": "{\"host\":\"my-cluster...redshift.amazonaws.com\",\"port\":5439,\"user\":\"awsuser\",\"password\":\"REPLACE_ME\"}" }
}
```

> The connector accepts both transparently: if the value starts with `{` it's
> parsed as inline JSON, otherwise it's treated as a file path. A malformed value
> surfaces as an `authentication` error naming the key (never the contents).

#### 10b. `geo-warehouse` — BigQuery (free sandbox, no card)

```bash
pip install -e "./packages/geo-warehouse[bigquery]"
# Service-account key:  env { "BIGQUERY_CREDENTIALS": "/abs/path/key.json" }
# OR (if org policy blocks SA keys) Application Default Credentials:
#   run `gcloud auth application-default login` && set quota project, then
#   env { "BIGQUERY_USE_ADC": "1" }
```
The `connection` argument is your **GCP project id**.

| # | Prompt | Expect (live) |
|---|--------|---------------|
| P80 | "Run warehouse spatial SQL `SELECT 1 AS n` on the bigquery engine, connection '\<project-id\>'." | One row, `engine: bigquery`. |
| P81 | "Run `SELECT ST_ASTEXT(ST_GEOGPOINT(-122.4, 37.8)) AS wkt` on bigquery, connection '\<project-id\>'." | `POINT(-122.4 37.8)` — confirms BigQuery GIS. |
| P82 | "Run `SELECT FROM` on bigquery, connection '\<project-id\>'." | A clean **validation** error (BigQuery 400 → validation), surfacing BigQuery's own parser message. |

#### 10c. `geo-warehouse` — Redshift (your AWS Redshift cluster)

```bash
pip install -e "./packages/geo-warehouse[redshift]"
```

`REDSHIFT_CONNECTION` is a JSON object (file path or inline — see "Configuring a
warehouse connection" above). **Where to find each value:**

- `host` — the cluster **endpoint** without the trailing `:5439/dev`. Provisioned:
  Redshift console → your cluster → *General information* → *Endpoint* (e.g.
  `my-cluster.abc123.us-east-1.redshift.amazonaws.com`). Serverless: the
  *Workgroup* → *Endpoint* (`<workgroup>.<account>.<region>.redshift-serverless.amazonaws.com`).
- `port` — `5439` (default).
- `user` / `password` — a database user (the admin user set at cluster creation,
  or one you created with `CREATE USER`).
- The **database** (e.g. `dev`) is **not** in the JSON — you pass it as the
  `connection` argument per query.

Password-auth example (`~/.config/geo/redshift.json`):
```json
{ "host": "my-cluster.abc123.us-east-1.redshift.amazonaws.com", "port": 5439, "user": "awsuser", "password": "REPLACE_ME" }
```

**More secure IAM auth** (no DB password — uses your AWS credentials from the
environment): provisioned `{ "iam": true, "cluster_identifier": "my-cluster", "region": "us-east-1", "db_user": "awsuser" }`; Redshift Serverless
`{ "iam": true, "is_serverless": true, "serverless_work_group": "<workgroup>", "region": "us-east-1" }` (the database, e.g. `dev`, is passed as the per-query `connection` argument, not in the JSON). TLS is enforced
either way. Your network must reach the cluster (publicly accessible, or run
Kiro where the VPC/security group allows 5439).

> **Serverless gotcha (verified live):** a Serverless workgroup is **not
> publicly accessible by default**, so the first query will hang ~60s and fail
> with `[upstream] redshift query failed: connection time out` — that's network
> reachability, *not* auth (an IAM/credential problem surfaces immediately as
> `[authentication]`). Fix: in the workgroup's *Network and security*, enable
> **Publicly accessible** and add an inbound rule for TCP **5439** from your IP
> (or run Kiro inside the VPC). Confirm with
> `nc -vz <workgroup>.<account>.<region>.redshift-serverless.amazonaws.com 5439`
> before re-running — no config change is needed once the socket connects.

| # | Prompt | Expect (live) |
|---|--------|---------------|
| P83 | "Run `SELECT 1 AS n` on the redshift engine, connection `dev`." | One row, `engine: redshift`. |
| P84 | "Run `SELECT ST_AsText(ST_Point(-122.4, 37.8)) AS wkt` on redshift, connection `dev`." | `POINT(-122.4 37.8)` (Redshift `GEOMETRY`). |
| P85 | "Run `SELECT FROM` on redshift, connection `dev`." | A clean **validation** error. |

#### 10d. `geo-warehouse` — Snowflake (30-day free trial, no card)

```bash
pip install -e "./packages/geo-warehouse[snowflake]"
```

`SNOWFLAKE_CONNECTION` is a JSON object (file path or inline). **Where to find
each value:**

- `account` — the **account identifier** (the trickiest field). From your
  Snowflake URL `https://<orgname>-<account_name>.snowflakecomputing.com`, use
  `<orgname>-<account_name>` (e.g. `myorg-prod1`). In Snowsight it's under your
  user menu → *Account* → *View account details* → *Account/Server*. The legacy
  `<account_locator>.<region>.<cloud>` form also works.
- `user` / `password` — your Snowflake login (the trial user you created).
- `warehouse` — a virtual warehouse to run on; trials get `COMPUTE_WH` by default.
- `role` — optional (e.g. `ACCOUNTADMIN` on a trial, or `SYSADMIN`).
- The **database** is the `connection` argument, not the JSON; `schema` may be set
  in the JSON if you want a default.

Example (`~/.config/geo/snowflake.json`):
```json
{ "account": "myorg-prod1", "user": "ME", "password": "REPLACE_ME", "warehouse": "COMPUTE_WH", "role": "ACCOUNTADMIN" }
```

More secure auth is supported via the same JSON: key-pair (`"private_key_file": "/path/rsa_key.p8"`, plus `"private_key_file_pwd"` if encrypted) or SSO/OAuth
(`"authenticator": "externalbrowser"` or an OAuth token). Snowflake's transport is
always TLS.

| # | Prompt | Expect (live) |
|---|--------|---------------|
| P86 | "Run `SELECT 1 AS n` on the snowflake engine, connection '\<db\>'." | One row, `engine: snowflake`. |
| P87 | "Run `SELECT ST_ASWKT(ST_MAKEPOINT(-122.4, 37.8)) AS wkt` on snowflake, connection '\<db\>'." | `POINT(-122.4 37.8)` (Snowflake `GEOGRAPHY`). |
| P88 | "Run `SELECT FROM` on snowflake, connection '\<db\>'." | A clean **validation** error. |

> A trial has the sample `SNOWFLAKE_SAMPLE_DATA` database, so you can test with
> `connection: SNOWFLAKE_SAMPLE_DATA` even before creating your own.

#### 10e. `geo-warehouse` — Databricks (Free Edition, no card)

```bash
pip install -e "./packages/geo-warehouse[databricks]"
```

`DATABRICKS_CONNECTION` is a JSON object (file path or inline). **Where to find
each value** — all three come from a **SQL warehouse**: in the Databricks
workspace, open *SQL Warehouses* → your warehouse (Free Edition auto-creates
"Starter Warehouse") → *Connection details* tab:

- `server_hostname` — the *Server hostname* (e.g. `dbc-xxxx.cloud.databricks.com`).
- `http_path` — the *HTTP path* (e.g. `/sql/1.0/warehouses/abc123def456`).
- `access_token` — a personal access token: top-right *Settings* → *Developer* →
  *Access tokens* → *Generate new token* (a `dapi…` string). If your account
  disables PATs, use OAuth M2M instead via `"client_id"` / `"client_secret"`.
- Optional `catalog` (e.g. `main`); the `connection` argument is the **schema**
  (e.g. `default`).

Example (`~/.config/geo/databricks.json`):
```json
{ "server_hostname": "dbc-xxxx.cloud.databricks.com", "http_path": "/sql/1.0/warehouses/abc123def456", "access_token": "dapiREPLACE_ME", "catalog": "main" }
```

| # | Prompt | Expect (live) |
|---|--------|---------------|
| P89 | "Run `SELECT 1 AS n` on the databricks engine, connection `default`." | One row, `engine: databricks`. |
| P90 | "Run `SELECT st_astext(st_point(-122.4, 37.8)) AS wkt` on databricks, connection `default`." | `POINT (-122.4 37.8)`. |
| P91 | "Run `SELECT FROM` on databricks, connection `default`." | A clean error — Databricks' lenient parser reports it as `UNRESOLVED_COLUMN` (`42703`), surfaced as **not-found** (the other engines report it as a **validation** syntax error). |

> Start the SQL warehouse before querying (a stopped warehouse cold-starts on
> first use, which can take a few seconds). Free Edition is fine for these checks.

#### 10f. `geo-query` — Amazon Athena (AWS)

```bash
pip install -e "./packages/geo-query[athena]"
# env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION,
#      ATHENA_S3_STAGING_DIR (s3://amzn-s3-demo-bucket/athena/), optional ATHENA_DATABASE
```
Needs one Glue Data Catalog table (or `information_schema`).

| # | Prompt | Expect (live) |
|---|--------|---------------|
| P92 | "Run spatial SQL `SELECT 1 AS n` on the athena engine." | One row, `engine: athena` (start → poll → return). |
| P93 | "Run `SELECT * FROM <db>.<table> LIMIT 5` on athena." | Up to 5 rows; NULL cells come back as `null`. |
| P94 | "Run `SELECT FROM` on athena." | A clean **validation** error — Athena rejects it at `StartQueryExecution` (`InvalidRequestException`, "Queries of this type are not supported"); a query that instead reaches a FAILED state with a syntax reason maps to validation too. |
| P95 | "Run `SELECT 1` on athena with no `ATHENA_S3_STAGING_DIR` set." | An **authentication** error naming `ATHENA_S3_STAGING_DIR`. |

#### 10g. `aws-geo-compute` — AWS Batch (Batch + Fargate targets)

```bash
pip install -e "./packages/aws-geo-compute[aws]"
# env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION,
#      AWS_BATCH_JOB_QUEUE, AWS_BATCH_JOB_DEFINITION,
#      optional AWS_GEO_COMPUTE_OUTPUT_S3 (s3://amzn-s3-demo-bucket/out)
```

| # | Prompt | Expect (live) |
|---|--------|---------------|
| P96 | "Submit a 'bulk-cog-conversion' job." | A `job_id` + target `aws-batch` (submitted to your Batch queue). |
| P97 | "Check the status of job \<id from P96\>." | One of `queued`/`running`/`succeeded`/`failed`; on success an `output_s3_uri` under your output prefix. |
| P98 | "Submit a 'distributed-spatial-join' job." | A clean **upstream** error: the EMR-Sedona target isn't run by the shipped Batch backend (inject a custom backend). |
| P99 | "Submit a 'bulk-cog-conversion' job with no `AWS_BATCH_JOB_QUEUE` set." | An **upstream** error naming `AWS_BATCH_JOB_QUEUE` (creds present, Batch config incomplete). |

> `plan_execution`/`select_target` (P62/P63) run offline; only `submit`/`status`
> need AWS. EMR-Sedona and SageMaker targets are bring-your-own-backend.

#### 10h. Clean up Phase 10 resources

Phase 10 is the only phase that creates billable resources. When testing is
complete, stop or delete each one to avoid ongoing charges:

- **AWS Batch** — cancel or stop any `running` jobs; if the job queue and job
  definition were created only for testing, delete them. Confirm in the Amazon
  ECS / EC2 consoles that no Fargate tasks or EC2 instances remain.
- **Amazon Redshift** — pause or delete the Serverless workgroup (and its
  namespace) or delete the provisioned cluster; verify nothing is left running.
- **Snowflake** — suspend the warehouse (`ALTER WAREHOUSE <name> SUSPEND;`) or
  let auto-suspend stop it.
- **Databricks** — stop the SQL warehouse.
- **Amazon Athena** — remove the S3 staging directory (`ATHENA_S3_STAGING_DIR`)
  and any CTAS output objects created during testing.
- **Credentials** — revoke or rotate any temporary credentials used, and remove
  test credential files.

Verify in each service console that the resources are fully stopped or deleted.

---

## 3. Optional PDAL backend for geo-pointcloud

The default COPC backend is pure-Python and always available. The production
`PdalCopcBackend` reads/writes standard `.copc.laz` via PDAL, which needs the
native PDAL library (pip alone cannot build it):

```bash
brew install pdal                 # macOS; or: conda install -c conda-forge pdal python-pdal
pip install -e "./packages/geo-pointcloud[pdal]"
pytest packages/geo-pointcloud/tests/test_pdal_backend.py -v
```

> **Note (PDAL + pyarrow).** PDAL links its own native Apache Arrow; importing
> `pdal` into the same process as pip `pyarrow` crashes the Arrow runtime. Each server runs as
> its own process in deployment, so this only affects the single-process test
> runner — the PDAL round-trip test runs in an isolated subprocess. Until the
> native lib is present, that test is **skipped** and the rest of the suite passes.

---

## 4. Optional companion: gdal-mcp (third-party)

`gdal-mcp` is a **third-party** GDAL/Rasterio MCP server
([JordanGunn/gdal-mcp](https://github.com/JordanGunn/gdal-mcp), MIT) — **not part
of this Power and not redistributed with it**; install it separately under its
own license. It's the local-file complement to the pack's cloud-native tools.

> **Prerequisite — `uv`/`uvx` must be installed and addressed by absolute path.**
> This companion launches via `uvx` (from [`uv`](https://docs.astral.sh/uv/)).
> Unlike the pack servers (which run from direct `.venv/bin/...` paths), this is
> the first server invoked through `uvx`, so two things bite:
> 1. **Install `uv`** if it isn't already: `brew install uv` (lands at
>    `/opt/homebrew/bin/uvx`), or `curl -LsSf https://astral.sh/uv/install.sh | sh`
>    (lands in `~/.local/bin`). Confirm with `which uvx`.
> 2. **Use the absolute path** in `command`. Kiro launched from Finder/Dock does
>    not inherit your shell `PATH`, so a bare `"uvx"` fails with
>    `spawn uvx ENOENT` even when `uvx` works in your terminal. Point `command`
>    at the full path (e.g. `/opt/homebrew/bin/uvx`).

```json
"gdal-mcp": {
  "command": "/opt/homebrew/bin/uvx",
  "args": ["--from", "gdal-mcp", "gdal", "--transport", "stdio"],
  "env": { "GDAL_MCP_WORKSPACES": "/path/to/your/geo/data" }
}
```

Smoke prompts: *"Buffer this polygon by 100 m"* (`vector_buffer`) or *"What's the
CRS and size of /path/to/your/data/scene.tif?"* (`raster_info`). Note its
**reflection middleware**: methodology-sensitive tools (reproject, resampling,
query extent) ask for a structured justification before running. These validate
the *companion*, not this Power.

---

## 5. Coverage checklist — verified vs. still untested

Legend: ✅ verified live in Kiro · ◐ partially exercised / blocked by account
entitlement (not a code issue) · ☐ not yet tested.

**Phases 1–8 (no credentials) — ✅ verified.** The local/compute tools (P1–P16),
open live-API tools (P17–P25, P35–P43), formats/imagery/raster/pointcloud
(P26–P34), `geo-ogc` (P44–P46), `geo-3d` (P47–P57), and the DuckDB engine
(P58–P61) were exercised in earlier in-Kiro passes (this is where the six real
bugs were found and fixed). Re-run any after code changes; otherwise consider
them covered.

**Phase 9 (credentialed refusals, no accounts) — ✅ verified.** The clean-refusal
and validation behavior (P62–P75) was confirmed; these need no accounts.

**Phase 10 (credentialed success, needs accounts):**

| Capability | Prompts | Status |
|------------|---------|--------|
| `geo-warehouse` **BigQuery** (scalar, GIS, error) | P80–P82 | ✅ **verified live** (via ADC against the free sandbox) |
| `geo-commercial-imagery` **Planet search** | P78 | ✅ **verified live** (~75 PSScene returned) |
| `geo-commercial-imagery` **Planet order** | P79 | ◐ request shape proven; **account lacks asset entitlement** (Planet 400 "no access to assets") |
| `geo-commercial-imagery` **Maxar** (search/order) | P76–P77 | ◐ reaches TPDI, **account not entitled to Maxar** (403) |
| `geo-warehouse` **Redshift** | P83–P85 | ✅ **verified live** (Serverless workgroup via IAM auth — scalar, `ST_*` spatial, and `42601` syntax error → `[validation]`) |
| `geo-warehouse` **Snowflake** | P86–P88 | ✅ **verified live** (trial account, password auth — scalar, `ST_MAKEPOINT`/`ST_ASWKT` spatial, and `42000`/`001003` syntax error → `[validation]`) |
| `geo-warehouse` **Databricks** | P89–P91 | ✅ **verified live** (Free Edition, PAT auth — scalar, `st_point`/`st_astext` spatial; its lenient parser reports `SELECT FROM` as `UNRESOLVED_COLUMN` `42703` → `[not-found]`) |
| `geo-query` **Athena** | P92–P95 | ✅ **verified live** (AWS creds + S3 staging — scalar via start→poll→return, CTAS table read with `NULL`→`null`, `InvalidRequestException`→`[validation]`, and the staging-dir guard → `[authentication]`) |
| `aws-geo-compute` **Batch** | P96–P99 | ✅ **verified live** (Fargate queue + job definition — submit→`job_id`, poll `running`→`succeeded` with synthesized `output_s3_uri`, EMR-Sedona target → `[upstream]` refusal, and the missing-`AWS_BATCH_JOB_QUEUE` guard → `[upstream]`) |

**Optional, not part of acceptance:**

| Item | Status |
|------|--------|
| PDAL backend (Section 3) | ✅ **verified live** (native PDAL installed; `.copc.laz` round-trip test passes) |
| `gdal-mcp` companion (Section 4) | ✅ **verified live** (third-party; connects via `uvx` and serves `raster_info`/`vector_buffer`) |

### What you still have not tested

1. **Maxar / Planet order success** (P76–P77, P79) — blocked by account
   entitlements, not code; would need a Sentinel Hub Maxar subscription / a Planet
   plan with download access. **This is the only remaining unexercised path**,
   and it is not a connector-code gap.

Every credentialed engine across the pack is now verified live. The warehouse
engines (BigQuery, Redshift, Snowflake, Databricks), the `geo-query` engines
(DuckDB + Athena), the `aws-geo-compute` AWS Batch backend, and Planet search
have all been exercised against real accounts; Phases 1–9 need no accounts. The
only remaining gaps are the two **entitlement-blocked** imagery paths above
(the code reaches the provider; the account isn't licensed) plus the two
optional native/third-party items — none of which are connector-code gaps.


### Phase 8 — DEM-zonal + GeoAI read-then-embed (new tools)

New tools added on existing servers (`geo-terrain.dem_zonal`,
`geo-foundation-models.embed_asset` / `detect_change_from_assets` /
`available_embedding_periods`). Reconnect the affected servers. The COG reads
issue real HTTP range requests; embedding scores under the default backend are a
deterministic stand-in (see the caveat), not calibrated measurements.

| # | Prompt | Exercises | Expect |
|---|--------|-----------|--------|
| P50 | "Per-zone mean/min/max **elevation** from the Copernicus GLO-30 DEM `s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N37_00_W120_00_DEM/Copernicus_DSM_COG_10_N37_00_W120_00_DEM.tif` over an EPSG:4326 polygon inside that 1°×1° tile (e.g. a small box around -119.55, 37.75)." | `geo-terrain` `dem_zonal` (elevation) | One zone with `min`/`max`/`mean`/`count` in metres; Sierra terrain gives hundreds–thousands of m, `no_data: false` |
| P51 | "Per-zone mean **slope** over the same DEM + polygon, passing `pixel_size_m` for the ~30 m cell (GLO-30 is degrees)." | `geo-terrain` `dem_zonal` (slope) | Per-zone slope stats in degrees (0–90) |
| P52 | "Embed the Sentinel-2 B04 COG `https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/11/S/KV/2026/6/S2B_11SKV_20260614_0_L2A/B04.tif` over EPSG:32611 bbox 247089,3921520,247560,3922075 with model Clay." | `geo-foundation-models` `embed_asset` | An `EmbeddingResult`: model `Clay`, `structure_only: false`, `backend: deterministic-local`, vector length = model dimension |
| P53 | "Detect change between the June-2024 and June-2026 B04 COGs of tile 11SKV over that same bbox with model Clay." | `geo-foundation-models` `detect_change_from_assets` | An `AssetChangeResult` with `change` in [0,1] and a `caveat` that the stand-in score is not calibrated |
| P54 | "List the available embedding periods." | `geo-foundation-models` `available_embedding_periods` | The published Clay v1.5 months (e.g. `2024-06`, `2025-06`) — deterministic, no network |

**Guard checks**

| # | Prompt | Expect |
|---|--------|--------|
| P55 | "Run dem_zonal with measure `curvature`." | A **validation error** naming `measure` (use elevation/slope/aspect) |
| P56 | "embed_asset with a window_bbox far outside the asset." | A **validation error**: the window does not overlap the asset |

> **Notes.** `dem_zonal` takes either `dem_href` (a specific COG, as in P50) or a
> named `dem_source` (e.g. `dem_source="glo30"` for Copernicus GLO-30, whose
> overlapping tiles are resolved and mosaicked automatically — handy when a zone
> straddles a 1° tile boundary). Zones must be in the DEM's CRS (GLO-30 is
> EPSG:4326); `slope`/`aspect` need ground cell size, so pass `pixel_size_m` for a
> degrees
> DEM. `embed_asset`/`detect_change_from_assets` read the COG window server-side
> so you pass an href + bbox instead of inline pixels. Change scores are only
> calibrated with a real-weight backend — the default deterministic stand-in
> scores any two differing windows ~0.5 and sets a `caveat` saying so.

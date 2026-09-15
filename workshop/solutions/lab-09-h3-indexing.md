# Lab 09 — H3 spatial indexing and heatmaps (solution)

Server: `geo-index` · Starter data: `exercises/06-h3-indexing/sample_points.csv` (10,000 points clustered around downtown, City Park, the airport, Washington Park, and Central Park)

H3 cell IDs are deterministic; counts depend on the point set.

## Exercise 1 — Index a single point

**Prompt:** *"Convert the coordinate (39.7392, -104.9903) — Denver's City and County Building — to an H3 cell at resolution 8. What does the cell ID look like? Get all 6 neighbors of that cell."*

**Tool:** `geo-index` → point-to-H3 (res 8) plus k-ring/neighbor lookup.

**Expected output:** a 15-character hex cell ID like `89283082803ffff` and its 6 equidistant neighbor IDs. At resolution 8 each cell is ~460 m across.

## Exercise 2 — Aggregate points into a heatmap

**Prompt:** *"I have 10,000 GPS points from vehicle tracking data across Denver. Aggregate them into H3 cells at resolution 8 and count events per cell. Which cells have the highest density?"*

**Tool:** `geo-index` → aggregate `sample_points.csv` into res-8 cells with per-cell counts.

**Expected output:** 10,000 points collapse to roughly a few hundred res-8 cells with counts. The densest cells sit over the downtown cluster; City Park, Washington Park, Central Park, and the airport show secondary peaks. Aggregation turns unrenderable point spray into a compact, analyzable grid.

## Exercise 3 — NDVI heatmap

**Prompt:** *"Take the NDVI results from the Denver area and aggregate them into an H3 resolution 9 heatmap. Calculate the mean NDVI per H3 cell. Which hexagons represent the greenest areas?"*

**Tool:** `geo-index` → mean-NDVI aggregation into res-9 (~174 m) cells.

**Expected output:** each hexagon carries a `mean_ndvi`; the greenest cells fall over City Park and Washington Park. Uniform-area cells make the comparison fair regardless of neighborhood shape.

## Exercise 4 — Multi-resolution analysis

**Prompt:** *"Show me the same data at resolution 7 (city-wide), resolution 9 (neighborhood), and resolution 11 (block-level). How does the pattern change at different scales?"*

**Tool:** `geo-index` → aggregate at res 7 (~1.2 km), 9 (~174 m), 11 (~25 m).

**Expected output:** coarse res-7 cells give a smooth city-wide picture (a park is one green hex); res-9 resolves neighborhood structure; res-11 exposes block-level variation inside each park. The takeaway: patterns are scale-dependent — check multiple resolutions.

## Exercise 5 — Neighbor analysis

**Prompt:** *"For the hottest H3 cell (highest value), get all cells within k-ring distance 2. Are the hot spots clustered or isolated?"*

**Tool:** `geo-index` → k-ring (k=2) around the peak cell.

**Expected output:** the k=2 ring returns up to 19 cells (center + 18). Around the downtown peak the surrounding cells also carry high counts, so the hot spot reads as a cluster rather than an isolated spike.

## Checkpoint answers

- Indexed coordinates to H3 cells and retrieved neighbors ✅
- Aggregated point data into an H3 heatmap (10,000 pts → a few hundred cells) ✅
- Combined NDVI results with H3 indexing ✅
- Understood how resolution choice affects the analysis ✅

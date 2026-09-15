# Labs 13–15 — Capstone: vegetation monitoring dashboard (solution)

Servers: `geo-stac`, `geo-vector`, `geo-raster`, `geo-index`, `geo-foundation-models`, `geo-embedding-search`, `geo-query` · Starter data: `exercises/09-capstone-app/`

The capstone spans three labs: design the app with Kiro specs (13), build the ETL pipeline (14), then run and extend it (15). Foundation-model outputs are deterministic stand-ins; scene counts and NDVI values vary with live data.

---

# Lab 13 — Spec-driven development (AI-DLC)

## Exercise 1 — Start a spec session

**Prompt:** *"Build a vegetation monitoring dashboard for Denver that shows NDVI trends over time, displays an H3 heatmap of vegetation health, and highlights areas with significant vegetation loss using change detection."*

**Tool:** Kiro spec session (Requirements → Design → Tasks → Implementation).

**Expected output:** Kiro opens a spec and responds with clarifying requirements questions rather than jumping to code.

## Exercise 2 — Answer requirements questions

**Prompt:** answer Kiro's questions — AOI `[-105.1, 39.6, -104.8, 39.9]`, Jan 2024→present monthly, interactive web map, alert on NDVI drop > 0.2 in one month, H3 resolution 9.

**Expected output:** a generated `requirements.md` with REQ-1 (AOI bbox), REQ-2 (monthly temporal coverage), REQ-3 (res-9 H3 heatmap), REQ-4 (change alerts on >0.2 drop), REQ-5 (interactive map + time slider).

## Exercise 3 — Review the design

**Prompt:** *"Review the design. Does the data flow make sense? Are the right MCP tools mapped to each step?"*

**Expected output:** a `design.md` mapping steps to Power Pack tools — `geo-stac` stac_search → `geo-raster` band_math → `geo-index` index_cell → `geo-raster` zonal_stats → `geo-foundation-models` change_detect → `geo-embedding-search` store_embedding. The design is the implementation backend.

## Exercise 4 — Review the task plan

**Prompt:** *"Approve the task plan and begin implementation."*

**Expected output:** a `tasks.md` with 7 tasks (config, STAC search, NDVI pipeline, H3 aggregation, change detection, dashboard, wiring). Tasks 2 and 3 run in parallel; Task 4 depends on 3; Task 5 depends on 4.

## Exercise 5 — Watch code generation

**Expected output:** Kiro implements tasks one at a time — Python that calls the MCP tools, a project structure (config, pipeline, dashboard), sensible file organization — pausing for your review/approval at each task. You stay in control and can redirect at any phase.

---

# Lab 14 — ETL pipeline

## Exercise 1 — Extract: gather source data

**Prompt:** *"For the vegetation monitoring app, search for all available Sentinel-2 scenes over Denver from January 2024 to present with less than 30% cloud cover. List the scenes by month."*

**Tools:** `geo-stac` → `stac_search`; then `geo-vector` for building footprints and park boundaries.

**Expected output:** a per-month scene inventory (a handful of scenes per month after cloud filtering) plus the vector "zones" (buildings, parks) that later steps summarize over.

## Exercise 2 — Transform: process the data

**Prompt:** *"For each monthly Sentinel-2 scene: 1) Calculate NDVI, 2) Aggregate to H3 resolution 9, 3) Calculate mean NDVI per H3 cell. Store the results as a time series."*

**Tools:** `geo-raster` (NDVI) → `geo-index` (H3 res 9) → `geo-raster` (`zonal_stats`).

**Expected output:** a `(h3_cell_id, month, mean_ndvi)` time-series table — the app's core dataset.

## Exercise 3 — Transform: detect changes

**Prompt:** *"Compare consecutive months and flag any H3 cell where NDVI dropped more than 0.2. Tag these as 'vegetation_loss_alert'. Include the change magnitude and month."*

**Tools:** `geo-foundation-models` → `change_detect` (or NDVI differencing).

**Expected output:** alert rows where the month-over-month NDVI drop exceeds 0.2 (e.g. 0.65 → 0.40 = 0.25 drop → alert; 0.65 → 0.60 = normal), each with magnitude and month.

## Exercise 4 — Load: populate your app

**Prompt:** *"Load the processed data into DuckDB using geo-query. Create tables for: 1) ndvi_monthly (h3_cell, month, mean_ndvi), 2) change_alerts (h3_cell, month, magnitude, alert_type), 3) zone_summary (zone_name, avg_ndvi, trend_direction)"*

**Tool:** `geo-query` → DuckDB spatial SQL.

**Expected output:** three populated tables (`ndvi_monthly`, `change_alerts`, `zone_summary`) queryable with SQL.

## Exercise 5 — Export for dashboard

**Prompt:** *"Export the H3 heatmap data as GeoJSON for the web dashboard. Each H3 cell should be a polygon with properties: mean_ndvi, month, alert_status. Also export the change alerts as a separate GeoJSON layer."*

**Tool:** `geo-query` → GeoJSON export.

**Expected output:** `ndvi_heatmap.geojson` (H3 polygons with NDVI/month/alert_status), `alerts.geojson` (loss cells), and `time_series.json` (monthly stats for the slider).

## Exercise 6 — Verify the pipeline end-to-end

**Prompt:** *"Run the complete ETL pipeline from scratch: search → process → index → store → export. Verify that the output files contain valid data and the record counts are reasonable."*

**Expected output:** validation passes — STAC returned scenes across multiple months, NDVI values sit in `[-1, +1]`, H3 cells cover the Denver metro AOI, some change alerts exist, and the GeoJSON files are valid polygon layers.

---

# Lab 15 — Run and explore the application

## Exercise 1 — Start the dashboard

**Prompt:** *"Run the vegetation monitoring dashboard. It should serve on port 8080 so I can access it through the browser."*

**Tool:** run the generated app (e.g. `python app.py --port 8080`) from `exercises/09-capstone-app/`.

**Expected output:** the dashboard serves on `http://<ec2-public-ip>:8080` with an interactive Denver map, H3 cells colored by NDVI (green healthy → brown sparse), a monthly time slider, and alert markers on loss cells.

## Exercise 2 — Explore the map

**Expected output:** panning/zooming shows the greenest cells over parks and foothills (as expected); clicking a cell shows its monthly NDVI time series; the slider reveals winter→spring green-up; alerts cluster in specific areas rather than scattering randomly.

## Exercise 3 — Extend with natural language

**Prompt:** *"Add a feature to the dashboard that shows the top 10 H3 cells with the highest vegetation loss this year. Display them as a ranked sidebar list with the cell location and loss magnitude."* (and) *"Add a layer toggle that shows building footprints underneath the H3 grid…"*

**Expected output:** Kiro edits the running app to add a ranked top-10 loss sidebar and a building-footprint toggle layer — natural language extends a live application.

## Exercise 4 — Use the app for new analysis

**Prompt:** *"The alerts are concentrated in the western suburbs. Use the Power Pack to investigate: search for recent satellite imagery over that specific area and generate an embedding. Search for similar areas globally — is this a widespread pattern or localized?"*

**Tools:** `geo-stac` (imagery) → `geo-foundation-models` `embed_tile` → `geo-embedding-search` `search_similar`.

**Expected output:** the app reveals a pattern, the Power Pack investigates it, and similarity search shows whether the western-suburb signal is widespread or localized — the analysis → investigation → insight feedback loop.

## Exercise 5 — ETL update: add new data

**Prompt:** *"A new month of Sentinel-2 data is available. Run the ETL pipeline for the latest month only and update the dashboard data. The app should show the new month in the time slider without reprocessing everything."*

**Tools:** `geo-stac` (new month) → `geo-raster`/`geo-index` → `geo-query` (append + re-export).

**Expected output:** an incremental run that appends the new month to the DuckDB tables and re-exports GeoJSON; the slider gains the new month with no full reprocessing.

## Exercise 6 — Bonus: add steering rules

**Prompt:** *"Create a Kiro steering rule that ensures anyone modifying the ETL pipeline always validates CRS before spatial operations, always checks NDVI values are in range [-1, +1], and always records provenance for reproducibility."*

**Expected output:** a `.kiro/steering/geo-pipeline-standards.md` file that activates when pipeline code is edited, encoding the CRS-validation, NDVI-range, and provenance rules.

---

## Checkpoint answers

**Lab 13**
- Started a spec session and defined requirements ✅
- Reviewed and approved a design document ✅
- Understood the task dependency structure ✅
- Code generation underway ✅

**Lab 14**
- Extracted source data (STAC + vector) ✅
- Transformed it (NDVI → H3 → time series) ✅
- Loaded into DuckDB and exported GeoJSON ✅
- Verified the pipeline end-to-end ✅

**Lab 15**
- Dashboard running and accessible on port 8080 ✅
- Interacted with the map (cells, time slider) ✅
- Extended the app with a new feature ✅
- Ran an incremental ETL update ✅
- Created a steering rule for pipeline quality ✅

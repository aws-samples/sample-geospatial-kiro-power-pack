# Workshop Solutions

Reference walkthroughs for each lab. These are **completed answers** for facilitators
and for participants who want to check their work — not code to run. Because the
workshop is driven by natural-language prompts to Kiro, each solution shows:

- the **prompt** to use,
- which **Power Pack MCP tool(s)** Kiro invokes,
- a **representative expected output** (values vary slightly by live data and date).

These are pre-downloaded to the EC2 instance at `C:\geoai-workshop\solutions\`.

## Contents

| File | Lab | Topic |
|------|-----|-------|
| `lab-03-stac-search.md` | 03 | STAC catalog search |
| `lab-04-vector-query.md` | 04 | Vector feature queries |
| `lab-05-federated-search.md` | 05 | Multi-catalog / federated search |
| `lab-06-crs-transform.md` | 06 | CRS transforms and validation |
| `lab-07-band-math.md` | 07 | NDVI / NDWI / NDBI band math |
| `lab-08-zonal-stats.md` | 08 | Zonal statistics and spatial joins |
| `lab-09-h3-indexing.md` | 09 | H3 indexing and heatmaps |
| `lab-10-embeddings.md` | 10 | Satellite tile embeddings |
| `lab-11-change-detection.md` | 11 | Temporal change detection |
| `lab-12-similarity-search.md` | 12 | Embedding similarity search |
| `lab-13-15-capstone.md` | 13-15 | Spec-driven app, ETL, and launch |

## A note on values

Where a lab hits live data (STAC APIs, public S3 COGs), exact scene IDs, cloud
cover, and pixel values will differ from run to run. The solutions give the
**shape** of a correct answer and representative magnitudes so you can tell a
right result from a wrong one. Foundation-model tools use deterministic stand-ins,
so embedding dimensionality and structure are stable even though the vectors are
not real Clay inference.

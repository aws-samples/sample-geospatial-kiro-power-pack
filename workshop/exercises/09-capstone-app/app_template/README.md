# Denver Vegetation Monitor — capstone starter

A minimal, dependency-free web dashboard (Leaflet) that visualizes the H3 NDVI
heatmap and vegetation-loss alerts produced by the ETL lab (Lab 14). Use it as
the starting point for the capstone (Labs 13-15) and extend it with Kiro.

## Run it

The workshop instance exposes port 8080. Serve this folder with any static
server and open it in the instance browser:

```powershell
cd C:\geoai-workshop\exercises\09-capstone-app\app_template
python -m http.server 8080
```

Then browse to `http://localhost:8080`. The map centers on Denver and loads the
two GeoJSON layers.

## Data

By default the app reads the sample data in `../sample_data/`:

- `ndvi_heatmap.geojson` — H3 cells colored by mean NDVI (green = healthy,
  red = sparse/urban, purple = alert)
- `alerts.geojson` — cells flagged for vegetation loss

When you run the ETL lab, replace these with the files you export so the
dashboard shows your own results. The data paths are set at the top of the
`<script>` block in `index.html` (the `DATA` object).

## Extend it (Lab 15)

Ideas to try with Kiro:

- Add a month slider driven by `../sample_data/time_series.json`
- Add a layer toggle for heatmap vs. alerts
- Color alerts by magnitude
- Add a legend entry for the alert count per month

Ask Kiro in natural language (for example: "Add a time slider to index.html that
filters the heatmap features by the `month` property"), and it will edit the file
for you.

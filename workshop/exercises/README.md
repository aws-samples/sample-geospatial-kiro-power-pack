# Workshop Exercises

These directories contain starter files for each lab exercise. They are pre-downloaded to the EC2 instance at `C:\geoai-workshop\exercises\`.

## Structure

```
exercises/
├── 01-stac-search/          # Lab 03: STAC catalog search exercises
│   └── search_config.json   # Pre-configured catalog endpoints
├── 02-vector-query/         # Lab 04: Vector data query exercises
│   └── denver_aoi.geojson   # Denver area-of-interest boundary
├── 03-crs-transform/        # Lab 06: CRS transformation exercises
│   └── sample_points.geojson
├── 04-band-math/            # Lab 07: NDVI/NDWI calculation exercises
│   └── scene_urls.json      # Pre-selected scene URLs for the exercises
├── 05-zonal-stats/          # Lab 08: Zonal statistics exercises
│   └── denver_neighborhoods.geojson
├── 06-h3-indexing/          # Lab 09: H3 hexagonal indexing exercises
│   └── sample_points.csv    # 10,000 sample GPS points
├── 07-embeddings/           # Lab 10: Embedding generation exercises
├── 08-change-detection/     # Lab 11: Change detection exercises
│   └── baseline_tiles.json  # Pre-selected tile locations
├── 07-embeddings/           # Lab 10: Embedding generation exercises
│   └── tile_locations.json  # Landscape tile bboxes for embedding comparison
└── 09-capstone-app/         # Labs 13-15: Capstone application
    ├── app_template/        # Starter Leaflet dashboard (index.html + README)
    └── sample_data/         # Small sample dataset (ndvi_heatmap, alerts, time_series)
```

## Solutions

Reference walkthroughs for every lab live in the sibling `source/solutions/`
directory (one markdown per lab, plus a README index). They show the prompt,
the Power Pack tool(s) Kiro invokes, and a representative expected output. On the
instance they are synced to `C:\geoai-workshop\solutions\`.

## Deploying exercises and solutions to the instance

The CloudFormation bootstrap syncs both trees from the workshop S3 bucket:

```
s3://<WorkshopDataBucket>/exercises/  -> C:\geoai-workshop\exercises\
s3://<WorkshopDataBucket>/solutions/  -> C:\geoai-workshop\solutions\
```

Before an event, upload these repo folders to that bucket:

```bash
aws s3 sync source/exercises/  s3://<WorkshopDataBucket>/exercises/
aws s3 sync source/solutions/  s3://<WorkshopDataBucket>/solutions/
```

## Data Sources

Exercise data is sourced from:
- Sentinel-2 L2A (ESA Copernicus, open license)
- Landsat Collection 2 (USGS, public domain)
- OpenStreetMap (ODbL)
- Overture Maps Foundation (ODbL)

---
name: geoai-embedding
description: "GeoAI embedding workflow: tile -> model -> embedding -> index."
activation:
  - "geo-foundation-models context"
  - "geo-embedding-search context"
---

# GeoAI Embedding

Encoded best practice for the Geospatial Power Pack (Pillar C): turn imagery into
searchable, comparable vectors with a repeatable pipeline —
**tile → model → embedding → index**. Getting the tiling and model consistency
right is what makes the resulting embeddings usable for similarity search and
change detection.

## Activation context

This skill applies in `geo-foundation-models` and `geo-embedding-search` context —
any code that embeds tiles with a geospatial foundation model (Clay,
Prithvi-EO-2.0, SatCLIP, SAMGeo) or indexes/searches those embeddings (LanceDB,
OpenSearch).

## The pipeline

1. **Tile** — cut the source imagery into fixed-size tiles that match the model's
   expected input (size, band order, bit depth, normalization). Keep each tile's
   **CRS, footprint geometry, datetime, and source** as metadata — an embedding
   without spatial/temporal provenance is not searchable.
2. **Model** — run the chosen foundation model to produce an embedding vector per
   tile. Record the **model identity and version** alongside every embedding.
3. **Embedding** — collect the vectors with their tile metadata.
4. **Index** — write vectors + metadata into a vector store (`geo-embedding-search`)
   for similarity / semantic search.

## Best practices

- **One model per index/comparison.** Embeddings from different models (or
  different versions/checkpoints) are **not comparable**. Never mix them in the
  same index or compare across them. Stamp `model` + `version` on each vector.
- **Consistent preprocessing.** Use the same tile size, band selection/order,
  resampling, and normalization the model was trained with. Inconsistent
  preprocessing silently degrades embedding quality.
- **Consistent dimensionality.** Every vector in an index must have the model's
  fixed embedding dimension. A **mismatched embedding dimension is a validation
  error**, not something to pad or truncate (Property 4 / Requirement 9.9).
- **Keep provenance.** Carry source asset, CRS, footprint, datetime, model, and
  version through to the index so any hit can be traced back to real ground.
- **Reuse tiling discipline.** Follow the `tiling` and `cloud-optimized-formats`
  skills — embed from COGs via windowed reads rather than downloading full scenes.

## Change detection

For embedding-based change detection: **tile two dates → embed both with the
same model → compare → threshold/report.**

- Tile the two acquisitions on the **same grid/footprint** so tiles correspond
  spatially.
- Embed both with the **identical model and version**.
- Compare corresponding tiles (e.g. embedding distance) and apply a documented
  threshold to flag change. Report the change footprint with provenance.

## Search behavior (verified by property test)

`geo-embedding-search` similarity results are **ordered by descending similarity**
and bounded to **at most K** results (Requirement 9.4). When wiring search:

- Don't re-sort or truncate in a way that breaks the descending-similarity order.
- Respect the K bound.
- Reject **out-of-range query length** and **mismatched embedding
  dimensionality** with an `Error_Taxonomy` validation error and no partial
  output (Property 4 / Requirements 9.7, 9.9, 15.5).

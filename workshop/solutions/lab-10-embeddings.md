# Lab 10 — Generate satellite embeddings (solution)

Servers: `geo-foundation-models` (`embed_tile`) / `geo-embedding-search` (`store_embedding`) · Starter data: `exercises/07-embeddings/tile_locations.json`

The embedding backend is a deterministic stand-in (`clay-v1.5-standin`), so vectors are reproducible; similarity magnitudes below are the expected pattern.

## Exercise 1 — Generate your first embedding

**Prompt:** *"Generate a Clay v1.5 embedding for a Sentinel-2 tile over downtown Denver (-105.0, 39.74, -104.98, 39.76) using bands B02, B03, B04, and B08. What dimensionality is the output?"*

**Tool:** `geo-foundation-models` → `embed_tile`.

**Expected output:** a **1024-dimensional** vector, `backend: clay-v1.5-standin`, plus provenance metadata. The stand-in matches the real model's shape and dtypes without GPU cost — swapping to `clay-v1.5` keeps the same API.

## Exercise 2 — Compare two locations

**Prompt:** *"Generate embeddings for two tiles: 1) Downtown Denver (urban), 2) Rocky Mountain National Park (forest). Calculate the cosine similarity between them. Are they similar or different?"*

**Tool:** `geo-foundation-models` → `embed_tile` (×2) then cosine similarity.

**Expected output:** **low similarity (< 0.5)** — the model separates urban from forest landscapes. Because the backend is deterministic, re-running gives the same value.

## Exercise 3 — Compare similar areas

**Prompt:** *"Generate embeddings for: 1) A suburban area in Denver (Lakewood), 2) A similar suburban area in Phoenix. How similar are their embeddings compared to the urban-vs-forest comparison?"*

**Tool:** `geo-foundation-models` → `embed_tile` (×2) then cosine similarity.

**Expected output:** **higher similarity (> 0.7)** than the urban-vs-forest pair — the model recognizes structural similarity across two different cities.

## Exercise 4 — Store embeddings for search

**Prompt:** *"Generate embeddings for 5 different landscape types around Denver (downtown, suburban, park, agricultural, mountainous) and store them in the embedding search index with metadata labels."*

**Tools:** `geo-foundation-models` → `embed_tile`, then `geo-embedding-search` → `store_embedding(vector, metadata)`.

**Expected output:** five 1024-d vectors stored with `label` metadata (downtown, suburban, park, agricultural, mountainous). This seeds the searchable index that Lab 12 queries.

## Exercise 5 — Examine provenance

**Prompt:** *"Show me the provenance metadata from the last embedding generation. What backend was used? What input hash was recorded?"*

**Tool:** `geo-foundation-models` → provenance from the last `embed_tile` call.

**Expected output:** `backend: clay-v1.5-standin`, `provenance: deterministic`, a SHA-256 `input_hash`, and a `model_version`. Same input → same hash → same vector, which is what makes results reproducible.

## Checkpoint answers

- Generated an embedding and understand its structure (1024-d vector) ✅
- Compared embeddings across landscape types (urban vs. forest < 0.5; suburb vs. suburb > 0.7) ✅
- Stored embeddings in the search index ✅
- Examined provenance metadata ✅

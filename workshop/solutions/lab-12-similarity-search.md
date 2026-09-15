# Lab 12 — Embedding search (find similar areas) (solution)

Server: `geo-embedding-search` (`search_similar`) · Builds on the embedding index seeded in Lab 10

The backend is a deterministic stand-in, so rankings are reproducible; the patterns below are the expected outcome.

## Exercise 1 — Search for similar landscapes

**Prompt:** *"Generate an embedding for a park area in Denver (City Park). Then search for the most similar areas in the embedding index. What does the system find?"*

**Tools:** `geo-foundation-models` → `embed_tile`, then `geo-embedding-search` → `search_similar(vector, k=5)`.

**Expected output:** the top matches are other green/vegetated tiles (parks, golf courses) stored in Lab 10; urban and mountain tiles rank lower. Search matches by landscape meaning, not keywords.

## Exercise 2 — Find similar urban patterns

**Prompt:** *"Embed a tile from Denver's downtown core (mixed high-rise and commercial). Search for similar urban patterns. Do the results make sense?"*

**Tools:** `geo-foundation-models` → `embed_tile`, then `geo-embedding-search` → `search_similar`.

**Expected output:** dense urban tiles rank highest, suburban tiles moderate, rural/natural tiles poor — a sensible gradient that confirms the embedding captures built-environment density.

## Exercise 3 — Anomaly detection

**Prompt:** *"Generate embeddings for a grid of H3 cells across Denver at resolution 8. For each cell, find its nearest neighbors. Which cells are most UNLIKE their neighbors (highest distance to nearest match)?"*

**Tools:** `geo-foundation-models` → `embed_tile` per cell, then `geo-embedding-search` → `search_similar` per cell (nearest-neighbor distance).

**Expected output:** the outlier cells are the ones whose nearest match is still far in embedding space — e.g. a park surrounded by dense urban, a construction site, or a lone industrial parcel. High nearest-neighbor distance = anomaly.

## Exercise 4 — Temporal anomaly

**Prompt:** *"Take the baseline forest embedding from Lab 11. Store it. Then take the more recent embedding. Search the index — does the recent embedding still match the baseline? If not, something changed."*

**Tools:** `geo-embedding-search` → `store_embedding` (baseline) then `search_similar` (recent).

**Expected output:** if the landscape is unchanged, the recent embedding returns the baseline as its top match with high similarity; if similarity has dropped, the baseline no longer tops the results — the same-tile-different-time signal that change occurred.

## Exercise 5 — Build a reference library

**Prompt:** *"Create a reference library of 5 land-use types with their embeddings: 1) Dense urban, 2) Suburban residential, 3) Park/green space, 4) Agriculture, 5) Mountain/forest. Store each with descriptive metadata. Then classify a new tile by finding its closest reference."*

**Tools:** `geo-foundation-models` → `embed_tile` (×5 + query), `geo-embedding-search` → `store_embedding` then `search_similar(k=1)`.

**Expected output:** five labeled reference vectors, and a new tile classified by its nearest reference (e.g. a suburban query returns "Suburban residential"). This is nearest-neighbor classification with no training — just labeled examples.

## Checkpoint answers

- Searched for similar areas using embedding vectors ✅
- Performed anomaly detection (outliers in embedding space) ✅
- Built a reference library for classification ✅
- Understood how temporal + spatial search combine ✅

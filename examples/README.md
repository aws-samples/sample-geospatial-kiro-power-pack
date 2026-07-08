# Examples

Runnable recipes that extend the Geospatial Power Pack. These live outside the
shipped packages so heavy, optional dependencies (e.g. torch, model weights)
never land in the pack's default install.

| Example | What it shows |
|---------|---------------|
| [`custom-embedding-backend/`](./custom-embedding-backend/) | **Start here.** Plug *any* real model into the `geo-foundation-models` backend in three steps, with a copy-paste `embed(payload)` template and a one-command self-test. |
| [`clay-local-backend/`](./clay-local-backend/) | A complete real-weight adapter for **Clay v1.5** (via `claymodel`), plus a smoke script driving `detect_change_from_assets` / `change_map`. Verified end-to-end. |
| [`prithvi-local-backend/`](./prithvi-local-backend/) | A real-weight adapter for **Prithvi-EO-2.0** (NASA/IBM, via `terratorch`). Verified end-to-end. |

Both real-model adapters (Clay, Prithvi) run through the *same* `embed(payload)`
contract and pass the *same* self-test with no changes to the pack's core code —
that's the extensibility point.

## Why a real backend?

The default embedding backend is a deterministic **stand-in**: honest but
uncalibrated (any two differing tiles score ~0.5). Wiring a real model makes
`embed_tile`, `embed_asset`, `detect_change`, `detect_change_from_assets`, and
`change_map` produce calibrated results. Configure one of:

- `GEO_FM_EMBED_LOCAL_CALLABLE=module:fn` — in-process, for small workflows.
- `GEO_FM_EMBED_ENDPOINT_URL=https://...` — a generic HTTPS inference endpoint.
- `GEO_FM_SAGEMAKER_ENDPOINT=name` — an AWS SageMaker endpoint (offload/scale).

Declare a custom model's embedding width with
`GEO_FM_EXTRA_MODELS="Name:Dimension,..."`, then validate with the live
self-test (`RUN_LIVE_EMBED=1 ... pytest -k live_backend`).

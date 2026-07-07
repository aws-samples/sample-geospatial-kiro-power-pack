# Bring your own embedding model

The `geo-foundation-models` server embeds tiles through a pluggable backend. The
default is an honest **deterministic stand-in** (change scores are ~0.5 noise);
plug in a real model to get *calibrated* `embed_*` / `detect_change` /
`change_map` results. This works for **any** model — Clay, Prithvi, SatCLIP, a
proprietary encoder, whatever — via the same three steps and one contract.

## The contract

Write a function `embed(payload) -> vector`:

```python
def embed(payload: dict) -> list[float]:
    # payload: {model, dimension, width, height, bands, format, dtype, data}
    # data is a flat pixel-major band-interleaved list: data[(row*W + col)*B + b]
    ...
    return vector   # length == payload["dimension"]
```

`template_embed.py` in this folder is a working, dependency-light (numpy-only)
implementation you can run today, with the model call marked `REPLACE FROM
HERE`. Copy it and drop in your model.

## Three steps

```bash
# 1. Point the server at your callable (module:function; must be importable).
export GEO_FM_EMBED_LOCAL_CALLABLE="template_embed:embed"
export PYTHONPATH="$PWD"           # so template_embed resolves

# 2. Declare your model's embedding width so the tool's dimension matches
#    what your model returns (embed_tile rejects a mismatch).
export GEO_FM_EXTRA_MODELS="MyModel:512"       # Name:Dimension, comma-separated

# 3. Prove it works — the live self-test (no network; embeds synthetic tiles).
RUN_LIVE_EMBED=1 GEO_FM_LIVE_MODEL=MyModel GEO_FM_LIVE_BANDS=4 \
  <repo>/.venv/bin/python -m pytest \
  packages/geo-foundation-models/tests/test_live_backend.py -q
```

A pass means the backend is **calibrated** (not the stand-in), its output
**dimension matches** the declared spec, it is **deterministic** (identical
input → change `0.0`), and it is **content-sensitive** (different input →
non-zero change). Then call the tools with `model="MyModel"`.

## Local vs remote

- **Local, in-process** (this example) — `GEO_FM_EMBED_LOCAL_CALLABLE`. Best for
  small/interactive workflows; the pack ships no weights, you bring the model.
- **Remote endpoint** — `GEO_FM_EMBED_ENDPOINT_URL` (generic HTTPS) or
  `GEO_FM_SAGEMAKER_ENDPOINT` (AWS SageMaker; `geo-foundation-models[remote]`
  extra) for offload/scale. Your endpoint receives the **same payload** and
  returns `{"vector": [...]}`, so one implementation serves both — a local
  function and an HTTP handler are interchangeable.

The self-test above works the same way for a remote backend; just export the
endpoint variables instead of the callable.

## Real examples (both verified end-to-end)

Two very different models plug into the *same* `embed(payload)` contract and pass
the *same* self-test with no core-code changes:

- `../clay-local-backend/` — **Clay v1.5** via `claymodel` (`model="Clay-v1.5"`),
  a datacube encoder API.
- `../prithvi-local-backend/` — **Prithvi-EO-2.0** (NASA/IBM) via `terratorch`
  (`model="Prithvi-EO-2.0"`), a backbone-registry ViT.

Each includes data-choice guidance and faithfulness caveats.

"""Template: adapt ANY model into the geo-foundation-models local backend.

Copy this file, keep the ``embed(payload)`` signature, and replace the marked
body with your model's inference. Wire it with::

    export GEO_FM_EMBED_LOCAL_CALLABLE="template_embed:embed"   # module:function
    export GEO_FM_EXTRA_MODELS="MyModel:512"                    # your model's dim

then call the tools with ``model="MyModel"``.

Out of the box this template is a **real, content-dependent** embedder (per-band
mean/std, numpy only — no heavy deps), so you can run the self-test immediately
(``RUN_LIVE_EMBED=1 ... pytest -k live_backend``) and confirm the whole seam
works *before* plugging in a real model.

Contract
--------
``embed(payload) -> list[float]`` where ``payload`` is::

    {"model", "dimension", "width", "height", "bands", "format", "dtype", "data"}

``data`` is a flat, pixel-major band-interleaved list: ``data[(row*W + col)*B + b]``
(``None`` when the caller passed no pixels). Return a vector of length
``payload["dimension"]`` (a bare list, or any shape
``coerce_embedding_vector`` accepts: ``[[...]]``, ``{"vector": [...]}``, ...).
The returned length must match the dimension you registered for the model.
"""

from __future__ import annotations

from typing import Any, List, Mapping

import numpy as np


def embed(payload: Mapping[str, Any]) -> List[float]:
    """Return an embedding for the tile payload (replace the body with a model)."""
    dim = int(payload["dimension"])
    width = int(payload["width"])
    height = int(payload["height"])
    bands = int(payload["bands"])
    data = payload.get("data")
    if data is None:
        return [0.0] * dim

    # ---- reshape the shared payload to [pixels, bands] ----------------------
    pixels = np.asarray(data, dtype="float64").reshape(height * width, bands)

    # ======================= REPLACE FROM HERE ==============================
    # Your model goes here. Example:
    #   import torch
    #   chips = to_chw_tensor(pixels, bands, height, width)   # [1, B, H, W]
    #   with torch.no_grad():
    #       vec = my_model.encode(chips).reshape(-1).tolist()
    #   return vec
    #
    # Placeholder (real, deterministic, content-dependent): a per-band
    # mean/std descriptor tiled to the requested dimension and L2-normalized.
    descriptor = np.concatenate([pixels.mean(axis=0), pixels.std(axis=0)])
    reps = int(np.ceil(dim / max(1, descriptor.size)))
    vector = np.tile(descriptor, reps)[:dim].astype("float64")
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector = vector / norm
    return [float(v) for v in vector]
    # ======================== REPLACE TO HERE ===============================

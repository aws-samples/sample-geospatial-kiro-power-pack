"""A real Prithvi-EO-2.0 (NASA/IBM) embedding callable for the local backend.

Second worked example (alongside Clay) showing the same `embed(payload) ->
vector` contract adapts a very different model — here via ``terratorch``'s
backbone registry. Kept under ``examples/`` so terratorch + weights never land
in the pack's default install.

Wiring
------
    pip install terratorch
    export GEO_FM_EMBED_LOCAL_CALLABLE="prithvi_embed:embed"
    export PRITHVI_BACKBONE=terratorch_prithvi_eo_v2_300   # or _600, _100_tl, ...
    # call the tools with model="Prithvi-EO-2.0" (1024-d, in the default registry)

Prithvi-EO-2.0 is a 6-band HLS model (Blue, Green, Red, NIR-narrow, SWIR1,
SWIR2). Read those 6 bands, in that order, from a single multi-band COG. The
adapter standardizes per band and resizes to the model's 224×224 input, then
returns the last-layer CLS-token embedding (1024-d).

Note: this is a plumbing-faithful adapter (per-tile standardization, not
Prithvi's official reflectance stats), sufficient to prove the seam is
calibrated, deterministic, and content-sensitive. For production, apply the
model's published normalization and real acquisition metadata.
"""

from __future__ import annotations

import os
import threading
from typing import Any, List, Mapping

_LOCK = threading.Lock()
_STATE: dict = {}

_INPUT_SIZE = int(os.environ.get("PRITHVI_INPUT_SIZE", "224"))
_EXPECTED_BANDS = 6


def _load():
    if "model" in _STATE:
        return _STATE
    with _LOCK:
        if "model" in _STATE:
            return _STATE
        import torch
        from terratorch import BACKBONE_REGISTRY

        name = os.environ.get("PRITHVI_BACKBONE", "terratorch_prithvi_eo_v2_300")
        model = BACKBONE_REGISTRY.build(name, pretrained=True)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        _STATE.update(torch=torch, model=model, device=device, name=name)
        return _STATE


def embed(payload: Mapping[str, Any]) -> List[float]:
    """Return a real Prithvi-EO-2.0 embedding (1024-d) for the tile payload."""
    st = _load()
    torch = st["torch"]

    width = int(payload["width"])
    height = int(payload["height"])
    bands = int(payload["bands"])
    data = payload.get("data")
    if data is None:
        raise ValueError("Prithvi needs pixel data; call with a real COG window")
    if bands != _EXPECTED_BANDS:
        raise ValueError(
            f"Prithvi-EO-2.0 expects {_EXPECTED_BANDS} HLS bands "
            f"(Blue, Green, Red, NIR-narrow, SWIR1, SWIR2), got {bands}. Read "
            f"those 6 bands, in order, from a single multi-band COG."
        )

    np = __import__("numpy")
    arr = np.asarray(data, dtype="float32").reshape(height, width, bands)
    arr = np.transpose(arr, (2, 0, 1))  # -> [B, H, W]

    # Per-band standardization (generic; see module note on official stats).
    mean = arr.mean(axis=(1, 2), keepdims=True)
    std = arr.std(axis=(1, 2), keepdims=True)
    arr = (arr - mean) / np.where(std > 0, std, 1.0)

    x = torch.from_numpy(arr).unsqueeze(0).to(st["device"])  # [1, B, H, W]
    if x.shape[-2:] != (_INPUT_SIZE, _INPUT_SIZE):
        x = torch.nn.functional.interpolate(
            x, size=(_INPUT_SIZE, _INPUT_SIZE), mode="bilinear", align_corners=False
        )

    with torch.no_grad():
        out = st["model"](x)
    feat = out[-1] if isinstance(out, (list, tuple)) else out  # [1, tokens, D]
    if feat.ndim == 3:
        vector = feat[:, 0, :]          # CLS token
    else:
        vector = feat.reshape(1, -1)
    vector = vector.detach().cpu().reshape(-1)
    return [float(v) for v in vector.tolist()]

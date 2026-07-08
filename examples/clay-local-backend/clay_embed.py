"""A real Clay v1.5 embedding callable for the geo-foundation-models LOCAL backend.

This is an **example** (not part of the shipped pack) showing how to plug a real
foundation model into ``LocalCallableBackend`` so ``embed_asset``,
``detect_change_from_assets``, and ``change_map`` return *calibrated* results
instead of the deterministic stand-in. It is intentionally kept under
``examples/`` so the pack's default install stays light — Clay pulls in torch +
the ``claymodel`` package + a multi-GB checkpoint, none of which the pack ships.

Wiring
------
As an in-process callable for the running MCP server::

    export GEO_FM_EMBED_LOCAL_CALLABLE="clay_embed:embed"   # module:function
    export CLAY_CKPT=/path/to/clay-v1.5.ckpt
    export CLAY_METADATA=/path/to/claymodel/configs/metadata.yaml
    export CLAY_SENSOR=sentinel-2-l2a        # or naip, landsat-c2l2-sr, ...

Then call the tools with model ``"Clay-v1.5"`` (1024-d — matches real Clay v1.5,
which is why the registry ships that spec). ``backend`` on results will read
``local:clay_embed.embed`` and ``calibrated`` will be true.

Contract
--------
The pack calls ``embed(payload)`` with the shared tile payload::

    {"model", "dimension", "width", "height", "bands", "format", "dtype", "data"}

``data`` is a flat, **pixel-major band-interleaved** list (``data[(r*W+c)*B + b]``)
— exactly what ``geo_common`` reads. This callable reshapes it to ``[B, H, W]``,
normalizes per band with the sensor's mean/std, resizes to Clay's 256×256 chip,
builds the per-band wavelength tensor, and runs the Clay encoder.

Faithfulness caveats (read before trusting magnitudes)
------------------------------------------------------
* The **band count and order** you read must match the sensor's ``band_order``
  in ``metadata.yaml`` (e.g. Sentinel-2 L2A = 10 bands in a fixed order). Read
  those bands, in that order, from a **single multi-band COG**. Earth Search
  publishes each S2 band as a *separate* single-band COG, so either stack them
  into one multi-band COG first, or test with a natively multi-band product such
  as **NAIP** (a single 4-band R/G/B/NIR COG — set ``CLAY_SENSOR=naip``).
* ``timestamps`` (week/hour) and lat/lon are passed as zeros here; Clay accepts
  that, but location/season conditioning is therefore omitted. For a faithful
  run, extend this callable to pass real values (the pack could forward the
  window bbox + datetime — see the "enrich the payload" follow-up).
"""

from __future__ import annotations

import os
import threading
from typing import Any, List, Mapping

# Heavy deps are imported lazily inside the loader so simply importing this
# module (e.g. for --help) does not require torch to be installed.
_LOCK = threading.Lock()
_STATE: dict = {}


def _config() -> dict:
    return {
        "ckpt": os.environ.get("CLAY_CKPT", "clay-v1.5.ckpt"),
        "metadata": os.environ.get("CLAY_METADATA", "configs/metadata.yaml"),
        "sensor": os.environ.get("CLAY_SENSOR", "sentinel-2-l2a"),
        "chip_size": int(os.environ.get("CLAY_CHIP_SIZE", "256")),
    }


def _load():
    """Load the Clay module + sensor metadata once (thread-safe, cached)."""
    if "model" in _STATE:
        return _STATE
    with _LOCK:
        if "model" in _STATE:
            return _STATE
        import torch
        import yaml
        from claymodel.module import ClayMAEModule

        cfg = _config()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # The v1.5 checkpoint saved metadata_path='configs/metadata.yaml' (a cwd-
        # relative path); override it so loading works from any directory.
        model = ClayMAEModule.load_from_checkpoint(
            cfg["ckpt"], map_location=device, metadata_path=cfg["metadata"]
        )
        model.eval()
        with open(cfg["metadata"], "r") as fh:
            metadata = yaml.safe_load(fh)
        if cfg["sensor"] not in metadata:
            raise KeyError(
                f"sensor {cfg['sensor']!r} not in {cfg['metadata']}; "
                f"available: {sorted(metadata)}"
            )
        # Embeddings, not reconstruction: disable masking for a full,
        # deterministic pass over every patch.
        model.model.encoder.mask_ratio = 0.0
        sensor = metadata[cfg["sensor"]]
        order = sensor["band_order"]
        waves_um = sensor["bands"]["wavelength"]
        mean = sensor["bands"]["mean"]
        std = sensor["bands"]["std"]
        _STATE.update(
            torch=torch,
            module=model,
            encoder=model.model.encoder,
            device=device,
            chip_size=cfg["chip_size"],
            band_order=order,
            # Clay consumes wavelengths in micrometres (metadata is μm) in band order.
            wavelengths=[float(waves_um[b]) for b in order],
            mean=[float(mean[b]) for b in order],
            std=[float(std[b]) for b in order],
            gsd=float(sensor["gsd"]),
            sensor_name=cfg["sensor"],
        )
        return _STATE


def embed(payload: Mapping[str, Any]) -> List[float]:
    """Return a real Clay v1.5 embedding (1024-d) for the pack's tile payload."""
    st = _load()
    torch = st["torch"]

    width = int(payload["width"])
    height = int(payload["height"])
    bands = int(payload["bands"])
    data = payload.get("data")
    if data is None:
        raise ValueError("Clay needs pixel data; call with a real COG window")

    expected = len(st["band_order"])
    if bands != expected:
        raise ValueError(
            f"sensor {st['sensor_name']!r} expects {expected} bands in order "
            f"{st['band_order']}, but the tile has {bands}. Read exactly those "
            f"bands, in that order, from a single multi-band COG."
        )

    np = __import__("numpy")
    # data is pixel-major band-interleaved: data[(r*W + c)*B + b]
    arr = np.asarray(data, dtype="float32").reshape(height, width, bands)
    arr = np.transpose(arr, (2, 0, 1))  # -> [B, H, W]

    mean = np.asarray(st["mean"], dtype="float32").reshape(bands, 1, 1)
    std = np.asarray(st["std"], dtype="float32").reshape(bands, 1, 1)
    arr = (arr - mean) / std

    chips = torch.from_numpy(arr).unsqueeze(0).to(st["device"])  # [1, B, H, W]
    size = st["chip_size"]
    if chips.shape[-2:] != (size, size):
        chips = torch.nn.functional.interpolate(
            chips, size=(size, size), mode="bilinear", align_corners=False
        )

    device = st["device"]
    # Optional spatio-temporal conditioning: use the payload's latlon/acquired
    # when the pack forwards them, else zeros (still valid — omits the context).
    time_vec = _encode_time(payload.get("acquired"), torch, device)
    latlon_vec = _encode_latlon(payload.get("latlon"), torch, device)
    datacube = {
        "platform": st["sensor_name"],
        "pixels": chips.to(torch.float32),                    # [1, C, H, W]
        "time": time_vec,                                     # [B, 4] (week/hour sin-cos)
        "latlon": latlon_vec,                                 # [B, 4] (lat/lon sin-cos)
        "gsd": torch.tensor(st["gsd"], device=device),        # scalar ground sample distance
        "waves": torch.tensor(st["wavelengths"], dtype=torch.float32, device=device),  # [N] μm
    }

    with torch.no_grad():
        encoded, *_ = st["encoder"](datacube)   # [1, 1 + num_patches, D]
    # The class token (index 0) is Clay's tile-level embedding.
    embedding = encoded[:, 0, :].detach().cpu().reshape(-1)
    return [float(v) for v in embedding.tolist()]


def _encode_time(acquired, torch, device):
    """[week, hour] -> sin/cos [1, 4]; zeros when no datetime is supplied."""
    if not acquired:
        return torch.zeros(1, 4, device=device)
    import math as _math
    from datetime import datetime

    try:
        dt = datetime.fromisoformat(str(acquired).replace("Z", "+00:00"))
    except ValueError:
        return torch.zeros(1, 4, device=device)
    week = dt.isocalendar().week / 52.0
    hour = dt.hour / 24.0
    vals = [
        _math.sin(2 * _math.pi * week), _math.cos(2 * _math.pi * week),
        _math.sin(2 * _math.pi * hour), _math.cos(2 * _math.pi * hour),
    ]
    return torch.tensor([vals], dtype=torch.float32, device=device)


def _encode_latlon(latlon, torch, device):
    """[lat, lon] -> sin/cos [1, 4]; zeros when no location is supplied."""
    if not latlon or len(latlon) != 2:
        return torch.zeros(1, 4, device=device)
    import math as _math

    lat, lon = float(latlon[0]), float(latlon[1])
    vals = [
        _math.sin(_math.pi * lat / 90.0), _math.cos(_math.pi * lat / 90.0),
        _math.sin(_math.pi * lon / 180.0), _math.cos(_math.pi * lon / 180.0),
    ]
    return torch.tensor([vals], dtype=torch.float32, device=device)

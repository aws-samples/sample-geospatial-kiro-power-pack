"""Model selection, tile validation, and the ``embed_tile`` core (Req 9.1/9.2/9.7).

This module holds the logic-bearing core of task 7.1, kept free of any MCP/HTTP
plumbing so it is easy to test and reuse:

* :data:`DEFAULT_MODELS` and :class:`ModelRegistry` - the **configurable** set
  of foundation models a request may select from. The registry includes at
  least Clay, Prithvi-EO-2.0, and SatCLIP (Requirement 9.2) and is configurable
  (callers may add or remove models). Each model fixes the embedding
  dimensionality (Requirement 9.1).
* :data:`DEFAULT_SUPPORTED_FORMATS` - the raster formats accepted for embedding
  (Requirement 9.1 / 9.7).
* :func:`validate_tile` - rejects empty, oversized, or unsupported-format tiles
  with an ``Error_Taxonomy`` :class:`~geo_common.errors.ValidationError`
  (Requirement 9.7), performing all checks **before** any embedding work so no
  embedding is produced on bad input.
* :class:`EmbeddingBackend` / :class:`DeterministicLocalBackend` - the pluggable
  embedding computation. The default backend computes a real,
  content-dependent, deterministic embedding of the model's dimensionality
  using a seeded RNG; production backends (Clay, Prithvi, SatCLIP weights) can
  be substituted without changing the validation/selection contract.
* :func:`embed_tile` - selects exactly one model per request from the registry
  (Requirement 9.2), validates the tile (Requirement 9.7), and returns a
  model-determined :class:`~geo_foundation_models.models.EmbeddingResult`
  (Requirement 9.1).
"""

from __future__ import annotations

import hashlib
import struct
from typing import Dict, Iterable, List, Optional

import numpy as np

from geo_common.errors import ValidationError

from geo_foundation_models.models import (
    MAX_TILE_DIMENSION,
    EmbeddingResult,
    ModelSpec,
    RasterTile,
)

__all__ = [
    "DEFAULT_MODELS",
    "DEFAULT_SUPPORTED_FORMATS",
    "ModelRegistry",
    "EmbeddingBackend",
    "DeterministicLocalBackend",
    "coerce_raster_tile",
    "validate_tile",
    "embed_tile",
]

#: The default configurable model set. Dimensions reflect each model's
#: published embedding width; they are configuration, not hard requirements,
#: and may be overridden by constructing a :class:`ModelRegistry` (Req 9.2).
#:
#: Note on "Clay": this 768-d entry matches **Clay v1** and is produced here by
#: the local *deterministic stub* backend (a stand-in until real model weights
#: are wired in). It is intentionally distinct from
#: :func:`geo_foundation_models.clay_embeddings.lookup_embeddings`, which
#: retrieves *real* published **Clay v1.5** (1024-d) embeddings. The two are
#: different versions produced by different backends and live in different
#: vector spaces, so their outputs are **not** comparable (e.g. via
#: ``detect_change``); the differing dimensionality also guards against mixing
#: them by accident.
DEFAULT_MODELS: Dict[str, ModelSpec] = {
    "Clay": ModelSpec(name="Clay", dimension=768),
    "Prithvi-EO-2.0": ModelSpec(name="Prithvi-EO-2.0", dimension=1024),
    "SatCLIP": ModelSpec(name="SatCLIP", dimension=256),
}

#: Raster formats accepted for embedding (compared case-insensitively).
DEFAULT_SUPPORTED_FORMATS = frozenset(
    {"geotiff", "gtiff", "cog", "png", "jpeg", "jpg", "numpy"}
)

#: Source identifier used on errors raised by this server.
_SOURCE = "geo-foundation-models"


class ModelRegistry:
    """The configurable set of foundation models a request may select from.

    Defaults to :data:`DEFAULT_MODELS` (Clay, Prithvi-EO-2.0, SatCLIP -
    Requirement 9.2) and is configurable: pass a custom mapping, or
    :meth:`register` / :meth:`unregister` models at runtime. Looking up an
    unknown model raises a :class:`~geo_common.errors.ValidationError` so a
    request can select **exactly one** known model (Requirement 9.2).
    """

    def __init__(self, models: Optional[Dict[str, ModelSpec]] = None) -> None:
        source = models if models is not None else DEFAULT_MODELS
        # Copy so the registry owns its mapping and the module default is not
        # mutated by later register/unregister calls.
        self._models: Dict[str, ModelSpec] = dict(source)

    def register(self, spec: ModelSpec) -> None:
        """Add or replace a model in the configurable set."""
        self._models[spec.name] = spec

    def unregister(self, name: str) -> None:
        """Remove a model from the configurable set, if present."""
        self._models.pop(name, None)

    def names(self) -> List[str]:
        """The names of all configured models."""
        return list(self._models.keys())

    def __contains__(self, name: object) -> bool:
        return name in self._models

    def get(self, model: str) -> ModelSpec:
        """Return the :class:`ModelSpec` for ``model``.

        Lookup is case-insensitive: ``"clay"`` resolves to the registered
        ``"Clay"`` (the result reports the canonical registered name). Raises
        :class:`~geo_common.errors.ValidationError` (taxonomy ``validation``)
        naming the unknown model and listing the configured choices, so an
        out-of-set selection produces no embedding (Req 9.2).
        """
        if isinstance(model, str):
            # Exact match first, then a case-insensitive fallback so a client
            # that sends "clay"/"satclip" resolves to the canonical spec.
            if model in self._models:
                return self._models[model]
            folded = model.casefold()
            for name, spec in self._models.items():
                if name.casefold() == folded:
                    return spec
        raise ValidationError(
            "unknown foundation model %r; choose exactly one of: %s"
            % (model, ", ".join(sorted(self._models))),
            source=_SOURCE,
            detail={"model": model, "available_models": sorted(self._models)},
        )


def coerce_raster_tile(tile: "RasterTile | dict | Any") -> RasterTile:
    """Return ``tile`` as a :class:`RasterTile`, raising cleanly on bad input.

    The MCP runtime coerces a JSON tile argument into a :class:`RasterTile`
    best-effort; if the client sends a shape that does not validate (for
    example a base64 image string, or a dict whose ``data`` is not a flat list
    of numbers) the raw value is passed through unchanged. This helper gives
    such input a single, well-defined failure mode: a :class:`RasterTile` is
    returned as-is, a mapping is validated into one, and anything else (or a
    mapping that fails validation) raises an ``Error_Taxonomy``
    :class:`~geo_common.errors.ValidationError` describing the expected tile
    shape - so a malformed tile yields a clean validation error rather than an
    opaque ``AttributeError`` and never reaches the embedding backend
    (Requirement 9.7).
    """
    if isinstance(tile, RasterTile):
        return tile

    from pydantic import ValidationError as PydanticValidationError

    if isinstance(tile, dict):
        try:
            return RasterTile.model_validate(tile)
        except PydanticValidationError as exc:
            raise ValidationError(
                "malformed tile: could not be parsed into a RasterTile "
                "(expected fields: width, height, bands, format, optional flat "
                "numeric data, dtype)",
                source=_SOURCE,
                detail={"errors": exc.errors(include_url=False)},
            )
    raise ValidationError(
        "malformed tile: expected a tile object with width, height, bands, "
        "format, and optional flat numeric pixel data, got %s"
        % type(tile).__name__,
        source=_SOURCE,
        detail={"received_type": type(tile).__name__},
    )


def validate_tile(
    tile: RasterTile,
    *,
    supported_formats: Iterable[str] = DEFAULT_SUPPORTED_FORMATS,
    max_dimension: int = MAX_TILE_DIMENSION,
) -> None:
    """Validate a tile for embedding, raising on any failure (Requirement 9.7).

    Checks, in order and **before** any embedding work, that the tile is:

    * in a supported raster ``format`` (case-insensitive);
    * non-empty - positive ``width``, ``height``, and ``bands``, and, when
      ``data`` is inlined, a non-empty, correctly-sized buffer;
    * not oversized - ``width`` and ``height`` each ``<= max_dimension``
      (default 1024).

    On any violation it raises an ``Error_Taxonomy``
    :class:`~geo_common.errors.ValidationError` identifying the failure
    (Requirement 9.7); because it returns ``None`` only on success and is called
    before embedding, a rejected tile yields no embedding.
    """
    tile = coerce_raster_tile(tile)
    supported = {fmt.lower() for fmt in supported_formats}
    if tile.format.strip().lower() not in supported:
        raise ValidationError(
            "unsupported raster format %r; supported formats: %s"
            % (tile.format, ", ".join(sorted(supported))),
            source=_SOURCE,
            detail={"format": tile.format, "supported_formats": sorted(supported)},
        )

    # Empty tile: no pixels or no bands.
    if tile.width <= 0 or tile.height <= 0 or tile.bands <= 0:
        raise ValidationError(
            "empty tile: width, height, and bands must all be positive "
            "(got width=%d, height=%d, bands=%d)"
            % (tile.width, tile.height, tile.bands),
            source=_SOURCE,
            detail={
                "width": tile.width,
                "height": tile.height,
                "bands": tile.bands,
            },
        )

    # Oversized tile: either dimension over the limit.
    if tile.width > max_dimension or tile.height > max_dimension:
        raise ValidationError(
            "oversized tile: %dx%d exceeds the %dx%d maximum"
            % (tile.width, tile.height, max_dimension, max_dimension),
            source=_SOURCE,
            detail={
                "width": tile.width,
                "height": tile.height,
                "max_dimension": max_dimension,
            },
        )

    # Inlined pixel data, when present, must be non-empty and correctly sized.
    if tile.data is not None:
        expected = tile.width * tile.height * tile.bands
        if len(tile.data) == 0:
            raise ValidationError(
                "empty tile: inlined pixel data is empty",
                source=_SOURCE,
                detail={"data_length": 0},
            )
        if len(tile.data) != expected:
            raise ValidationError(
                "malformed tile: inlined pixel data length %d does not match "
                "width * height * bands = %d"
                % (len(tile.data), expected),
                source=_SOURCE,
                detail={"data_length": len(tile.data), "expected_length": expected},
            )


class EmbeddingBackend:
    """Pluggable embedding computation for a validated tile.

    A backend turns a validated :class:`RasterTile` plus a chosen
    :class:`ModelSpec` into a ``dimension``-length embedding vector. The
    validation/selection contract in :func:`embed_tile` is independent of the
    backend, so production model weights (Clay, Prithvi-EO-2.0, SatCLIP) can be
    dropped in by subclassing this interface.
    """

    #: Provenance identifier recorded on every :class:`EmbeddingResult` this
    #: backend produces, so a real model-weight backend (e.g. ``"clay-v1.5"``)
    #: is distinguishable from the deterministic local stand-in.
    backend_id: str = "unknown"

    def embed(self, tile: RasterTile, spec: ModelSpec) -> List[float]:  # pragma: no cover - interface
        raise NotImplementedError


class DeterministicLocalBackend(EmbeddingBackend):
    """A real, content-dependent, deterministic local embedding backend.

    Without GPU model weights this backend still produces a *genuine* function
    of the input: it hashes a canonical encoding of the tile (its dimensions,
    band count, format, dtype, and any inlined pixel data) together with the
    model name, seeds a NumPy PCG64 generator with that hash, draws
    ``spec.dimension`` standard-normal values, and L2-normalizes them. The
    result is therefore:

    * the dimensionality determined by the selected model (Requirement 9.1);
    * deterministic - the same tile and model always yield the same vector;
    * content-dependent - different tiles (or models) yield different vectors;
    * unit-norm - convenient for cosine-based change detection downstream.
    """

    #: This backend is a deterministic stand-in, not real model inference.
    backend_id = "deterministic-local"

    def embed(self, tile: RasterTile, spec: ModelSpec) -> List[float]:
        seed = self._seed(tile, spec)
        rng = np.random.Generator(np.random.PCG64(seed))
        vector = rng.standard_normal(spec.dimension)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector = vector / norm
        return [float(v) for v in vector]

    @staticmethod
    def _seed(tile: RasterTile, spec: ModelSpec) -> int:
        """Derive a 64-bit seed from the model and tile content."""
        hasher = hashlib.sha256()
        hasher.update(spec.name.encode("utf-8"))
        hasher.update(struct.pack("<i", spec.dimension))
        hasher.update(tile.format.strip().lower().encode("utf-8"))
        hasher.update(tile.dtype.encode("utf-8"))
        hasher.update(struct.pack("<iii", tile.width, tile.height, tile.bands))
        if tile.data is not None:
            # Encode the pixel values in a fixed binary layout so the seed is
            # stable across runs/platforms.
            hasher.update(struct.pack("<%df" % len(tile.data), *tile.data))
        digest = hasher.digest()
        return int.from_bytes(digest[:8], "little", signed=False)


#: The process-wide default backend used when none is supplied.
_DEFAULT_BACKEND = DeterministicLocalBackend()


def embed_tile(
    tile: RasterTile,
    model: str,
    *,
    registry: Optional[ModelRegistry] = None,
    backend: Optional[EmbeddingBackend] = None,
    supported_formats: Iterable[str] = DEFAULT_SUPPORTED_FORMATS,
    max_dimension: int = MAX_TILE_DIMENSION,
) -> EmbeddingResult:
    """Embed a tile with exactly one selected model (Req 9.1, 9.2, 9.7).

    Selects ``model`` from the configurable ``registry`` (default
    :class:`ModelRegistry` containing Clay, Prithvi-EO-2.0, and SatCLIP -
    Requirement 9.2); an unknown model raises ``ValidationError``. Validates the
    tile against the supported-format set and the ``<= 1024 x 1024`` size limit
    **before** computing anything, so an empty, oversized, or unsupported tile
    raises ``ValidationError`` and yields no embedding (Requirement 9.7). On
    success returns an :class:`EmbeddingResult` whose ``dimension`` is
    determined by the selected model (Requirement 9.1).
    """
    reg = registry if registry is not None else ModelRegistry()
    # Selection first: a request must name exactly one known model (Req 9.2).
    spec = reg.get(model)
    # Normalize the tile (a raw JSON dict that failed best-effort coercion in
    # the MCP runtime surfaces here as a clean ValidationError, not an opaque
    # AttributeError), then validate before any embedding work (Req 9.7).
    tile = coerce_raster_tile(tile)
    validate_tile(tile, supported_formats=supported_formats, max_dimension=max_dimension)

    eng = backend if backend is not None else _DEFAULT_BACKEND
    vector = eng.embed(tile, spec)
    # Defensive: the embedding dimensionality is model-determined (Req 9.1).
    if len(vector) != spec.dimension:  # pragma: no cover - guards a faulty backend
        raise ValidationError(
            "backend produced %d-d vector but model %r expects %d-d"
            % (len(vector), spec.name, spec.dimension),
            source=_SOURCE,
        )
    return EmbeddingResult(
        model=spec.name,
        dimension=spec.dimension,
        vector=vector,
        backend=getattr(eng, "backend_id", "unknown"),
        structure_only=tile.data is None,
    )

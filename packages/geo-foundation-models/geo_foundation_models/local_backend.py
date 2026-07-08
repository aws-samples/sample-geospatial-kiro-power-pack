"""A real-weight embedding backend that runs a model **in-process** (local).

This is the local counterpart to
:class:`~geo_foundation_models.remote_backend.RemoteEndpointBackend`: for small
or interactive workflows you often don't want to stand up (or pay for) a remote
inference endpoint. :class:`LocalCallableBackend` runs a real model in the same
process by calling an embedding function you provide — so the pack's default
install stays light (no torch/weights dependency shipped here), while anyone who
has a local model (a companion package, a notebook function, a small torch
wrapper) can plug it in and get **calibrated** embeddings without a server.

The local function receives the same payload mapping a remote endpoint would
(:func:`~geo_foundation_models.remote_backend.build_tile_payload`) and returns a
vector, so one implementation is interchangeable between a local callable and an
HTTP handler::

    # my_models.py
    def embed(payload: dict) -> list[float]:
        # payload has model/dimension/width/height/bands/format/dtype/data
        return my_clay_model.embed(payload)      # a real, in-process model

Then either inject it directly (``LocalCallableBackend(embed_fn=embed)``) or,
for the running MCP server, point the environment at it::

    export GEO_FM_EMBED_LOCAL_CALLABLE="my_models:embed"

For a heavier turnkey local model, run it behind a localhost HTTP server and use
:class:`RemoteEndpointBackend` with ``GEO_FM_EMBED_ENDPOINT_URL=http://localhost:...``
instead; this callable seam is the lightweight, in-process path.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, List, Mapping, Optional

from geo_common.errors import ValidationError

from geo_foundation_models.embedding import EmbeddingBackend
from geo_foundation_models.models import ModelSpec, RasterTile
from geo_foundation_models.remote_backend import (
    build_tile_payload,
    coerce_embedding_vector,
)

__all__ = ["LocalCallableBackend", "load_callable"]

_SOURCE = "geo-foundation-models"

#: A local embedding callable: ``payload_mapping -> vector`` (any shape
#: :func:`~geo_foundation_models.remote_backend.coerce_embedding_vector` accepts).
LocalEmbedFn = Callable[[Mapping[str, Any]], Any]


def load_callable(dotted_path: str) -> LocalEmbedFn:
    """Import and return the embedding callable named by ``dotted_path``.

    Accepts ``"package.module:attr"`` or ``"package.module.attr"``. Raises an
    ``Error_Taxonomy`` :class:`~geo_common.errors.ValidationError` when the path
    is malformed, the module/attribute cannot be imported, or the target is not
    callable — so a misconfiguration fails cleanly at startup rather than at the
    first embed.
    """
    if not isinstance(dotted_path, str) or not dotted_path.strip():
        raise ValidationError(
            "local embedding callable path must be a non-empty string like "
            "'my_module:embed'",
            source=_SOURCE,
            detail={"parameter": "dotted_path"},
        )
    path = dotted_path.strip()
    module_name, sep, attr = path.partition(":")
    if not sep:
        # Fall back to the last dotted segment as the attribute.
        module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValidationError(
            "malformed local callable path %r; expected 'module:attr' or "
            "'module.attr'" % (dotted_path,),
            source=_SOURCE,
            detail={"parameter": "dotted_path"},
        )
    try:
        module = importlib.import_module(module_name)
        target = getattr(module, attr)
    except (ImportError, AttributeError) as exc:
        raise ValidationError(
            "could not load local embedding callable %r: %s"
            % (dotted_path, str(exc) or type(exc).__name__),
            source=_SOURCE,
            detail={"parameter": "dotted_path"},
            original=str(exc) or type(exc).__name__,
        ) from exc
    if not callable(target):
        raise ValidationError(
            "local embedding target %r is not callable" % (dotted_path,),
            source=_SOURCE,
            detail={"parameter": "dotted_path"},
        )
    return target


class LocalCallableBackend(EmbeddingBackend):
    """Embed a tile by calling a local, in-process embedding function.

    Provide the function directly via ``embed_fn`` or by import path via
    ``dotted_path`` (exactly one). The function is called with the shared tile
    payload mapping and must return a vector (a sequence of numbers, or a
    mapping/batched shape that
    :func:`~geo_foundation_models.remote_backend.coerce_embedding_vector`
    understands). ``backend_id`` is the provenance recorded on results; it
    defaults to ``"local:<module.attr>"`` (or ``"local-callable"``) and is
    deliberately not ``"deterministic-local"`` so change tools treat the result
    as calibrated.
    """

    def __init__(
        self,
        *,
        embed_fn: Optional[LocalEmbedFn] = None,
        dotted_path: Optional[str] = None,
        backend_id: Optional[str] = None,
    ) -> None:
        if (embed_fn is None) == (dotted_path is None):
            raise ValidationError(
                "provide exactly one of embed_fn or dotted_path for the local "
                "embedding backend",
                source=_SOURCE,
                detail={"parameter": "embed_fn|dotted_path"},
            )
        if dotted_path is not None:
            self._embed_fn = load_callable(dotted_path)
            default_id = "local:%s" % dotted_path.strip().replace(":", ".")
        else:
            self._embed_fn = embed_fn  # type: ignore[assignment]
            default_id = "local-callable"
        self.backend_id = backend_id or default_id

    def embed(self, tile: RasterTile, spec: ModelSpec) -> List[float]:
        """Embed ``tile`` with ``spec`` via the local callable.

        Any exception raised inside the callable is normalized to an
        ``Error_Taxonomy`` :class:`~geo_common.errors.ValidationError` naming the
        local backend, and the returned object is validated into a flat list of
        finite floats (dimensionality is enforced by ``embed_tile``).
        """
        payload = build_tile_payload(tile, spec)
        try:
            result = self._embed_fn(payload)
        except Exception as exc:  # noqa: BLE001 - normalize any callable failure
            raise ValidationError(
                "local embedding callable raised: %s"
                % (str(exc) or type(exc).__name__),
                source=_SOURCE,
                detail={"backend": self.backend_id},
                original=str(exc) or type(exc).__name__,
            ) from exc
        return coerce_embedding_vector(result, source=_SOURCE)

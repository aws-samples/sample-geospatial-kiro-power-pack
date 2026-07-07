"""A real-weight embedding backend that calls a remote inference endpoint.

This is the production seam the change-detection / change-map tools are "gated
on": the default :class:`~geo_foundation_models.embedding.DeterministicLocalBackend`
is an honest stand-in with no semantic structure, so a change score under it is
~0.5 noise. :class:`RemoteEndpointBackend` swaps in a **real** model by
offloading the tile embedding to a configured inference endpoint — matching the
pack's "bring compute to the data / offload heavy compute" posture instead of
shipping a multi-GB local torch + weights extra.

Two transports are supported, chosen by configuration:

* **Generic HTTPS endpoint** — POST the tile as JSON to any hosted model server
  (a SageMaker *async/serverless* HTTPS URL, a self-hosted TorchServe, etc.),
  with optional header auth. Uses a blocking ``httpx.Client``.
* **AWS SageMaker runtime endpoint** — invoke a named SageMaker endpoint through
  ``boto3`` (``sagemaker-runtime``). ``boto3`` is an optional dependency
  (``geo-foundation-models[remote]``); a clear error is raised if it is missing.

The :class:`~geo_foundation_models.embedding.EmbeddingBackend` seam is
synchronous, so ``embed`` here is a blocking call — appropriate for a
single-client MCP stdio server and a natural fit for the synchronous SageMaker
SDK. No change to the existing ``embed_tile`` / ``embed_asset`` pipeline is
needed: any server method that accepts a ``backend`` accepts this one.

Wire contract (default, documented and adaptable)
-------------------------------------------------
Request body (``application/json``)::

    {"model": "Clay", "dimension": 768, "width": W, "height": H,
     "bands": B, "format": "GTiff", "dtype": "uint16", "data": [ ...floats... ]}

Response — any of::

    {"vector": [ ...floats... ]}
    {"embedding": [ ...floats... ]}
    {"predictions": [[ ...floats... ]]}     # first row is used
    [ ...floats... ]                          # a bare JSON array
    [[ ...floats... ]]                        # first row is used
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, List, Mapping, Optional

from geo_common.errors import UpstreamError, ValidationError

from geo_foundation_models.embedding import EmbeddingBackend
from geo_foundation_models.models import ModelSpec, RasterTile

__all__ = [
    "RemoteEndpointBackend",
    "backend_from_env",
    "build_tile_payload",
    "coerce_embedding_vector",
    "ENV_ENDPOINT_URL",
    "ENV_SAGEMAKER_ENDPOINT",
    "ENV_API_KEY",
    "ENV_API_KEY_HEADER",
    "ENV_BACKEND_ID",
    "ENV_REGION",
    "ENV_LOCAL_CALLABLE",
]

_SOURCE = "geo-foundation-models"

#: Environment keys that configure the remote backend (read in ``main()``).
ENV_ENDPOINT_URL = "GEO_FM_EMBED_ENDPOINT_URL"
ENV_SAGEMAKER_ENDPOINT = "GEO_FM_SAGEMAKER_ENDPOINT"
ENV_API_KEY = "GEO_FM_EMBED_API_KEY"
ENV_API_KEY_HEADER = "GEO_FM_EMBED_API_KEY_HEADER"
ENV_BACKEND_ID = "GEO_FM_EMBED_BACKEND_ID"
ENV_REGION = "GEO_FM_EMBED_REGION"
#: Environment key selecting an in-process local embedding callable (see
#: :mod:`geo_foundation_models.local_backend`).
ENV_LOCAL_CALLABLE = "GEO_FM_EMBED_LOCAL_CALLABLE"


def build_tile_payload(tile: "RasterTile", spec: "ModelSpec") -> "dict[str, Any]":
    """The canonical embedding request payload shared by every real backend.

    Both the remote HTTPS/SageMaker transport and the in-process local callable
    receive exactly this mapping, so one embedding implementation works in
    either place (a local function and an HTTP handler are interchangeable).
    """
    return {
        "model": spec.name,
        "dimension": spec.dimension,
        "width": tile.width,
        "height": tile.height,
        "bands": tile.bands,
        "format": tile.format,
        "dtype": tile.dtype,
        "data": list(tile.data) if tile.data is not None else None,
        # Optional spatio-temporal context for location/time-aware backends
        # (None when the caller did not supply it).
        "latlon": list(tile.latlon) if tile.latlon is not None else None,
        "acquired": tile.acquired,
    }

#: A callable that turns the request payload (a JSON-serializable mapping) into
#: the raw response text/bytes. Injected by tests to exercise ``embed`` without
#: any network or AWS access.
InvokeFn = Callable[[Mapping[str, Any]], "str | bytes"]


class RemoteEndpointBackend(EmbeddingBackend):
    """Embed a tile by calling a configured remote inference endpoint.

    Exactly one transport must be configured: ``endpoint_url`` (generic HTTPS)
    or ``sagemaker_endpoint`` (a SageMaker endpoint name). ``api_key`` (HTTPS
    only) is sent in ``api_key_header`` as ``f"{api_key_prefix}{api_key}"``.
    ``backend_id`` is the provenance recorded on every resulting embedding — it
    is deliberately **not** ``"deterministic-local"`` so change tools can tell a
    real backend is active.

    ``invoke`` (test-only) fully replaces the transport with a callable, so the
    parsing/validation contract can be tested hermetically.
    """

    def __init__(
        self,
        *,
        endpoint_url: Optional[str] = None,
        sagemaker_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_header: str = "Authorization",
        api_key_prefix: str = "Bearer ",
        region: Optional[str] = None,
        backend_id: Optional[str] = None,
        timeout_s: float = 30.0,
        invoke: Optional[InvokeFn] = None,
    ) -> None:
        configured = [t for t in (endpoint_url, sagemaker_endpoint) if t]
        if invoke is None and len(configured) != 1:
            raise ValidationError(
                "configure exactly one remote transport: endpoint_url "
                "(generic HTTPS) or sagemaker_endpoint (SageMaker endpoint name)",
                source=_SOURCE,
                detail={"parameter": "endpoint_url|sagemaker_endpoint"},
            )
        self.endpoint_url = endpoint_url
        self.sagemaker_endpoint = sagemaker_endpoint
        self.api_key = api_key
        self.api_key_header = api_key_header
        self.api_key_prefix = api_key_prefix
        self.region = region
        self.timeout_s = timeout_s
        self._invoke = invoke
        if backend_id:
            self.backend_id = backend_id
        elif sagemaker_endpoint:
            self.backend_id = f"sagemaker:{sagemaker_endpoint}"
        else:
            self.backend_id = "remote-endpoint"

    # -- EmbeddingBackend interface ------------------------------------

    def embed(self, tile: RasterTile, spec: ModelSpec) -> List[float]:
        """Embed ``tile`` with ``spec`` by calling the configured endpoint.

        Raises an ``Error_Taxonomy`` :class:`~geo_common.errors.UpstreamError`
        when the endpoint fails or returns an unusable payload, so a remote
        failure surfaces on the shared taxonomy rather than as a raw library
        exception. Dimensionality is enforced by ``embed_tile`` downstream; a
        clearly-wrong response (empty / non-numeric) is rejected here.
        """
        payload = self._payload(tile, spec)
        raw = self._call(payload)
        return self._parse_vector(raw)

    # -- payload / response helpers ------------------------------------

    @staticmethod
    def _payload(tile: RasterTile, spec: ModelSpec) -> "dict[str, Any]":
        return build_tile_payload(tile, spec)

    def _call(self, payload: Mapping[str, Any]) -> "str | bytes":
        if self._invoke is not None:
            return self._invoke(payload)
        if self.sagemaker_endpoint:
            return self._call_sagemaker(payload)
        return self._call_https(payload)

    def _call_https(self, payload: Mapping[str, Any]) -> str:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers[self.api_key_header] = f"{self.api_key_prefix}{self.api_key}"
        try:
            resp = httpx.post(
                self.endpoint_url,
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as exc:
            raise UpstreamError(
                "remote embedding endpoint request failed: %s"
                % (str(exc) or type(exc).__name__),
                source=_SOURCE,
                original=str(exc) or type(exc).__name__,
            ) from exc
        if resp.status_code >= 400:
            raise UpstreamError(
                "remote embedding endpoint returned HTTP %d" % resp.status_code,
                source=_SOURCE,
                detail={"status_code": resp.status_code},
            )
        return resp.text

    def _call_sagemaker(self, payload: Mapping[str, Any]) -> bytes:
        try:
            import boto3  # type: ignore
        except Exception as exc:  # noqa: BLE001 - optional dependency
            raise ValidationError(
                "SageMaker transport requires boto3; install the "
                "geo-foundation-models[remote] extra",
                source=_SOURCE,
                original=str(exc) or type(exc).__name__,
            ) from exc
        try:
            client = boto3.client("sagemaker-runtime", region_name=self.region)
            resp = client.invoke_endpoint(
                EndpointName=self.sagemaker_endpoint,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(payload).encode("utf-8"),
            )
            return resp["Body"].read()
        except Exception as exc:  # noqa: BLE001 - normalize any boto/endpoint error
            raise UpstreamError(
                "SageMaker endpoint invocation failed: %s"
                % (str(exc) or type(exc).__name__),
                source=_SOURCE,
                original=str(exc) or type(exc).__name__,
            ) from exc

    def _parse_vector(self, raw: "str | bytes") -> List[float]:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise UpstreamError(
                "remote endpoint returned a non-JSON response",
                source=_SOURCE,
                original=str(exc) or type(exc).__name__,
            ) from exc
        return coerce_embedding_vector(parsed, source=_SOURCE)


def coerce_embedding_vector(candidate: Any, *, source: str = _SOURCE) -> List[float]:
    """Normalize a backend's returned object into a flat list of finite floats.

    Accepts the shapes a real embedding backend (remote endpoint or local
    callable) might return: a bare sequence of numbers, a single batched row
    (``[[...]]``), or a mapping under ``vector`` / ``embedding`` / ``embeddings``
    / ``predictions``. Raises an ``Error_Taxonomy``
    :class:`~geo_common.errors.UpstreamError` for an unusable shape (missing
    key, empty, non-numeric, or non-finite), so a misbehaving backend surfaces
    on the taxonomy rather than as a raw exception.
    """
    if isinstance(candidate, Mapping):
        for key in ("vector", "embedding", "embeddings", "predictions"):
            if key in candidate:
                candidate = candidate[key]
                break
        else:
            raise UpstreamError(
                "embedding backend result has no vector/embedding/predictions key",
                source=source,
                detail={"keys": sorted(map(str, candidate.keys()))},
            )
    # Unwrap a single batched row: [[...]] or predictions=[[...]].
    if (
        isinstance(candidate, (list, tuple))
        and len(candidate) > 0
        and isinstance(candidate[0], (list, tuple))
    ):
        candidate = candidate[0]

    if not isinstance(candidate, (list, tuple)) or len(candidate) == 0:
        raise UpstreamError(
            "embedding backend returned an empty or non-array embedding",
            source=source,
        )
    try:
        vector = [float(v) for v in candidate]
    except (TypeError, ValueError) as exc:
        raise UpstreamError(
            "embedding backend result contains non-numeric values",
            source=source,
            original=str(exc) or type(exc).__name__,
        ) from exc
    import math

    if any(not math.isfinite(v) for v in vector):
        raise UpstreamError(
            "embedding backend result contains non-finite values",
            source=source,
        )
    return vector


def backend_from_env(
    env: Optional[Mapping[str, str]] = None,
) -> "Optional[EmbeddingBackend]":
    """Build a real-weight backend from environment config, or ``None``.

    Selects, in precedence order:

    1. ``GEO_FM_EMBED_LOCAL_CALLABLE`` — an in-process
       :class:`~geo_foundation_models.local_backend.LocalCallableBackend`
       (real model, no endpoint) for small/interactive workflows;
    2. ``GEO_FM_SAGEMAKER_ENDPOINT`` — a SageMaker
       :class:`RemoteEndpointBackend`;
    3. ``GEO_FM_EMBED_ENDPOINT_URL`` — a generic-HTTPS
       :class:`RemoteEndpointBackend`.

    Returns ``None`` when none is set, so the server falls back to the honest
    deterministic stand-in. Configure exactly one; the order above breaks ties.
    """
    env = env if env is not None else os.environ
    backend_id = (env.get(ENV_BACKEND_ID) or "").strip() or None

    local = (env.get(ENV_LOCAL_CALLABLE) or "").strip()
    if local:
        # Deferred import avoids a module-load cycle (local_backend imports the
        # shared coerce/payload helpers from here).
        from geo_foundation_models.local_backend import LocalCallableBackend

        return LocalCallableBackend(dotted_path=local, backend_id=backend_id)

    sagemaker = (env.get(ENV_SAGEMAKER_ENDPOINT) or "").strip()
    url = (env.get(ENV_ENDPOINT_URL) or "").strip()
    if not sagemaker and not url:
        return None
    api_key = (env.get(ENV_API_KEY) or "").strip() or None
    api_key_header = (env.get(ENV_API_KEY_HEADER) or "").strip() or "Authorization"
    backend_id = (env.get(ENV_BACKEND_ID) or "").strip() or None
    region = (env.get(ENV_REGION) or env.get("AWS_REGION") or "").strip() or None
    if sagemaker:
        return RemoteEndpointBackend(
            sagemaker_endpoint=sagemaker,
            region=region,
            backend_id=backend_id,
        )
    return RemoteEndpointBackend(
        endpoint_url=url,
        api_key=api_key,
        api_key_header=api_key_header,
        region=region,
        backend_id=backend_id,
    )

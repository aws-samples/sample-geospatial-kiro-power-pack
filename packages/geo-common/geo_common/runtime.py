"""Shared MCP stdio runtime for every Geospatial Power Pack server.

Every server registers its tools on :class:`~geo_common.server.BaseGeoServer`
via :meth:`~geo_common.server.BaseGeoServer.register_tool`. This module turns
those registered tools into a live **MCP server over stdio** so Kiro can list
and call them through a ``mcp.json`` ``command`` entry, exactly like the
life-sciences reference pack.

The design keeps the heavy dependency (`the ``mcp`` SDK) **lazily imported**:
nothing here imports ``mcp`` at module load, so the unit/property/integration
test suites - which exercise the tool callables directly - never need the SDK
installed. The SDK is only required at deploy time, when a server's console
entry point actually serves stdio.

Public surface:

* :func:`run_server` - the one call each server's ``main()`` makes; it runs the
  startup credential guard, then serves the server's tools over stdio until the
  client disconnects.
* :func:`build_tool_input_schema` / :func:`tool_result_to_text` - the pure
  helpers that map a registered tool's signature to a JSON-Schema ``inputSchema``
  and serialize a tool result to MCP text content. These are import-light and
  unit-tested without the SDK.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import typing
from typing import TYPE_CHECKING, Any, Dict, List

from geo_common.errors import GeoError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from geo_common.server import BaseGeoServer, RegisteredTool

__all__ = [
    "run_server",
    "serve_stdio",
    "build_tool_input_schema",
    "coerce_tool_arguments",
    "tool_result_to_text",
]


# ---------------------------------------------------------------------------
# Pure helpers (no ``mcp`` dependency) - unit-tested directly
# ---------------------------------------------------------------------------

#: Best-effort mapping from a Python annotation (as a type or its string name,
#: since ``from __future__ import annotations`` stores annotations as strings)
#: to a JSON-Schema primitive ``type``. Anything unrecognized is left untyped
#: (permissive), which is a valid JSON Schema that accepts any value.
_JSON_TYPE_BY_NAME = {
    "int": "integer",
    "float": "number",
    "str": "string",
    "bool": "boolean",
    "bytes": "string",
    "list": "array",
    "tuple": "array",
    "sequence": "array",
    "dict": "object",
    "mapping": "object",
}


def _annotation_to_json_type(annotation: Any) -> "str | None":
    """Map a parameter annotation to a JSON-Schema ``type`` (best effort).

    Handles both real types and string annotations (e.g. ``"int"``,
    ``"Optional[str]"``, ``"Sequence[str]"``). Returns ``None`` when no
    confident mapping exists, leaving the property untyped/permissive.
    """
    if annotation is inspect.Parameter.empty or annotation is None:
        return None

    # Resolve to a lowercase base-type name from either a type or a string.
    if isinstance(annotation, type):
        name = annotation.__name__.lower()
    else:
        name = str(annotation).lower()

    # Strip common typing wrappers / qualifiers so the base type shows through.
    for token, base in (
        ("list", "list"),
        ("tuple", "tuple"),
        ("sequence", "sequence"),
        ("dict", "dict"),
        ("mapping", "mapping"),
    ):
        if token in name:
            return _JSON_TYPE_BY_NAME[base]
    for token in ("bool", "int", "float", "str", "bytes"):
        # Match the token as a whole word-ish fragment.
        if token in name:
            return _JSON_TYPE_BY_NAME[token]
    return None


def _resolved_hints(func: Any) -> Dict[str, Any]:
    """Best-effort resolved type hints for ``func`` (empty on failure).

    ``from __future__ import annotations`` stores annotations as strings, so a
    parameter typed as a pydantic model shows up as the bare name ``"RasterTile"``
    in the signature. Resolving the hints turns it back into the real class so
    :func:`build_tool_input_schema` can emit the model's JSON Schema. Any
    resolution failure yields ``{}`` and the caller falls back to the
    string-based best-effort mapping.
    """
    try:
        return typing.get_type_hints(func)
    except Exception:  # noqa: BLE001 - unresolved annotations -> no expansion
        return {}


def _pydantic_schema_for_hint(hint: Any) -> "tuple[Dict[str, Any] | None, Dict[str, Any]]":
    """Return ``(property_schema, defs)`` for a pydantic-model-typed ``hint``.

    Handles a direct ``Model`` annotation, ``Optional[Model]`` / ``Union[...]``,
    and ``list[Model]`` / ``tuple[Model, ...]`` so a model-typed tool parameter
    advertises its full field shape instead of an opaque ``{}``. Pydantic emits
    shared/nested models under a ``$defs`` block referenced by **root-relative**
    ``#/$defs/<Name>`` pointers, which only resolve if ``$defs`` lives at the
    *root* of the tool ``inputSchema``; this therefore returns the model's own
    ``$defs`` separately (second element) for the caller to merge into the root,
    leaving the property schema's refs intact. Returns ``(None, {})`` when no
    pydantic model is involved or its schema cannot be produced.
    """
    try:
        import pydantic
    except ImportError:  # pragma: no cover - pydantic is a geo-common dependency
        return None, {}

    if hint is None or hint is inspect.Parameter.empty:
        return None, {}

    # Direct pydantic model annotation.
    if isinstance(hint, type) and issubclass(hint, pydantic.BaseModel):
        try:
            schema = hint.model_json_schema(ref_template="#/$defs/{model}")
        except Exception:  # noqa: BLE001 - defensive; fall back to untyped
            return None, {}
        defs = schema.pop("$defs", None) or schema.pop("definitions", None) or {}
        return schema, defs

    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    # Optional[Model] / Union[...]: use the first member that yields a schema.
    if origin is typing.Union:
        for member in args:
            if member is type(None):
                continue
            schema, defs = _pydantic_schema_for_hint(member)
            if schema is not None:
                return schema, defs
        return None, {}

    # list[Model] / tuple[Model, ...]: an array whose items are the model schema.
    if origin in (list, tuple) and args:
        item_schema, defs = _pydantic_schema_for_hint(args[0])
        if item_schema is not None:
            return {"type": "array", "items": item_schema}, defs
        return None, {}

    return None, {}


def build_tool_input_schema(func: Any) -> Dict[str, Any]:
    """Build a JSON-Schema ``inputSchema`` for a registered tool callable.

    Introspects ``func``'s signature (a bound method, so ``self`` is already
    removed) and produces an ``object`` schema whose ``properties`` are the
    tool's parameters and whose ``required`` list is the parameters without a
    default. A parameter typed as a pydantic model (or ``Optional`` / ``list`` /
    ``tuple`` of one) advertises that model's full JSON Schema - its fields,
    types, and required keys - so an MCP client can construct a valid argument
    instead of guessing at an opaque object; other parameter types are mapped
    best-effort. ``**kwargs`` makes the schema permissive
    (``additionalProperties: true``). The result is always a valid JSON Schema,
    even when types cannot be inferred.
    """
    properties: Dict[str, Any] = {}
    required: List[str] = []
    defs: Dict[str, Any] = {}
    additional = False

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return {"type": "object", "properties": {}, "additionalProperties": True}

    hints = _resolved_hints(func)

    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            additional = True
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.POSITIONAL_ONLY,
        ):
            # MCP tool arguments are a JSON object (named), so positional-only
            # and *args parameters are not representable; skip them.
            continue

        # Prefer a full pydantic model schema when the parameter is model-typed
        # so the client sees the expected fields; otherwise fall back to the
        # best-effort primitive-type mapping.
        prop, prop_defs = _pydantic_schema_for_hint(hints.get(name))
        if prop is None:
            prop = {}
            json_type = _annotation_to_json_type(param.annotation)
            if json_type is not None:
                prop["type"] = json_type
        elif prop_defs:
            # Hoist the model's $defs to the root so the property schema's
            # "#/$defs/<Name>" references resolve. Same-named models produce
            # identical definitions, so a plain merge is safe.
            defs.update(prop_defs)
        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    if defs:
        schema["$defs"] = defs
    schema["additionalProperties"] = additional
    return schema


def coerce_tool_arguments(func: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce JSON tool arguments to the callable's annotated pydantic types.

    MCP delivers tool-call arguments as plain JSON (dicts / lists / scalars), but
    a tool may type a parameter as a pydantic model (e.g. ``geometry:
    GeoJSONGeometry``) or a list of models. This walks ``func``'s resolved type
    hints and, for each provided argument whose annotation is a
    ``pydantic.BaseModel`` subclass (or ``Optional[...]`` / ``list[...]`` /
    ``tuple[...]`` of one), validates the raw value into that model so the tool
    receives the type it expects.

    Coercion is **best-effort and total**: if pydantic is unavailable, the type
    hints cannot be resolved, or a value fails validation, the original value is
    passed through unchanged so the tool's own validation surfaces the error as
    an ``Error_Taxonomy`` ``ValidationError`` rather than this helper raising.
    """
    try:
        import pydantic
    except ImportError:  # pragma: no cover - pydantic is a geo-common dependency
        return arguments

    try:
        hints = typing.get_type_hints(func)
    except Exception:  # noqa: BLE001 - unresolved annotations -> skip coercion
        return arguments

    coerced: Dict[str, Any] = {}
    for key, value in arguments.items():
        coerced[key] = _coerce_value(value, hints.get(key), pydantic.BaseModel)
    return coerced


def _coerce_value(value: Any, hint: Any, base: type) -> Any:
    """Coerce one value to ``hint`` when it names a pydantic model (best effort)."""
    if hint is None:
        return value

    # Direct model annotation: validate a mapping into the model.
    if isinstance(hint, type) and issubclass(hint, base):
        if isinstance(value, dict):
            try:
                return hint.model_validate(value)
            except Exception:  # noqa: BLE001 - let the tool validate/raise
                return value
        return value

    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    # Optional[...] / Union[...]: try each non-None member, recursing so that
    # Optional[Model], Optional[list[Model]], etc. are all handled.
    if origin is typing.Union:
        for member in args:
            if member is type(None):
                continue
            result = _coerce_value(value, member, base)
            if result is not value:
                return result
        return value

    # list[Model] / tuple[Model, ...]: coerce each mapping element.
    if origin in (list, tuple) and args:
        member = args[0]
        if isinstance(value, list) and isinstance(member, type) and issubclass(member, base):
            out = []
            for item in value:
                if isinstance(item, dict):
                    try:
                        out.append(member.model_validate(item))
                        continue
                    except Exception:  # noqa: BLE001
                        pass
                out.append(item)
            return out
        return value

    return value


def tool_result_to_text(result: Any) -> str:
    """Serialize a tool result to a JSON (or plain) text string for MCP.

    Pydantic models (and lists/tuples of them) are dumped via ``model_dump``;
    mappings and other JSON-friendly values are ``json.dumps``-ed; plain
    strings pass through unchanged. A non-serializable value falls back to its
    ``str()`` form so a tool result can always be returned as text content.
    """
    if isinstance(result, str):
        return result

    def _coerce(value: Any) -> Any:
        # Pydantic v2 model -> JSON-able dict.
        dump = getattr(value, "model_dump", None)
        if callable(dump):
            try:
                return value.model_dump(mode="json")
            except TypeError:  # pragma: no cover - non-pydantic model_dump
                return value.model_dump()
        if isinstance(value, (list, tuple)):
            return [_coerce(item) for item in value]
        if isinstance(value, dict):
            return {key: _coerce(item) for key, item in value.items()}
        return value

    coerced = _coerce(result)
    try:
        return json.dumps(coerced, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(result)


# ---------------------------------------------------------------------------
# stdio MCP runtime (lazily imports the ``mcp`` SDK)
# ---------------------------------------------------------------------------


def _import_mcp():
    """Import the ``mcp`` SDK, or raise a clear, actionable error.

    Kept lazy so the test suites (which call tool functions directly) never
    require the SDK; it is only needed when actually serving stdio at deploy
    time.
    """
    try:
        from mcp import types  # noqa: F401
        from mcp.server import Server  # noqa: F401
        from mcp.server.stdio import stdio_server  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without mcp
        raise RuntimeError(
            "the 'mcp' package is required to serve a Geospatial Power Pack "
            "server over stdio. Install it (it is a dependency of geo-common: "
            "`uv pip install -e ./packages/geo-common`) and retry."
        ) from exc
    return Server, stdio_server, types


async def serve_stdio(server: "BaseGeoServer") -> None:
    """Serve ``server``'s registered tools over an MCP stdio connection.

    Builds an MCP :class:`~mcp.server.Server`, wires a ``list_tools`` handler
    (every registered tool, with a generated ``inputSchema``) and a
    ``call_tool`` handler (dispatches to the registered callable, awaiting it
    when it is a coroutine, and serializing the result to text content), then
    runs over stdin/stdout until the client disconnects. A
    :class:`~geo_common.errors.GeoError` raised by a tool is surfaced as an MCP
    tool error carrying its taxonomy category and secret-free message.
    """
    Server, stdio_server, types = _import_mcp()

    app = Server(server.server_name)

    @app.list_tools()
    async def list_tools() -> list:
        return [
            types.Tool(
                name=tool.name,
                description=tool.description or "",
                inputSchema=build_tool_input_schema(tool.func),
            )
            for tool in server.tools.values()
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: "Dict[str, Any] | None") -> list:
        tool: "RegisteredTool" = server.get_tool(name)  # NotFoundError if absent
        kwargs = coerce_tool_arguments(tool.func, dict(arguments or {}))
        try:
            result = tool.func(**kwargs)
            if inspect.isawaitable(result):
                result = await result
        except GeoError as exc:
            # Secret-free by construction; surface category + message as an error.
            raise RuntimeError("[%s] %s" % (exc.category.value, str(exc))) from exc
        return [types.TextContent(type="text", text=tool_result_to_text(result))]

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


def run_server(server: "BaseGeoServer") -> None:
    """Run a server's startup guard, then serve its tools over stdio.

    This is the single call a server's ``main()`` makes. It first runs the
    startup credential guard (:meth:`~geo_common.server.BaseGeoServer.start`),
    which refuses to start when a *Required* credential is missing (naming each
    missing ``mcp.json`` key), then serves stdio until the client disconnects.
    A clean ``KeyboardInterrupt`` / EOF shutdown returns normally.
    """
    server.start()
    try:
        asyncio.run(serve_stdio(server))
    except KeyboardInterrupt:  # pragma: no cover - interactive shutdown
        pass

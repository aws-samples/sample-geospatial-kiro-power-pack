"""Unit tests for the shared MCP stdio runtime helpers (geo_common.runtime).

These cover the import-light, SDK-free parts of the runtime:

* :func:`build_tool_input_schema` - turns a registered tool's signature into a
  valid JSON-Schema ``inputSchema`` (properties, required, additionalProperties).
* :func:`tool_result_to_text` - serializes tool results (pydantic models, lists
  of models, mappings, scalars, strings) to MCP text content.

The stdio serving path itself lazily imports the ``mcp`` SDK and is exercised at
deploy time; here we assert the runtime is wired onto ``BaseGeoServer`` (the
``run`` entry point exists) and that a server with a missing *Required*
credential refuses to start before any serving is attempted.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import pytest
from pydantic import BaseModel

from geo_common.errors import AuthenticationError, ErrorCategory
from geo_common.models import CredentialClassification, CredentialSpec
from geo_common.runtime import build_tool_input_schema, coerce_tool_arguments, tool_result_to_text
from geo_common.server import BaseGeoServer


# --- build_tool_input_schema ------------------------------------------------


class _Sample(BaseModel):
    name: str
    value: int


class _Leaf(BaseModel):
    kind: str


class _Branch(BaseModel):
    leaves: List[_Leaf]


def test_input_schema_marks_required_and_optional_params() -> None:
    async def tool(*, bbox: Sequence[float], limit: int = 1000, name: Optional[str] = None):
        return None

    schema = build_tool_input_schema(tool)
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"bbox", "limit", "name"}
    # Only the parameter without a default is required.
    assert schema["required"] == ["bbox"]
    # Best-effort JSON types.
    assert schema["properties"]["bbox"]["type"] == "array"
    assert schema["properties"]["limit"]["type"] == "integer"
    assert schema["additionalProperties"] is False


def test_input_schema_skips_self_and_flags_var_keyword() -> None:
    class Obj:
        def tool(self, *, q: str, **extra):  # noqa: ANN001
            return None

    schema = build_tool_input_schema(Obj().tool)
    assert "self" not in schema["properties"]
    assert set(schema["properties"]) == {"q"}
    assert schema["required"] == ["q"]
    # **extra -> additionalProperties true (permissive).
    assert schema["additionalProperties"] is True


def test_input_schema_is_valid_when_no_params() -> None:
    async def tool():
        return None

    schema = build_tool_input_schema(tool)
    assert schema == {"type": "object", "properties": {}, "additionalProperties": False}


def test_input_schema_expands_pydantic_model_param() -> None:
    """A pydantic-model param advertises its fields, not an opaque ``{}``."""

    async def tool(*, sample: _Sample, note: str = "x"):
        return None

    schema = build_tool_input_schema(tool)
    tile_schema = schema["properties"]["sample"]
    # The model's fields are exposed so an MCP client can build a valid object.
    assert tile_schema["type"] == "object"
    assert set(tile_schema["properties"]) == {"name", "value"}
    assert set(tile_schema.get("required", [])) == {"name", "value"}
    assert schema["required"] == ["sample"]


def test_input_schema_expands_optional_and_list_of_models() -> None:
    """``Optional[Model]`` and ``list[Model]`` params expand to model schemas."""

    async def tool(*, one: Optional[_Sample] = None, many: Optional[List[_Sample]] = None):
        return None

    schema = build_tool_input_schema(tool)
    one_schema = schema["properties"]["one"]
    assert one_schema["type"] == "object"
    assert set(one_schema["properties"]) == {"name", "value"}

    many_schema = schema["properties"]["many"]
    assert many_schema["type"] == "array"
    assert set(many_schema["items"]["properties"]) == {"name", "value"}


def test_input_schema_multitype_union_advertises_every_branch() -> None:
    """A multi-type ``Union`` param advertises all accepted forms via ``anyOf``.

    Regression: previously only the first union member (a pydantic model) was
    emitted, so an MCP client never saw that a path string was also accepted
    (the ``to_geoparquet`` ``src`` symptom). Every branch must be advertised.
    """

    async def tool(*, src: Union[_Sample, Dict[str, Any], str]):
        return None

    schema = build_tool_input_schema(tool)
    src_schema = schema["properties"]["src"]
    assert "anyOf" in src_schema
    types = {branch.get("type") for branch in src_schema["anyOf"]}
    # The model branch (object with fields), a generic object, and a string.
    assert "string" in types
    assert "object" in types
    # The model's fields still surface on its branch.
    model_branch = next(b for b in src_schema["anyOf"] if b.get("properties"))
    assert set(model_branch["properties"]) == {"name", "value"}


def test_input_schema_optional_model_stays_single_object() -> None:
    """``Optional[Model]`` collapses to the model object schema (no ``anyOf``)."""

    async def tool(*, one: Optional[_Sample] = None):
        return None

    one_schema = build_tool_input_schema(tool)["properties"]["one"]
    assert "anyOf" not in one_schema
    assert one_schema["type"] == "object"
    assert set(one_schema["properties"]) == {"name", "value"}


def test_input_schema_hoists_nested_model_defs_to_root() -> None:
    """A model with nested sub-models emits root ``$defs`` so refs resolve.

    Regression: pydantic references shared sub-models via root-relative
    ``#/$defs/<Name>`` pointers. If the per-parameter schema kept its ``$defs``
    nested under the property, those pointers would dangle (a JSON-Schema
    validator raises "PointerToNowhere"). The defs must be hoisted to the root
    of the tool ``inputSchema``.
    """

    async def tool(*, left: _Branch, right: _Branch):
        return None

    schema = build_tool_input_schema(tool)
    # Shared sub-model is defined once at the root, not nested under a property.
    assert "_Leaf" in schema.get("$defs", {})
    assert "$defs" not in schema["properties"]["left"]
    assert "$defs" not in schema["properties"]["right"]

    # The schema's refs resolve and a well-formed payload validates.
    jsonschema = pytest.importorskip("jsonschema")
    payload = {"left": {"leaves": [{"kind": "a"}]}, "right": {"leaves": []}}
    jsonschema.validate(payload, schema)  # raises if a $ref dangles


# --- tool_result_to_text ----------------------------------------------------


def test_serialize_pydantic_model() -> None:
    text = tool_result_to_text(_Sample(name="a", value=1))
    assert '"name": "a"' in text
    assert '"value": 1' in text


def test_serialize_list_of_models() -> None:
    text = tool_result_to_text([_Sample(name="a", value=1), _Sample(name="b", value=2)])
    assert text.startswith("[")
    assert '"name": "a"' in text and '"name": "b"' in text


def test_serialize_mapping_and_scalars() -> None:
    assert tool_result_to_text({"k": 1, "v": [1, 2]}) == '{"k": 1, "v": [1, 2]}'
    assert tool_result_to_text(42) == "42"
    assert tool_result_to_text(True) == "true"


def test_serialize_plain_string_passthrough() -> None:
    assert tool_result_to_text("already text") == "already text"


# --- coerce_tool_arguments --------------------------------------------------


class _Geometry(BaseModel):
    type: str
    coordinates: list


def test_coerce_dict_argument_into_pydantic_model() -> None:
    async def tool(*, geometry: _Geometry, dst_crs: str):
        return None

    coerced = coerce_tool_arguments(
        tool, {"geometry": {"type": "Point", "coordinates": [1.0, 2.0]}, "dst_crs": "EPSG:3857"}
    )
    assert isinstance(coerced["geometry"], _Geometry)
    assert coerced["geometry"].type == "Point"
    # Non-model arguments pass through unchanged.
    assert coerced["dst_crs"] == "EPSG:3857"


def test_coerce_optional_and_list_of_models() -> None:
    from typing import List, Optional

    async def tool(*, one: Optional[_Geometry] = None, many: Optional[List[_Geometry]] = None):
        return None

    coerced = coerce_tool_arguments(
        tool,
        {
            "one": {"type": "Point", "coordinates": [0, 0]},
            "many": [
                {"type": "Point", "coordinates": [1, 1]},
                {"type": "Point", "coordinates": [2, 2]},
            ],
        },
    )
    assert isinstance(coerced["one"], _Geometry)
    assert isinstance(coerced["many"], list) and all(
        isinstance(g, _Geometry) for g in coerced["many"]
    )


def test_coerce_passes_through_on_invalid_or_unannotated() -> None:
    async def tool(*, geometry: _Geometry, note=None):
        return None

    # Invalid payload for the model -> passed through unchanged (tool validates).
    bad = {"geometry": {"wrong": "shape"}, "note": 42}
    coerced = coerce_tool_arguments(tool, bad)
    assert coerced["geometry"] == {"wrong": "shape"}
    assert coerced["note"] == 42


# --- BaseGeoServer.run wiring + startup guard -------------------------------


def test_base_server_exposes_run_entry_point() -> None:
    server = BaseGeoServer()
    assert hasattr(server, "run") and callable(server.run)


def test_run_refuses_to_start_when_required_credential_missing() -> None:
    class _Guarded(BaseGeoServer):
        server_name = "guarded"

        def required_credentials(self) -> List[CredentialSpec]:
            return [
                CredentialSpec(
                    source="Example",
                    mcp_json_key="EXAMPLE_REQUIRED_KEY",
                    classification=CredentialClassification.REQUIRED,
                )
            ]

    server = _Guarded()
    # run() calls start() first; the guard fires before any stdio serving.
    with pytest.raises(AuthenticationError) as exc_info:
        server.run()
    assert exc_info.value.category is ErrorCategory.AUTHENTICATION
    assert "EXAMPLE_REQUIRED_KEY" in str(exc_info.value)

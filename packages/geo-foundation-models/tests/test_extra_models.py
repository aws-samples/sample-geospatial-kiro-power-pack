"""Tests for GEO_FM_EXTRA_MODELS parsing and registry_from_env (bring-your-own model)."""

from __future__ import annotations

import pytest

from geo_common.errors import ValidationError

from geo_foundation_models.embedding import (
    DEFAULT_MODELS,
    extra_models_from_env,
    registry_from_env,
)


def test_empty_or_unset_yields_no_extras():
    assert extra_models_from_env(env={}) == []
    assert extra_models_from_env(env={"GEO_FM_EXTRA_MODELS": "  "}) == []


def test_parses_multiple_models():
    specs = extra_models_from_env(env={"GEO_FM_EXTRA_MODELS": "MyModel:2048, Foo:512"})
    by_name = {s.name: s.dimension for s in specs}
    assert by_name == {"MyModel": 2048, "Foo": 512}


def test_name_with_colons_is_supported():
    # rpartition on the last colon keeps names like "org:model" intact.
    (spec,) = extra_models_from_env(env={"GEO_FM_EXTRA_MODELS": "org:model:1024"})
    assert spec.name == "org:model" and spec.dimension == 1024


@pytest.mark.parametrize("bad", ["NoDim", "Name:", ":512", "Name:abc", "Name:0", "Name:-4"])
def test_malformed_entries_are_validation_errors(bad):
    with pytest.raises(ValidationError):
        extra_models_from_env(env={"GEO_FM_EXTRA_MODELS": bad})


def test_registry_from_env_merges_defaults_and_extras():
    reg = registry_from_env(env={"GEO_FM_EXTRA_MODELS": "MyModel:2048"})
    # Defaults still present...
    assert reg.get("Clay-v1.5").dimension == 1024
    # ...plus the extra, selectable by name.
    assert reg.get("MyModel").dimension == 2048


def test_extra_can_override_a_default_dimension():
    reg = registry_from_env(env={"GEO_FM_EXTRA_MODELS": "SatCLIP:9999"})
    assert reg.get("SatCLIP").dimension == 9999
    # Untouched defaults keep their dimension.
    assert reg.get("Clay").dimension == DEFAULT_MODELS["Clay"].dimension

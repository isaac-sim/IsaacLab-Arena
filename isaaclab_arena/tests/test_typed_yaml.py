# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify strict typed YAML loading without importing the simulation runtime."""

from dataclasses import dataclass

import pytest

from isaaclab_arena.hydra.typed_yaml import load_typed_yaml_cfg, render_typed_yaml_cfg


@dataclass
class ExampleCfg:
    required_name: str
    count: int = 2


def test_load_typed_yaml_applies_values_and_schema_defaults(tmp_path):
    config_path = tmp_path / "example.yaml"
    config_path.write_text("required_name: example\n", encoding="utf-8")

    cfg = load_typed_yaml_cfg(config_path, ExampleCfg, config_name="example")

    assert cfg == ExampleCfg(required_name="example", count=2)


def test_render_typed_yaml_round_trips_without_resolving_template_tokens(tmp_path):
    config_path = tmp_path / "rendered.yaml"
    config_path.write_text(
        render_typed_yaml_cfg(ExampleCfg(required_name="{{host:server}}", count=4)),
        encoding="utf-8",
    )

    cfg = load_typed_yaml_cfg(config_path, ExampleCfg, config_name="example")

    assert cfg == ExampleCfg(required_name="{{host:server}}", count=4)


def test_load_typed_yaml_rejects_unknown_fields(tmp_path):
    config_path = tmp_path / "unknown.yaml"
    config_path.write_text("required_name: example\nunknown: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown"):
        load_typed_yaml_cfg(config_path, ExampleCfg, config_name="example")


def test_load_typed_yaml_rejects_incompatible_values(tmp_path):
    config_path = tmp_path / "wrong_type.yaml"
    config_path.write_text("required_name: example\ncount: not-an-integer\n", encoding="utf-8")

    with pytest.raises(ValueError, match="count"):
        load_typed_yaml_cfg(config_path, ExampleCfg, config_name="example")


def test_load_typed_yaml_rejects_missing_required_fields(tmp_path):
    config_path = tmp_path / "missing.yaml"
    config_path.write_text("count: 3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="required_name"):
        load_typed_yaml_cfg(config_path, ExampleCfg, config_name="example")

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import yaml

from isaaclab_arena.agentic_environment_generation.spec_io import (
    rejected_env_graph_spec_path,
    safe_filename_stem,
    write_rejected_env_graph_spec,
)


class TestSafeFilenameStem:
    def test_replaces_unsafe_chars_and_trims(self):
        assert safe_filename_stem("llm_gen maple/table") == "llm_gen_maple_table"
        assert safe_filename_stem("__weird..name__") == "weird..name"

    def test_empty_or_all_unsafe_falls_back(self):
        assert safe_filename_stem("") == "unnamed_env"
        assert safe_filename_stem("///") == "unnamed_env"


class TestWriteRejectedEnvGraphSpec:
    def test_writes_invalid_prefixed_yaml_with_traces(self, tmp_path):
        data = {"env_name": "droid pick place", "objects": [{"id": "bowl", "registry_name": "not_a_real_asset"}]}
        path = write_rejected_env_graph_spec(data, tmp_path, ("objects.0: Unknown asset registry_name",))
        assert path == rejected_env_graph_spec_path("droid pick place", tmp_path)
        assert path.name == "invalid_droid_pick_place.yaml"
        text = path.read_text(encoding="utf-8")
        assert "# objects.0: Unknown asset registry_name" in text
        assert yaml.safe_load(text) == data

    def test_writes_unnamed_spec_holding_unserializable_values(self, tmp_path):
        path = write_rejected_env_graph_spec({"objects": [{"id": object()}]}, tmp_path)
        assert path.name == "invalid_unnamed_env.yaml"
        assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8"))["objects"][0]["id"], str)

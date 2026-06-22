# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from isaaclab_arena.agentic_environment_generation.spec_io import safe_filename_stem


class TestSafeFilenameStem:
    def test_replaces_unsafe_chars_and_trims(self):
        assert safe_filename_stem("llm_gen maple/table") == "llm_gen_maple_table"
        assert safe_filename_stem("__weird..name__") == "weird..name"

    def test_empty_or_all_unsafe_falls_back(self):
        assert safe_filename_stem("") == "unnamed_env"
        assert safe_filename_stem("///") == "unnamed_env"

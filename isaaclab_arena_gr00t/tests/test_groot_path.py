# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

from isaaclab_arena_gr00t.utils import groot_path
from isaaclab_arena_gr00t.utils.io_utils import load_gr00t_modality_config_from_file


def _clear_gr00t_modules(monkeypatch):
    """Remove cached gr00t modules so import resolution exercises sys.path."""
    for module_name in list(sys.modules):
        if module_name == "gr00t" or module_name.startswith("gr00t."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)


def test_ensure_gr00t_importable_adds_submodule_root(monkeypatch):
    _clear_gr00t_modules(monkeypatch)
    monkeypatch.setattr(sys, "path", [path for path in sys.path if "submodules/Isaac-GR00T" not in path])
    monkeypatch.setattr(groot_path.importlib.util, "find_spec", lambda name: None if name == "gr00t" else None)

    gr00t_source_root = groot_path.ensure_gr00t_importable()

    assert gr00t_source_root is not None
    assert sys.path[0] == str(gr00t_source_root)


def test_builtin_gr00t_modality_config_loads_from_submodule(monkeypatch):
    _clear_gr00t_modules(monkeypatch)
    monkeypatch.setattr(sys, "path", [path for path in sys.path if "submodules/Isaac-GR00T" not in path])

    modality_config = load_gr00t_modality_config_from_file(
        modality_config_path=None,
        embodiment_tag="OXE_DROID",
    )

    assert set(modality_config) == {"video", "state", "action", "language"}

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import pkgutil

import isaaclab_arena_cosmos3

for _importer, _modname, _ispkg in pkgutil.iter_modules(isaaclab_arena_cosmos3.__path__):
    if not _ispkg:
        importlib.import_module(f"isaaclab_arena_cosmos3.{_modname}")

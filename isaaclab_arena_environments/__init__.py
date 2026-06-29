# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import pkgutil

import isaaclab_arena_environments

# libero_cameras is a perception-camera config helper (not a registered environment); it has
# top-level Isaac Lab configclass imports and is imported lazily by libero_object_packing's get_env
# (after the SimulationApp boots). Keep it out of the eager package-import sweep so importing this
# package pre-boot (e.g. eval_runner's module-top import) does not pull in the franka configclass chain.
_NON_ENVIRONMENT_MODULES = {"cli", "example_environment_base", "libero_cameras"}

for _importer, _modname, _ispkg in pkgutil.iter_modules(isaaclab_arena_environments.__path__):
    if not _ispkg and _modname not in _NON_ENVIRONMENT_MODULES:
        importlib.import_module(f"isaaclab_arena_environments.{_modname}")

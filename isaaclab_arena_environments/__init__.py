# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import pkgutil

import isaaclab_arena_environments

# libero_cameras / maple_cameras are perception-camera config helpers (not registered environments); they
# have top-level Isaac Lab configclass imports and are imported lazily by their environments' get_env
# (after the SimulationApp boots). Keep them out of the eager package-import sweep so importing this
# package pre-boot (e.g. eval_runner's module-top import) does not pull in the franka configclass chain.
_NON_ENVIRONMENT_MODULES = {"cli", "example_environment_base", "libero_cameras", "maple_cameras"}

for _importer, _modname, _ispkg in pkgutil.iter_modules(isaaclab_arena_environments.__path__):
    if not _ispkg and _modname not in _NON_ENVIRONMENT_MODULES:
        importlib.import_module(f"isaaclab_arena_environments.{_modname}")

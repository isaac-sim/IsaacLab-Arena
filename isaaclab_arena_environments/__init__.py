# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import pathlib
import pkgutil

import isaaclab_arena_environments

_NON_ENVIRONMENT_MODULES = {"cli", "example_environment_base", "mdp"}
_PACKAGE_DIRECTORY = pathlib.Path(__file__).parent

for _importer, _modname, _ispkg in pkgutil.iter_modules(isaaclab_arena_environments.__path__):
    _is_python_package = _ispkg and (_PACKAGE_DIRECTORY / _modname / "__init__.py").is_file()
    if (not _ispkg or _is_python_package) and _modname not in _NON_ENVIRONMENT_MODULES:
        importlib.import_module(f"isaaclab_arena_environments.{_modname}")

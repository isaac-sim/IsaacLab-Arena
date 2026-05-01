# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import pkgutil

import isaaclab_arena_environments

_NON_ENVIRONMENT_MODULES = {"cli", "example_environment_base"}

# Subpackages whose modules host @register_environment decorators we want
# to fire at import time. Keep this list explicit — recursing into every
# subpackage (e.g. ``mdp``, which does ``from isaaclab.envs.mdp import *``)
# pulls heavy isaaclab/scipy modules before Kit boots and trips the
# OpenBLAS-fork crash inside SimulationApp.
_RECURSE_SUBPACKAGES = {"llm_generated"}


def _is_trial_straggler(leaf: str) -> bool:
    """``avocadoPnPbowltable_t3`` ⇒ True; ``avocado_in_bowl`` ⇒ False.

    Auto-gen writes ``<env>_t<N>.py`` trial files alongside the canonical
    un-suffixed file; we register only the canonical one.
    """
    if "_t" not in leaf:
        return False
    suffix = leaf.rsplit("_t", 1)[-1]
    return suffix.isdigit()


# 1) Top-level env modules.
for _importer, _modname, _ispkg in pkgutil.iter_modules(isaaclab_arena_environments.__path__):
    if _ispkg or _modname in _NON_ENVIRONMENT_MODULES:
        continue
    importlib.import_module(f"isaaclab_arena_environments.{_modname}")

# 2) Explicitly-recursed subpackages (currently just ``llm_generated``).
for _subpkg in _RECURSE_SUBPACKAGES:
    try:
        _sub = importlib.import_module(f"isaaclab_arena_environments.{_subpkg}")
    except ModuleNotFoundError:
        continue
    for _importer, _modname, _ispkg in pkgutil.iter_modules(_sub.__path__):
        if _ispkg or _modname in _NON_ENVIRONMENT_MODULES or _is_trial_straggler(_modname):
            continue
        importlib.import_module(f"isaaclab_arena_environments.{_subpkg}.{_modname}")

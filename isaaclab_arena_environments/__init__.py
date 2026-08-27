# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import pkgutil

_NON_ENVIRONMENT_MODULES = {"cli", "example_environment_base"}
_ENVIRONMENTS_REGISTERED = False


def register_environments() -> None:
    """Import all first-party environment modules so their decorators register them."""
    global _ENVIRONMENTS_REGISTERED
    if _ENVIRONMENTS_REGISTERED:
        return

    for _importer, modname, ispkg in pkgutil.iter_modules(__path__):
        if not ispkg and modname not in _NON_ENVIRONMENT_MODULES:
            importlib.import_module(f"{__name__}.{modname}")
    _ENVIRONMENTS_REGISTERED = True

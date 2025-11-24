# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


def reload_arena_modules():
    """Reload all isaaclab_arena modules."""
    import importlib
    import os
    import sys

    # Clear Python's bytecode cache
    if hasattr(importlib, "invalidate_caches"):
        importlib.invalidate_caches()

    # Get all isaaclab_arena modules currently loaded
    isaaclab_arena_modules = [
        (name, module)
        for name, module in sys.modules.items()
        if name.startswith("isaaclab_arena") and module is not None
    ]

    if len(isaaclab_arena_modules) == 0:
        print("No isaaclab_arena modules found")
        return

    # We skip this step due to import issues in our asset registry
    # # Reload modules from bottom to top (deepest/leaf modules first)
    # isaaclab_arena_modules.sort(key=lambda x: x[0].count('.'), reverse=True)

    for module_name, module in isaaclab_arena_modules:
        try:
            # Delete the .pyc cache file to force recompilation from source
            module_cached = getattr(module, "__cached__", None)
            if module_cached and os.path.exists(module_cached):
                os.remove(module_cached)

            print(f"Reloading {module_name}")
            importlib.reload(module)

        except Exception as e:
            print(f"[WARNING] Failed to reload {module_name}: {e}")

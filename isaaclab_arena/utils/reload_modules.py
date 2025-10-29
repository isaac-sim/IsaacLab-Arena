# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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

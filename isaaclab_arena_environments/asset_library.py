# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Register environment-owned assets during normal environment-library discovery.

Only immediate environment-local ``assets.py`` modules are imported. Runtime helpers,
prebuild scripts, and other nested modules are not registration entry points.
"""

import importlib
from pathlib import Path

for _asset_module_path in sorted(Path(__file__).parent.glob("*/assets.py")):
    importlib.import_module(f"{__package__}.{_asset_module_path.parent.name}.assets")

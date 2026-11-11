# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import warnings

if not __debug__:
    warnings.warn(
        "Python is running with optimizations enabled (-O, -OO or PYTHONOPTIMIZE), which strips assert statements. "
        "Isaac Lab Arena relies on asserts to check user input, consider running without optimizations.",
        RuntimeWarning,
        stacklevel=2,
    )

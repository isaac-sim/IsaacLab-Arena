# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G1 Factory environments and local compatibility registrations.

This package intentionally keeps Factory-specific assets, tasks, reset events,
and termination predicates out of Arena core while still reusing Arena main's
environment builder, registries, embodiment, and runner stack.
"""

# Importing this package registers the local Factory assets with Arena's
# AssetRegistry before any Factory environment attempts to resolve them.
from . import assets as _assets  # noqa: F401

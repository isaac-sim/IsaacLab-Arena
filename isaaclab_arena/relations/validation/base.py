# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import ClassVar


class PlacementValidator(ABC):
    """Shared identity and stage of a placement check."""

    check: ClassVar[str]
    """Unique check name within its validation stage."""
    stage: ClassVar[str]
    """The stage whose poses this check evaluates: pre_physics or post_physics."""

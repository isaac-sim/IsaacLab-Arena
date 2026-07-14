# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase


class CameraProfileBase(ABC):
    """Named reviewed camera setup that can be applied to a compatible embodiment."""

    name: ClassVar[str]
    description: ClassVar[str]
    compatible_embodiments: ClassVar[frozenset[str]]

    @classmethod
    def assert_compatible(cls, embodiment_registry_name: str) -> None:
        assert embodiment_registry_name in cls.compatible_embodiments, (
            f"Camera profile '{cls.name}' is not compatible with embodiment "
            f"'{embodiment_registry_name}'. Compatible embodiments: {sorted(cls.compatible_embodiments)}"
        )

    @classmethod
    @abstractmethod
    def apply(cls, embodiment: EmbodimentBase) -> None:
        """Apply this camera profile to an already constructed embodiment."""

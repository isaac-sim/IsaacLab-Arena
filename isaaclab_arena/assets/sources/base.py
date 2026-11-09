# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Common and path-backed USD source types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class UsdSource(ABC):
    """Resolve a USD asset without coupling consumers to its acquisition mechanism."""

    @abstractmethod
    def identity(self) -> str:
        """Return a stable identity for the source."""

    @abstractmethod
    def resolve(self) -> str:
        """Return a usable local or remote USD path."""


@dataclass(frozen=True)
class UsdPathSource(UsdSource):
    """Use an already-readable filesystem or URI path."""

    path: str

    def __post_init__(self) -> None:
        assert self.path, "USD path must not be empty"

    def identity(self) -> str:
        """Return the path as this source's stable identity."""
        return self.path

    def resolve(self) -> str:
        """Return the path without performing acquisition."""
        return self.path

# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Type

from isaaclab_arena.remote_policy.server_side_policy import ServerSidePolicy


@dataclass(frozen=True)
class PolicyEntry:
    policy_type: str
    entry_point: str  # "module_path:ClassName"


class PolicyRegistry:
    def __init__(self) -> None:
        self._entries: Dict[str, PolicyEntry] = {}

    def register(self, policy_type: str, entry_point: str) -> None:
        if policy_type in self._entries:
            raise ValueError(f"Policy type {policy_type!r} already registered")
        if ":" not in entry_point:
            raise ValueError(
                f"Invalid entry_point {entry_point!r} for policy_type={policy_type!r} "
                "(expected 'module_path:ClassName')"
            )
        self._entries[policy_type] = PolicyEntry(policy_type, entry_point)

    def available_policy_types(self) -> List[str]:
        return sorted(self._entries.keys())

    def resolve_class(self, policy_type: str) -> Type[ServerSidePolicy]:
        if policy_type not in self._entries:
            raise ValueError(
                f"Unknown policy_type={policy_type!r}. "
                f"Available options: {self.available_policy_types()}"
            )

        entry = self._entries[policy_type]
        module_path, class_name = entry.entry_point.split(":", 1)

        try:
            module = __import__(module_path, fromlist=[class_name])
        except ImportError as exc:
            raise ImportError(
                f"Failed to import module '{module_path}' for policy_type={policy_type!r}. "
                "This usually means the corresponding policy package is not installed "
                "in the current server environment."
            ) from exc

        try:
            cls = getattr(module, class_name)
        except AttributeError as exc:
            raise ImportError(
                f"Module '{module_path}' does not define class '{class_name}' "
                f"for policy_type={policy_type!r}."
            ) from exc

        if not issubclass(cls, ServerSidePolicy):
            raise TypeError(
                f"Resolved class '{class_name}' from '{module_path}' is not a ServerSidePolicy "
                f"subclass (policy_type={policy_type!r})."
            )
        return cls


policy_registry = PolicyRegistry()

# Built-in registrations
policy_registry.register(
    "gr00t_closedloop",
    "isaaclab_arena_gr00t.gr00t_remote_policy:Gr00tRemoteServerSidePolicy",
)


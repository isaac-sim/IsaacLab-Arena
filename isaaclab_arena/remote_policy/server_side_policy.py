# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from .action_protocol import ActionMode, ActionProtocol


class ServerSidePolicy(ABC):
    """Server-side policy interface."""

    def __init__(self) -> None:
        self._protocol: ActionProtocol | None = None
        self._task_description: str | None = None

    # ------------ protocol API ------------

    @abstractmethod
    def _build_protocol(self) -> ActionProtocol:
        """Subclasses must build and return an ActionProtocol instance."""

    @property
    def protocol(self) -> ActionProtocol:
        if self._protocol is None:
            self._protocol = self._build_protocol()
            if self._protocol.mode is ActionMode.NONE:
                raise ValueError(
                    f"{self.__class__.__name__} built an ActionProtocol "
                    f"with mode=NONE, which is not allowed."
                )
        return self._protocol

    def get_init_info(self, requested_action_mode: str) -> dict[str, Any]:
        """Default handshake using the configured ActionProtocol."""
        proto = self.protocol

        try:
            requested_mode_enum = ActionMode(requested_action_mode)
        except ValueError:
            return {
                "status": "invalid_action_mode",
                "message": f"Requested action_mode={requested_action_mode!r} is invalid.",
            }

        if requested_mode_enum is not proto.mode:
            return {
                "status": "unsupported_action_mode",
                "message": (
                    f"Requested action_mode={requested_mode_enum.value!r} "
                    f"is not supported by this policy. "
                    f"Supported: {proto.mode.value!r}."
                ),
            }

        return {
            "status": "success",
            "config": proto.to_dict(),
        }

    # ------------ core abstract methods ------------

    @abstractmethod
    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        ...

    @abstractmethod
    def reset(
        self,
        env_ids: list[int] | None = None,
        reset_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    def set_task_description(self, task_description: str | None) -> dict[str, Any]:
        ...

    # ------------ shared helper ------------

    def unpack_observation(self, flat_obs: Dict[str, Any]) -> Dict[str, Any]:
        nested: Dict[str, Any] = {}
        for key_path, value in flat_obs.items():
            cur = nested
            parts = key_path.split(".")
            for k in parts[:-1]:
                cur = cur.setdefault(k, {})
            cur[parts[-1]] = value
        return nested

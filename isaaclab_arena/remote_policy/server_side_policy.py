# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026,
# The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import Any

from isaaclab_arena.remote_policy.action_protocol import ActionMode, ActionProtocol


class ServerSidePolicy(ABC):
    """Base class for server-side remote policies.

    This class defines:
      * The protocol- and handshake-related API that the PolicyServer relies on.
      * A minimal configuration hook via ``config_class`` and ``from_dict``.
      * A CLI construction pattern via ``add_args_to_parser`` and ``from_args``,
        mirroring the design of :class:`isaaclab_arena.policy.policy_base.PolicyBase`
        on the client side.

    Concrete server-side policies (e.g. GR00T-based ones) should:
      * Implement ``_build_protocol()`` and the core RPC methods.
      * Optionally define a dataclass as ``config_class``.
      * Implement ``add_args_to_parser(parser)`` and ``from_args(args)``
        so they can be instantiated directly from command-line arguments.
    """

    # Optional: subclasses can define this to enable from_dict()
    config_class: type | None = None

    def __init__(self, config: Any | None = None) -> None:
        """Base constructor for server-side policies.

        Args:
            config: Optional configuration object (for example, a dataclass
                instance). Subclasses are free to interpret this as needed.
        """
        self.config = config
        self._protocol: ActionProtocol | None = None
        self._task_description: str | None = None

    # ------------------------------------------------------------------
    # Config helpers (mirroring PolicyBase.from_dict)
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> ServerSidePolicy:
        """Create a policy instance from a configuration dictionary.

        Path: dict -> ConfigDataclass -> Policy instance

        This mirrors :meth:`PolicyBase.from_dict` on the client side.
        """
        if cls.config_class is None:
            raise NotImplementedError(f"{cls.__name__} must define 'config_class' to use from_dict().")

        config = cls.config_class(**config_dict)  # type: ignore[misc]
        return cls(config)  # type: ignore[call-arg]

    # ------------------------------------------------------------------
    # Protocol / handshake API
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_protocol(self) -> ActionProtocol:
        """Subclasses must build and return an ActionProtocol instance."""
        raise NotImplementedError

    @property
    def protocol(self) -> ActionProtocol:
        """Return the ActionProtocol associated with this policy.

        The protocol is lazily constructed on first access via ``_build_protocol()``.
        """
        if self._protocol is None:
            self._protocol = self._build_protocol()
        if self._protocol.mode is None:
            raise ValueError(f"{self.__class__.__name__} has an ActionProtocol with mode=None, which is not allowed.")
        return self._protocol

    def get_init_info(self, requested_action_mode: str) -> dict[str, Any]:
        """Handle the initial handshake with the client.

        Checks that the requested action mode is valid and supported by
        this policy's ActionProtocol, and returns either an error status
        or the protocol configuration as a plain dictionary.
        """
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
                    "is not supported by this policy. "
                    f"Supported: {proto.mode.value!r}."
                ),
            }

        return {
            "status": "success",
            "config": proto.to_dict(),
        }

    # ------------------------------------------------------------------
    # Core RPC methods (to be used by PolicyServer)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_action(
        self,
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute one or more actions given an observation payload.

        Args:
            observation: Flat observation dictionary received from the client.

        Returns:
            A dictionary that must contain at least an ``"action"`` entry
            whose structure is compatible with the negotiated ActionProtocol.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the policy state.

        Subclasses may override this if they maintain per-environment or
        global state that needs to be cleared between episodes.
        """
        ...

    def set_task_description(
        self,
        task_description: str | None,
    ) -> dict[str, Any]:
        """Set the task description and return a small status/config payload.

        The default implementation stores the description locally and
        echoes it back. Subclasses can override this to perform additional
        updates or validation.
        """
        self._task_description = task_description
        return {"task_description": self._task_description or ""}

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def unpack_observation(self, flat_obs: dict[str, Any]) -> dict[str, Any]:
        """Convert a flat dotted-key observation dict into a nested dict.

        For example, a key ``"camera_obs.pov.rgb"`` becomes
        ``nested["camera_obs"]["pov"]["rgb"]``.
        """
        nested: dict[str, Any] = {}
        for key_path, value in flat_obs.items():
            cur = nested
            parts = key_path.split(".")
            for k in parts[:-1]:
                cur = cur.setdefault(k, {})
            cur[parts[-1]] = value
        return nested

    # ------------------------------------------------------------------
    # CLI helpers (to mirror PolicyBase.add_args_to_parser / from_args)
    # ------------------------------------------------------------------

    @staticmethod
    @abstractmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add policy-specific CLI arguments to the parser.

        Server-side policies are expected to implement this so that
        :mod:`remote_policy_server_runner` can delegate CLI argument
        definitions to the selected policy class.
        """
        raise NotImplementedError("ServerSidePolicy subclasses must implement add_args_to_parser().")

    @staticmethod
    @abstractmethod
    def from_args(args: argparse.Namespace) -> ServerSidePolicy:
        """Construct a server-side policy instance from CLI arguments.

        This mirrors the ``from_args(args)`` pattern used by client-side
        policies deriving from :class:`PolicyBase`.
        """
        raise NotImplementedError("ServerSidePolicy subclasses must implement from_args(args).")

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClientState:
    """Per-client state managed by the PolicyServer.

    Each connected client gets its own ``ClientState`` instance, created when
    the client completes ``initialize_session()``. The server passes a reference to this
    object into every ``ServerSidePolicy`` method so that policies can store
    per-client / per-env data without resorting to global singletons.

    Built-in fields:
        num_envs: Number of environments this client is running.
        instructions: Per-env task description strings (length ``num_envs``).

    Extensibility:
        Use ``metadata`` for policy-specific per-client data (e.g. a model
        handle, a session token).  Use :meth:`register_per_env_field` /
        :meth:`get_per_env_field` for per-env data (e.g. image histories,
        lidar buffers).  This avoids hardcoding policy-specific fields into
        the dataclass while still providing a typed, validated interface.

    Example::

        # In your ServerSidePolicy subclass:
        def get_init_info(self, ...) -> dict:
            resp = super().get_init_info(...)
            # Register a per-env image history list during init
            client_state.register_per_env_field("image_histories", default_factory=list)
            return resp

        def get_action(self, obs, *, client_state, env_ids, **kw):
            histories = client_state.get_per_env_field("image_histories")
            # histories is a list of length num_envs
    """

    num_envs: int
    instructions: list[str | None] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Internal storage for registered per-env fields.
    # Maps field_name -> list of length num_envs.
    _per_env_fields: dict[str, list] = field(default_factory=dict, repr=False)

    @classmethod
    def create(cls, num_envs: int) -> ClientState:
        """Create a new ``ClientState`` with empty per-env arrays."""
        return cls(
            num_envs=num_envs,
            instructions=[None] * num_envs,
            metadata={},
        )

    # ------------------------------------------------------------------
    # Per-env field helpers
    # ------------------------------------------------------------------

    def register_per_env_field(
        self,
        name: str,
        *,
        default_factory: Any = None,
        default: Any = None,
    ) -> list:
        """Register a custom per-env field and return the list.

        Args:
            name: Field name (e.g. ``"image_histories"``).
            default_factory: Callable that returns the default value for each
                env slot (e.g. ``list`` for empty lists).  Takes precedence
                over *default*.
            default: Static default value copied to each env slot.  Must be
                immutable (e.g. ``None``, ``0``, ``""``) — mutable defaults
                like ``[]`` or ``{}`` will alias across slots.  Use
                *default_factory* for mutable values.

        Returns:
            The per-env list (length ``num_envs``).  If the field was
            already registered, returns the existing list (idempotent).
        """
        if name in self._per_env_fields:
            return self._per_env_fields[name]

        if default_factory is not None:
            data = [default_factory() for _ in range(self.num_envs)]
        else:
            import copy
            if default is not None and isinstance(default, (list, dict, set)):
                data = [copy.deepcopy(default) for _ in range(self.num_envs)]
            else:
                data = [default] * self.num_envs
        self._per_env_fields[name] = data
        return data

    def get_per_env_field(self, name: str) -> list:
        """Get a registered per-env field by name.

        Raises:
            KeyError: If the field has not been registered.
        """
        if name not in self._per_env_fields:
            raise KeyError(
                f"Per-env field {name!r} not registered. "
                f"Available: {list(self._per_env_fields.keys())}"
            )
        return self._per_env_fields[name]

    def has_per_env_field(self, name: str) -> bool:
        """Check if a per-env field has been registered."""
        return name in self._per_env_fields
